"""Replay-based same-state proxy and open-loop world-model diagnostics."""

from __future__ import annotations

from collections import Counter

import mujoco
import torch

from supermimic.envs.joint_imagination import (
  body_collision_proxy,
  foot_clearance_proxy,
  foot_contact_proxy,
  foot_slip_proxy,
  imagined_done,
  landing_speed_proxy,
)


def _mean(x):
  return float(x.float().mean())


def _correlation(x, y):
  x, y = x.float().flatten(), y.float().flatten()
  x, y = x - x.mean(), y - y.mean()
  den = x.norm() * y.norm()
  return None if den < 1e-12 else float((x * y).sum() / den)


def slip_contact_breakdown(pos, vel, contact, command, height_threshold=0.025, width=0.01):
  """Explain proxy costs separately for contact and non-contact moving feet."""
  active = (command[:, :2].norm(dim=-1) + command[:, 2].abs() > 0.05)
  actual = contact.bool()
  probability = torch.sigmoid((height_threshold - pos[..., 2]) / width)
  speed = vel[..., :2].norm(dim=-1)
  contribution = probability * speed.square() * active[:, None]
  result = {}
  for name, mask in (("contact", actual), ("no_contact", ~actual)):
    mask = mask & active[:, None]
    result[name] = {
      "foot_samples": int(mask.sum()),
      "mean_horizontal_speed_mps": _mean(speed[mask]) if mask.any() else None,
      "mean_proxy_contact_probability": _mean(probability[mask]) if mask.any() else None,
      "proxy_slip_per_environment": float(contribution[mask].sum() / len(command)),
    }
  return result


@torch.no_grad()
def compare_slip_candidates(buffer):
  """Compare fixed candidates on disjoint environment IDs, without replacement."""
  n, envs = buffer.num_transitions, buffer.num_envs
  if envs < 2:
    return {"available": False, "reason": "requires at least two environments"}
  base = buffer._step if n == buffer.buffer_size else 0
  slots = (base + torch.arange(n, device=buffer.device)) % buffer.buffer_size
  replay = buffer._storage[:, slots]
  pos = replay["foot_site_pos"].reshape(envs, n, 2, 3)
  vel = replay["foot_site_vel"].reshape(envs, n, 2, 3)
  contact = replay["foot_contact"]
  command = replay["command"]
  active = (command[..., :2].norm(dim=-1) + command[..., 2].abs() > 0.05)
  speed_sq = vel[..., :2].square().sum(-1) * active.unsqueeze(-1)
  real = (contact * speed_sq).sum(-1)
  split = envs // 2
  out = {"available": True, "split": "first versus second environment IDs", "candidates": []}
  for height, width in ((0.025, 0.01), (0.015, 0.003)):
    p = torch.sigmoid((height - pos[..., 2]) / width)
    predicted = (p * speed_sq).sum(-1)
    sections = {}
    for name, part in (("first_half", slice(None, split)), ("second_half", slice(split, None))):
      sections[name] = {
        "environments": len(range(*part.indices(envs))),
        "states": real[part].numel(),
        "slip_absolute_error": _mean((predicted[part] - real[part]).abs()),
        "contact_brier": _mean((p[part] - contact[part]).square()),
        "noncontact_slip_per_environment_step": _mean((p[part] * speed_sq[part] * (1 - contact[part])).sum(-1)),
        "missed_contact_slip_per_environment_step": _mean(((1 - p[part]) * speed_sq[part] * contact[part]).sum(-1)),
      }
    out["candidates"].append({"height_m": height, "width_m": width, "groups": sections})
  return out


@torch.no_grad()
def inspect_collision_geometry(runner, sample_per_class=128, seed=0):
  """Reconstruct current geometry; historical force labels cannot be recovered."""
  if sample_per_class < 1:
    raise ValueError("sample_per_class must be positive")
  buffer = runner.collector.buffer
  n = buffer.num_transitions
  base = buffer._step if n == buffer.buffer_size else 0
  slots = (base + torch.arange(n, device=buffer.device)) % buffer.buffer_size
  replay = buffer._storage[:, slots].reshape(-1)
  states = replay["state"].cpu()
  positive = (replay["self_collision_cost"] > 0).cpu()
  model = runner.env.sim.mj_model
  robot = runner.env.scene["robot"]
  root = int(robot.indexing.root_body_id)
  root_joint = int(model.body_jntadr[root])
  if int(model.jnt_type[root_joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
    raise ValueError("Expected a free joint on the robot root")
  root_qpos = int(model.jnt_qposadr[root_joint])
  joint_qpos = [int(model.jnt_qposadr[int(i)]) for i in robot.indexing.joint_ids]
  robot_bodies = set(int(i) for i in robot.indexing.body_ids)
  data = mujoco.MjData(model)
  generator = torch.Generator(device="cpu").manual_seed(seed)
  results = {}
  for label, mask in (("positive_history", positive), ("negative_history", ~positive)):
    available = mask.nonzero().flatten()
    indices = available[torch.randperm(len(available), generator=generator)[:sample_per_class]]
    pairs = Counter()
    any_pair = 0
    for idx in indices.tolist():
      full = states[idx].numpy()
      mujoco.mj_resetData(model, data)
      data.qpos[root_qpos:root_qpos + 7] = full[:7]
      data.qpos[joint_qpos] = full[7:7 + len(joint_qpos)]
      mujoco.mj_forward(model, data)
      row_pairs = set()
      for j in range(data.ncon):
        contact = data.contact[j]
        g1, g2 = int(contact.geom[0]), int(contact.geom[1])
        if int(model.geom_bodyid[g1]) not in robot_bodies or int(model.geom_bodyid[g2]) not in robot_bodies:
          continue
        names = [model.geom(g).name or f"geom_{g}" for g in (g1, g2)]
        row_pairs.add(tuple(sorted(names)))
      any_pair += bool(row_pairs)
      pairs.update(row_pairs)
    results[label] = {
      "available_states": len(available), "sampled_states": len(indices),
      "states_with_current_self_contact": any_pair,
      "top_current_geom_pairs": [
        {"geom_a": a, "geom_b": b, "state_count": count}
        for (a, b), count in pairs.most_common(20)
      ],
    }
  return results


@torch.no_grad()
def evaluate_joint_replay(
  runner, horizon: int = 8, batch_size: int = 128, same_state_limit: int = 32768,
  *, strict_disturbances: bool = False,
):
  """Compare on replay samples. Open-loop windows exclude terminal interiors.

  One-step termination uses a separate sample including terminal transitions;
  timeouts are excluded. Same-state proxies compare different physical units.
  """
  if horizon < 1 or batch_size < 1 or same_state_limit < 1:
    raise ValueError("horizon, batch_size and same_state_limit must be positive")
  buffer, wm = runner.collector.buffer, runner.wm
  if horizon + 1 > buffer.num_transitions:
    raise ValueError("Replay needs at least horizon + 1 collected frames")
  try:
    batch = buffer.sample(horizon + 1, batch_size, boundary_field="wm_invalid")
    disturbance_filter = "excluded_marked_transitions"
  except RuntimeError as exc:
    if not str(exc).startswith("no valid sequences"):
      raise
    if strict_disturbances:
      raise ValueError("No evaluation windows without resets or marked pushes; collect longer trajectories") from exc
    batch = buffer.sample(horizon + 1, batch_size)
    disturbance_filter = "unavailable_or_insufficient_unmarked_windows"
  one, _ = buffer.sample_causal(1, batch_size)
  one = one[:, 0]
  available = buffer.num_envs * buffer.num_transitions
  count = min(same_state_limit, available)
  selected = torch.randperm(available, device=buffer.device)[:count]
  env_ids = torch.div(selected, buffer.num_transitions, rounding_mode="floor")
  ages = selected % buffer.num_transitions
  base = buffer._step if buffer.num_transitions == buffer.buffer_size else 0
  slots = (base + ages) % buffer.buffer_size
  current = buffer._storage[env_ids, slots]
  pos = current["foot_site_pos"].reshape(-1, 2, 3)
  vel = current["foot_site_vel"].reshape(-1, 2, 3)
  command = current["command"]
  active = (command[:, :2].norm(dim=-1) + command[:, 2].abs() > 0.05).float()
  contact = current["foot_contact"]
  contact_prob = foot_contact_proxy(pos[..., 2])
  real_slip = active * (contact * vel[..., :2].square().sum(-1)).sum(-1)
  height, width = runner.cfg.policy.slip_contact_height, runner.cfg.policy.slip_contact_width
  proxy_slip = foot_slip_proxy(pos, vel, command, height, width)
  real_clearance = active * (
    (current["foot_height"] - 0.1).abs() * vel[..., :2].norm(dim=-1)
  ).sum(-1)
  real_landing = active * (current["foot_force"] * current["foot_first_contact"]).sum(-1)
  proxy_landing = landing_speed_proxy(pos, vel, command)
  full = current["state"]
  body_pos, body_quat = runner.fk(full[:, :3], full[:, 3:7], full[:, 7:36])
  collision = body_collision_proxy(body_pos, body_quat)
  capsule_separation = runner.capsule_collision(full[:, :3], full[:, 3:7], full[:, 7:36])
  capsule_hit = capsule_separation < 0
  capsule_any = capsule_hit.any(-1)
  actual_collision = current["self_collision_cost"]
  same_state = {
    "samples": count,
    "available_unique_states": available,
    "sampling": "without_replacement",
    "contact_disagreement": _mean((contact_prob >= 0.5) != contact.bool()),
    "contact_real_fraction": _mean(contact),
    "foot_height_mae_m": _mean((pos[..., 2] - current["foot_height"]).abs()),
    "slip_real_mean": _mean(real_slip),
    "slip_proxy_parameters_m": {"height": height, "width": width},
    "slip_proxy_mean": _mean(proxy_slip),
    "slip_absolute_error": _mean((real_slip - proxy_slip).abs()),
    "slip_by_real_contact": slip_contact_breakdown(pos, vel, contact, command, height, width),
    "clearance_real_mean": _mean(real_clearance),
    "clearance_proxy_mean": _mean(foot_clearance_proxy(pos, vel, command)),
    "clearance_absolute_error": _mean((real_clearance - foot_clearance_proxy(pos, vel, command)).abs()),
    "landing_event_count": int((current["foot_first_contact"] * active[:, None]).sum()),
    "landing_force_real_mean_n": _mean(real_landing),
    "landing_speed_proxy_mean_m2ps2": _mean(proxy_landing),
    "landing_force_speed_correlation": _correlation(real_landing, proxy_landing),
    "self_collision_real_positive_count": int((actual_collision > 0).sum()),
    "body_proxy_positive_count": int((collision > 0).sum()),
    "collision_both_positive_count": int(((collision > 0) & (actual_collision > 0)).sum()),
    "self_collision_cost_mean": _mean(actual_collision),
    "body_proxy_mean_m2": _mean(collision),
    "capsule_collision": {
      "geometry": "selected same-side hand/wrist vs hip/thigh capsules",
      "positive_count": int(capsule_any.sum()),
      "history_positive_and_capsule_count": int((capsule_any & (actual_collision > 0)).sum()),
      "history_positive_capsule_negative_count": int((~capsule_any & (actual_collision > 0)).sum()),
      "history_negative_capsule_positive_count": int((capsule_any & (actual_collision <= 0)).sum()),
      "pair_counts": {
        f"{a} / {b}": int(capsule_hit[:, i].sum())
        for i, (a, b) in enumerate(runner.capsule_collision.pair_names)
      },
    },
  }
  rep = wm.representation
  state = rep.represent(batch["state"][:, 0])
  aux = rep.extract_aux_state(batch["state"][:, 0])
  transition = (batch["foot_contact"][:, 1:] != batch["foot_contact"][:, :-1]).any(dim=(-1, -2))
  standing = batch["standing"][:, 0]
  masks = {
    "standing_at_start": standing,
    "moving_at_start": ~standing,
    "contact_changed_within_window": transition,
    "contact_unchanged_within_window": ~transition,
    "tilted_over_30deg_at_start": state[:, 3] > -0.8660254,
  }
  context_delta = (batch["sim_ctx"] - batch["sim_ctx"][:, :1]).abs().amax(dim=(-1, -2))
  groups = {name: {"windows": int(mask.sum()), "steps": []} for name, mask in masks.items()}
  rollout = []
  for t in range(horizon):
    aux, state = wm.dynamics(state, batch["action"][:, t], aux, batch["sim_ctx"][:, t])
    truth = batch["state"][:, t + 1]
    actual = rep.represent(truth)
    ppos, _ = runner.fk(aux[:, :3], aux[:, 3:7], state[:, 10:39])
    tpos, _ = runner.fk(truth[:, :3], truth[:, 3:7], truth[:, 7:36])
    dot = (aux[:, 3:7] * truth[:, 3:7]).sum(-1).abs().clamp(max=1)
    ankle_error = (ppos[:, 2:4] - tpos[:, 2:4]).norm(dim=-1).mean(-1)
    velocity_error = (state[:, 4:7] - actual[:, 4:7]).norm(dim=-1)
    rollout.append({
      "step": t + 1,
      "root_position_error_m": _mean((aux[:, :3] - truth[:, :3]).norm(dim=-1)),
      "root_orientation_error_rad": _mean(2 * dot.acos()),
      "joint_position_mae_rad": _mean((state[:, 10:39] - actual[:, 10:39]).abs()),
      "joint_velocity_mae_radps": _mean((state[:, 39:] - actual[:, 39:]).abs()),
      "ankle_position_error_m": _mean(ankle_error),
      "linear_velocity_error_mps": _mean(velocity_error),
    })
    if t + 1 in (1, 4, 8, 16, 32, horizon):
      for name, mask in masks.items():
        groups[name]["steps"].append({
          "step": t + 1,
          "ankle_position_error_m": _mean(ankle_error[mask]) if mask.any() else None,
          "linear_velocity_error_mps": _mean(velocity_error[mask]) if mask.any() else None,
        })
  one_state = rep.represent(one["state"])
  one_aux = rep.extract_aux_state(one["state"])
  pred_aux, pred_state = wm.dynamics(one_state, one["action"], one_aux, one["sim_ctx"])
  probability = imagined_done(pred_aux, pred_state)
  valid = ~one["truncated"]
  if disturbance_filter == "excluded_marked_transitions":
    valid &= ~one["wm_invalid"]
  predicted = probability >= 0.5
  observed = one["terminated"]
  termination = {
    "valid_samples": int(valid.sum()),
    "actual_terminated": int((observed & valid).sum()),
    "predicted_terminated": int((predicted & valid).sum()),
    "false_positive": int((predicted & ~observed & valid).sum()),
    "false_negative": int((~predicted & observed & valid).sum()),
    "mean_predicted_probability": _mean(probability[valid]) if valid.any() else None,
  }
  return {
    "same_state": same_state,
    "slip_candidate_comparison": compare_slip_candidates(buffer),
    "open_loop_sampling": {
      "samples": batch_size, "sampling": "with_replacement",
      "disturbance_filter": disturbance_filter,
      "context_changed_windows": int((context_delta > 1e-6).sum()),
      "max_context_change": float(context_delta.max()),
    },
    "open_loop": rollout,
    "open_loop_by_scenario": groups,
    "one_step_termination": termination,
  }
