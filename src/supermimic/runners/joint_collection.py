"""Collect reference-free joint-task transitions for world-model training."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.sim_context import SimContext


def full_state(env) -> torch.Tensor:
  """GenCoord full-state layout, with root position relative to env origin."""
  data = env.scene["robot"].data
  return torch.cat(
    (
      data.root_link_pos_w - env.scene.env_origins,
      data.root_link_quat_w,
      data.joint_pos,
      data.root_link_lin_vel_w,
      data.root_link_ang_vel_w,
      data.joint_vel,
    ),
    dim=-1,
  )


class JointReplayCollector:
  """Record pre-step state/action/command, plus the transition's done flags.

  The buffer's sequence sampler excludes episode boundaries, so the next row's
  state is a valid next state only for nonterminal transitions. Foot anchors
  are stored relative to env origin, matching the stored root position.
  """

  def __init__(self, env, buffer_size: int):
    self.env = env
    self.sim_context = SimContext.from_env(env)
    state = full_state(env)
    action_dim = env.action_manager.total_action_dim
    obs_dim = env.observation_manager.compute_group("actor").shape[-1]
    self.foot_site_ids, names = env.scene["robot"].find_sites(
      ("left_foot", "right_foot"), preserve_order=True
    )
    if names != ["left_foot", "right_foot"]:
      raise ValueError(f"Expected two foot sites, found {names}")
    self.buffer = ReplayBuffer(
      schema={
        "state": FieldSpec(state.shape[-1]),
        "action": FieldSpec(action_dim),
        "prev_action": FieldSpec(action_dim),
        "actor_obs": FieldSpec(obs_dim),
        "command": FieldSpec(17),
        "hand_transition_start": FieldSpec(29),
        "hand_transition_target": FieldSpec(29),
        "hand_transition_elapsed": FieldSpec(),
        "hand_command_remaining": FieldSpec(),
        "twist_command_remaining": FieldSpec(),
        "foot_anchor": FieldSpec(6),
        "standing": FieldSpec(dtype=torch.bool),
        "foot_site_pos": FieldSpec(6),
        "foot_site_vel": FieldSpec(6),
        "foot_height": FieldSpec(2),
        "foot_contact": FieldSpec(2),
        "foot_first_contact": FieldSpec(2),
        "foot_force": FieldSpec(2),
        "self_collision_cost": FieldSpec(),
        "wm_invalid": FieldSpec(dtype=torch.bool),
        "sim_ctx": FieldSpec(self.sim_context.output_dim),
        "terminated": FieldSpec(dtype=torch.bool),
        "truncated": FieldSpec(dtype=torch.bool),
      },
      buffer_size=buffer_size,
      num_envs=env.num_envs,
      device=env.device,
    )

  def step(self, action: torch.Tensor, actor_obs: torch.Tensor):
    env = self.env
    hand = env.command_manager.get_term("hands")
    robot = env.scene["robot"]
    sensor = env.scene["feet_ground_contact"]
    contact_data = sensor.data
    collision = env.scene["self_collision"].data
    manager = env.event_manager
    interval_names = manager._mode_term_names.get("interval", [])
    if "push_robot" in interval_names:
      index = interval_names.index("push_robot")
      time_left = manager._interval_term_time_left[index]
      push_event = (time_left - env.step_dt < 1e-6).expand(env.num_envs).clone()
    else:
      push_event = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.last_push_event = push_event
    if collision.force_history is not None:
      collision_cost = (collision.force_history.norm(dim=-1) > 10.0).any(dim=1).sum(dim=-1).float()
    else:
      collision_cost = collision.found.sum(dim=-1).float()
    fields = {
      "state": full_state(env).detach().clone(),
      "action": action.detach().clone(),
      "prev_action": env.action_manager.action.detach().clone(),
      "actor_obs": actor_obs.detach().clone(),
      "command": torch.cat(
        (env.command_manager.get_command("twist"), hand.command), dim=-1
      )
      .detach()
      .clone(),
      "hand_transition_start": hand.transition_start.detach().clone(),
      "hand_transition_target": hand.transition_target.detach().clone(),
      "hand_transition_elapsed": (hand.transition_time_left_at_start - hand.time_left)
      .detach()
      .clone(),
      "hand_command_remaining": hand.time_left.detach().clone(),
      "twist_command_remaining": env.command_manager.get_term("twist")
      .time_left.detach()
      .clone(),
      "foot_anchor": (hand.foot_anchor - env.scene.env_origins[:, None, :])
      .flatten(start_dim=1)
      .detach()
      .clone(),
      "standing": hand.was_standing.detach().clone(),
      "foot_site_pos": (robot.data.site_pos_w[:, self.foot_site_ids] - env.scene.env_origins[:, None, :]).flatten(start_dim=1).detach().clone(),
      "foot_site_vel": robot.data.site_lin_vel_w[:, self.foot_site_ids].flatten(start_dim=1).detach().clone(),
      "foot_height": env.scene["foot_height_scan"].data.heights.detach().clone(),
      "foot_contact": (contact_data.found > 0).float().detach().clone(),
      "foot_first_contact": sensor.compute_first_contact(dt=env.step_dt).float().detach().clone(),
      "foot_force": contact_data.force.norm(dim=-1).detach().clone(),
      "self_collision_cost": collision_cost.detach().clone(),
      "wm_invalid": push_event,
      "sim_ctx": self.sim_context.read(env).detach().clone(),
    }
    obs, reward, terminated, truncated, info = env.step(action)
    fields["terminated"] = terminated.detach().clone()
    fields["truncated"] = truncated.detach().clone()
    self.buffer.insert(TensorDict(fields, batch_size=[env.num_envs]))
    return obs, reward, terminated, truncated, info
