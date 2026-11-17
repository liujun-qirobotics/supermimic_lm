"""Multi-step joint-task policy optimization through the learned dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict

from supermimic.envs.joint_imagination import joint_imagination_step
from supermimic.learners.policy_learner import gae_returns
from supermimic.models import build_mlp
from supermimic.tasks.joint_command import interpolate_reference_joints


def sensor_from_real_obs(actor_obs: torch.Tensor) -> torch.Tensor:
  """Velocity-task actor order -> SuperMimic 93-D proprio order for G1."""
  if actor_obs.shape[-1] != 127:
    raise ValueError("Expected the 127-D SuperMimic-Joint-G1 actor observation")
  return torch.cat((actor_obs[..., 6:9], actor_obs[..., 3:6], actor_obs[..., 9:96]), -1)


@dataclass
class JointPolicyCfg:
  horizon: int = 32
  batch_size: int = 128
  gamma: float = 0.99
  lam: float = 0.95
  actor_lr: float = 2e-4
  critic_lr: float = 2e-4
  grad_clip: float = 1.0
  clearance_weight: float = 2.0
  landing_weight: float = 0.0
  collision_weight: float = 0.0
  capsule_collision_weight: float = 0.0
  slip_contact_height: float = 0.025
  slip_contact_width: float = 0.01


class JointPolicyLearner:
  """BPTT actor update with a task-conditioned critic bootstrap."""

  def __init__(self, actor, world_model, fk, cfg: JointPolicyCfg, foot_site_offsets, capsule_collision=None):
    if cfg.horizon < 1:
      raise ValueError("horizon must be positive")
    self.actor, self.wm, self.fk, self.cfg = actor, world_model, fk, cfg
    self.foot_site_offsets = foot_site_offsets
    self.capsule_collision = capsule_collision
    self.device = next(actor.parameters()).device
    critic_dim = world_model.representation.rep_state_dim + 7 + 17 + 6 + 1 + 29 + 29 + 1
    critic_dim += world_model.collector.sim_context.output_dim
    self.critic = build_mlp(critic_dim, (256, 128), 1).to(self.device)
    self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

  @staticmethod
  def _critic_obs(
    state, aux, command, anchor, standing, sim_ctx, hand_start, hand_end, hand_elapsed
  ):
    return torch.cat(
      (
        state,
        aux,
        command,
        anchor,
        standing.unsqueeze(-1).to(state.dtype),
        sim_ctx,
        hand_start,
        hand_end,
        hand_elapsed.unsqueeze(-1),
      ),
      -1,
    )

  def update(self) -> dict[str, float]:
    batch, _ = self.wm.collector.buffer.sample_causal(
      self.actor.history_length, self.cfg.batch_size
    )
    # Future random command resamples cannot be predicted by the world model.
    # Draw only starts whose hand and velocity commands persist through rollout.
    required = self.cfg.horizon * self.wm.dynamics.dt
    for attempt in range(33):
      current = batch[:, -1]
      invalid = (current["hand_command_remaining"] <= required) | (
        current["twist_command_remaining"] <= required
      )
      if not invalid.any():
        break
      if attempt == 32:
        raise RuntimeError(
          "Replay lacks command-stable rollout starts; collect more transitions"
        )
      replacement, _ = self.wm.collector.buffer.sample_causal(
        self.actor.history_length, self.cfg.batch_size
      )
      batch[invalid] = replacement[invalid]
    start = batch[:, -1]
    rep = self.wm.representation
    state = rep.represent(start["state"]).detach()
    aux = rep.extract_aux_state(start["state"]).detach()
    command, anchor, standing, sim_ctx = (
      start["command"],
      start["foot_anchor"],
      start["standing"],
      start["sim_ctx"],
    )
    last_action = start["prev_action"]
    hand_start, hand_end, hand_elapsed = (
      start["hand_transition_start"],
      start["hand_transition_target"],
      start["hand_transition_elapsed"],
    )
    transition_duration = self.wm.collector.env.command_manager.get_term(
      "hands"
    ).cfg.transition_duration
    history = sensor_from_real_obs(batch["actor_obs"])
    survival = torch.ones(state.shape[0], device=self.device)
    total = torch.zeros_like(survival)
    critic_obs, rewards, dones = [], [], []
    dynamics = self.wm.dynamics
    dynamics.eval().requires_grad_(False)
    self.critic.requires_grad_(False)
    try:
      for t in range(self.cfg.horizon):
        inputs = TensorDict({"history": history, "command": command}, [len(state)])
        critic_obs.append(
          self._critic_obs(
            state,
            aux,
            command,
            anchor,
            standing,
            sim_ctx,
            hand_start,
            hand_end,
            hand_elapsed,
          ).detach()
        )
        action = self.actor(inputs)
        aux, state, reward, done, _ = joint_imagination_step(
          dynamics,
          self.fk,
          aux,
          state,
          action,
          sim_ctx,
          command,
          anchor,
          standing,
          last_action,
          rep._soft_joint_pos_limits,
          self.foot_site_offsets,
          self.cfg.clearance_weight,
          self.cfg.landing_weight,
          self.cfg.collision_weight,
          self.cfg.slip_contact_height,
          self.cfg.slip_contact_width,
          self.capsule_collision,
          self.cfg.capsule_collision_weight,
        )
        rewards.append(reward.detach())
        dones.append(done.detach())
        total = total + self.cfg.gamma**t * survival * reward
        survival = survival * (1.0 - done)
        sensor = rep.compute_actor_sensor(
          state,
          sim_ctx,
          aux_state=aux,
          joint_state=rep.joint_state(state),
          last_action=action,
          noisy=False,
        )
        context = self.wm.collector.sim_context
        bias_cols = context.find("encoder_bias")
        if bias_cols is not None:
          bias = context.denormalize(sim_ctx, bias_cols)
          sensor = torch.cat(
            (sensor[:, :6], sensor[:, 6:35] + bias, sensor[:, 35:]), dim=-1
          )
        # Match the velocity environment's per-channel uniform observation noise.
        bounds = rep._proprio_noise_bounds.clone()
        bounds[35:64] = 1.5
        sensor = sensor + ((torch.rand_like(sensor) * 2 - 1) * bounds).detach()
        history = torch.cat((history[:, 1:], sensor.unsqueeze(1)), dim=1)
        hand_elapsed = hand_elapsed + dynamics.dt
        hand_reference = interpolate_reference_joints(
          hand_start, hand_end, hand_elapsed / transition_duration
        )
        root = torch.zeros_like(aux[:, :3])
        identity = torch.zeros_like(aux[:, 3:7])
        identity[:, 0] = 1.0
        hand_fk = self.wm.collector.env.command_manager.get_term("hands").fk
        hand_pos, hand_quat = hand_fk(root, identity, hand_reference)
        hand_command = torch.cat((hand_pos, hand_quat), dim=-1).flatten(start_dim=1)
        command = torch.cat((command[:, :3], hand_command), dim=-1)
        last_action = action
      final_obs = self._critic_obs(
        state,
        aux,
        command,
        anchor,
        standing,
        sim_ctx,
        hand_start,
        hand_end,
        hand_elapsed,
      )
      value = self.critic(final_obs).squeeze(-1)
      loss = -(total + self.cfg.gamma**self.cfg.horizon * survival * value).mean()
      self.actor_optimizer.zero_grad(set_to_none=True)
      loss.backward()
      actor_grad = torch.nn.utils.clip_grad_norm_(
        self.actor.parameters(), self.cfg.grad_clip
      )
      self.actor_optimizer.step()
    finally:
      dynamics.requires_grad_(True)
      self.critic.requires_grad_(True)

    obs = torch.stack(critic_obs)
    with torch.no_grad():
      values = self.critic(obs)
      last_value = self.critic(final_obs.detach())
      returns = gae_returns(
        values,
        last_value,
        torch.stack(rewards).unsqueeze(-1),
        torch.stack(dones).unsqueeze(-1),
        self.cfg.gamma,
        self.cfg.lam,
      )
    critic_loss = (self.critic(obs) - returns).square().mean()
    self.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_grad = torch.nn.utils.clip_grad_norm_(
      self.critic.parameters(), self.cfg.grad_clip
    )
    self.critic_optimizer.step()
    return {
      "actor_loss": float(loss.detach()),
      "critic_loss": float(critic_loss.detach()),
      "actor_grad": float(actor_grad),
      "critic_grad": float(critic_grad),
      "imagined_reward": float(torch.stack(rewards).mean()),
    }
