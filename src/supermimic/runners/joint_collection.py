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
    self.buffer = ReplayBuffer(
      schema={
        "state": FieldSpec(state.shape[-1]),
        "action": FieldSpec(action_dim),
        "prev_action": FieldSpec(action_dim),
        "actor_obs": FieldSpec(obs_dim),
        "command": FieldSpec(17),
        "foot_anchor": FieldSpec(6),
        "standing": FieldSpec(dtype=torch.bool),
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
      "foot_anchor": (hand.foot_anchor - env.scene.env_origins[:, None, :])
      .flatten(start_dim=1)
      .detach()
      .clone(),
      "standing": hand.was_standing.detach().clone(),
      "sim_ctx": self.sim_context.read(env).detach().clone(),
    }
    obs, reward, terminated, truncated, info = env.step(action)
    fields["terminated"] = terminated.detach().clone()
    fields["truncated"] = truncated.detach().clone()
    self.buffer.insert(TensorDict(fields, batch_size=[env.num_envs]))
    return obs, reward, terminated, truncated, info
