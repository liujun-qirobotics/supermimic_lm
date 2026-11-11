"""World-model learner for reference-free G1 joint-task transitions."""

from __future__ import annotations

import torch

from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics
from supermimic.representations import StateRepresentation, create_representation
from supermimic.runners.joint_collection import JointReplayCollector, full_state
from supermimic.utils.helpers import extract_action_scale_offset


class JointWorldModel:
  """Train existing GenCoord dynamics using the joint task's replay buffer.

  This object trains only dynamics; it does not yet implement imagined policy
  updates, task rewards, checkpoint orchestration, or a complete train runner.
  """

  def __init__(
    self,
    collector: JointReplayCollector,
    cfg: WorldModelLearnerCfg,
    hidden_dims: tuple[int, ...] = (512, 512, 512),
  ) -> None:
    self.collector = collector
    env = collector.env
    state = full_state(env)
    action_dim = env.action_manager.total_action_dim
    default_q = env.scene["robot"].data.default_joint_pos[0].to(env.device)
    self.representation = create_representation(
      "gencoord",
      state.shape[-1],
      action_dim,
      env.step_dt,
      env.cfg.decimation,
      default_q,
      "proprio",
    ).to(env.device)
    # Bind base-class limits without GenCoord's reference-motion FK binding.
    StateRepresentation.bind(self.representation, env, collector.sim_context)
    scale, offset = extract_action_scale_offset(env)
    self.dynamics = Dynamics(
      self.representation,
      action_dim,
      collector.sim_context.output_dim,
      scale,
      offset,
      hidden_dims,
    ).to(env.device)
    self.learner = WorldModelLearner(
      self.dynamics, self.representation, cfg, device=env.device
    )
    self.cfg = cfg

  def update(self) -> dict[str, torch.Tensor]:
    samples = [
      self.collector.buffer.sample(self.cfg.horizon, self.cfg.batch_size)
      for _ in range(self.cfg.num_batches)
    ]
    states = torch.stack([sample["state"] for sample in samples])
    actions = torch.stack([sample["action"] for sample in samples])
    self.learner.update_normalization(states, actions)
    return self.learner.update(samples)
