"""Reference-free G1 locomotion and bimanual task for real simulation.

The initial TCP is the wrist-yaw link origin; this is not a hand contact point.
Pose targets use the full pelvis frame and are sampled via joint-space FK.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import mdp
from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from supermimic.tasks.joint_command import (
  hand_pose_in_body_frame,
  interpolate_reference_joints,
)
from supermimic.utils.differentiable_fk import DifferentiableFK


@dataclass(kw_only=True)
class HandPoseCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  shoulder_pitch_range: tuple[float, float] = (-0.15, 0.45)
  elbow_range: tuple[float, float] = (0.0, 1.4)
  transition_duration: float = 1.0
  left_body: str = "left_wrist_yaw_link"
  right_body: str = "right_wrist_yaw_link"

  def build(self, env):
    return HandPoseCommand(self, env)


class HandPoseCommand(CommandTerm):
  cfg: HandPoseCommandCfg

  def __init__(self, cfg: HandPoseCommandCfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self.hand_ids, names = self.robot.find_bodies(
      (cfg.left_body, cfg.right_body), preserve_order=True
    )
    if names != [cfg.left_body, cfg.right_body]:
      raise ValueError(f"Expected two wrist bodies, found {names}")
    self.foot_ids, feet = self.robot.find_bodies(
      ("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True
    )
    if feet != ["left_ankle_roll_link", "right_ankle_roll_link"]:
      raise ValueError(f"Expected two ankle bodies, found {feet}")
    if (
      cfg.transition_duration <= 0
      or cfg.transition_duration > cfg.resampling_time_range[0]
    ):
      raise ValueError(
        "transition_duration must be positive and no longer than the minimum resampling interval"
      )
    model = env.sim.mj_model
    names = [model.joint(int(i)).name for i in self.robot.indexing.joint_ids]
    root = model.body(int(self.robot.indexing.root_body_id)).name
    bodies = [
      model.body(int(self.robot.indexing.body_ids[i])).name for i in self.hand_ids
    ]
    self.fk = DifferentiableFK(model, root, names, bodies).to(self.device)
    self._sample_indices = [
      self.robot.joint_names.index(f"{side}_{joint}_joint")
      for side in ("left", "right")
      for joint in ("shoulder_pitch", "elbow")
    ]
    limits = self.robot.data.soft_joint_pos_limits[0, self._sample_indices]
    ranges = torch.tensor(
      [cfg.shoulder_pitch_range, cfg.elbow_range] * 2, device=self.device
    )
    if bool(
      (
        (ranges[:, 0] > ranges[:, 1])
        | (ranges[:, 0] < limits[:, 0])
        | (ranges[:, 1] > limits[:, 1])
      ).any()
    ):
      raise ValueError("hand sampling joint ranges must lie within G1 soft limits")
    self._sample_ranges = ranges
    self._command = torch.zeros(self.num_envs, 14, device=self.device)
    self._command[:, 3] = 1.0
    self._command[:, 10] = 1.0
    self.transition_start = self.robot.data.default_joint_pos.clone()
    self.transition_target = self.transition_start.clone()
    self.reference_joints = self.transition_start.clone()
    self.transition_time_left_at_start = torch.zeros(self.num_envs, device=self.device)
    self._resetting = False
    self.foot_anchor = torch.zeros(self.num_envs, 2, 3, device=self.device)
    self.was_standing = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

  @property
  def command(self):
    return self._command

  def current_poses(self):
    data = self.robot.data
    return torch.stack(
      [
        hand_pose_in_body_frame(
          data.root_link_pos_w,
          data.root_link_quat_w,
          data.body_link_pos_w[:, body_id],
          data.body_link_quat_w[:, body_id],
        )
        for body_id in self.hand_ids
      ],
      dim=1,
    )

  def _resample_command(self, env_ids):
    count = len(env_ids)
    q = self.robot.data.default_joint_pos[env_ids].clone()
    random = torch.rand(count, 4, device=self.device)
    q[:, self._sample_indices] = self._sample_ranges[:, 0] + random * (
      self._sample_ranges[:, 1] - self._sample_ranges[:, 0]
    )
    self.transition_target[env_ids] = q
    if self._resetting:
      self.reference_joints[env_ids] = self.robot.data.joint_pos[env_ids]
    self.transition_start[env_ids] = self.reference_joints[env_ids]
    self.transition_time_left_at_start[env_ids] = self.time_left[env_ids]

  def reset(self, env_ids):
    self.was_standing[env_ids] = False
    self._resetting = True
    try:
      return super().reset(env_ids)
    finally:
      self._resetting = False

  def _update_command(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    elapsed = self.transition_time_left_at_start[ids] - self.time_left[ids]
    reference = interpolate_reference_joints(
      self.transition_start[ids],
      self.transition_target[ids],
      elapsed / self.cfg.transition_duration,
    )
    self.reference_joints[ids] = reference
    root = torch.zeros(reference.shape[0], 3, device=self.device)
    quat = torch.zeros(reference.shape[0], 4, device=self.device)
    quat[:, 0] = 1.0
    pos, orientation = self.fk(root, quat, reference)
    self._command[ids] = torch.cat((pos, orientation), -1).flatten(start_dim=1)
    twist = self._env.command_manager.get_command("twist")[ids]
    standing = torch.linalg.vector_norm(twist, dim=-1) < 0.05
    capture = standing & ~self.was_standing[ids]
    positions = self.robot.data.body_link_pos_w[ids][:, self.foot_ids, :]
    self.foot_anchor[ids] = torch.where(
      capture[:, None, None], positions, self.foot_anchor[ids]
    )
    self.was_standing[ids] = standing

  def _update_metrics(self):
    pass


def current_hand_poses(env, command_name: str = "hands"):
  term = env.command_manager.get_term(command_name)
  return term.current_poses().flatten(start_dim=1)


def hand_position_reward(env, hand: int, std: float, command_name: str = "hands"):
  term = env.command_manager.get_term(command_name)
  target = term.command.reshape(-1, 2, 7)[:, hand, :3]
  actual = term.current_poses()[:, hand, :3]
  return torch.exp(-(actual - target).square().sum(dim=-1) / std**2)


def hand_orientation_reward(env, hand: int, std: float, command_name: str = "hands"):
  term = env.command_manager.get_term(command_name)
  target = term.command.reshape(-1, 2, 7)[:, hand, 3:]
  actual = term.current_poses()[:, hand, 3:]
  similarity = (actual * target).sum(dim=-1).square().clamp(max=1.0)
  return torch.exp(-(1.0 - similarity) / std**2)


def standing_foot_anchor_reward(env, std: float, command_name: str = "hands"):
  """Soft ankle-position objective only while commanded to stand still."""
  term = env.command_manager.get_term(command_name)
  feet = term.robot.data.body_link_pos_w[:, term.foot_ids, :]
  error = (feet - term.foot_anchor).square().sum(dim=(-1, -2))
  return torch.where(term.was_standing, torch.exp(-error / std**2), 0.0)


def joint_g1_env_cfg(play: bool = False):
  cfg = unitree_g1_flat_env_cfg(play=play)
  # Initial task ranges; gradually widen only after evaluating real rollouts.
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (-0.4, 0.6)
  twist.ranges.lin_vel_y = (-0.3, 0.3)
  twist.ranges.ang_vel_z = (-0.5, 0.5)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_heading_envs = 0.0
  twist.rel_world_envs = 0.0
  twist.rel_standing_envs = 0.2
  cfg.commands["hands"] = HandPoseCommandCfg(resampling_time_range=(3.0, 6.0))
  for group in ("actor", "critic"):
    cfg.observations[group].terms["hand_targets"] = ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "hands"}
    )
    cfg.observations[group].terms["hand_poses"] = ObservationTermCfg(
      func=current_hand_poses
    )
  # The base velocity task's full-body default-pose reward opposes moving the
  # arms and crouching. Keep other regularization and upright constraints.
  cfg.rewards.pop("pose", None)
  for hand, name in enumerate(("left", "right")):
    cfg.rewards[f"{name}_hand_pos"] = RewardTermCfg(
      func=hand_position_reward, weight=1.0, params={"hand": hand, "std": 0.2}
    )
    cfg.rewards[f"{name}_hand_rot"] = RewardTermCfg(
      func=hand_orientation_reward, weight=0.5, params={"hand": hand, "std": 0.5}
    )
  cfg.rewards["standing_foot_anchor"] = RewardTermCfg(
    func=standing_foot_anchor_reward, weight=1.0, params={"std": 0.1}
  )
  return cfg
