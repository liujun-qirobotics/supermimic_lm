"""Reference-free G1 locomotion and bimanual task for real simulation.

The initial TCP is the wrist-yaw link origin; this is not a hand contact point.
Pose targets use the full pelvis frame. This first sampling scheme perturbs the
current pose and does not guarantee kinematic reachability.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import mdp
from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

from supermimic.tasks.joint_command import hand_pose_in_body_frame


@dataclass(kw_only=True)
class HandPoseCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  position_delta: float = 0.08
  rotation_delta: float = 0.25
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
    self._command = torch.zeros(self.num_envs, 14, device=self.device)
    self._command[:, 3] = 1.0
    self._command[:, 10] = 1.0
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
    current = self.current_poses()[env_ids]
    count = len(env_ids)
    position_noise = torch.empty(count, 2, 3, device=self.device).uniform_(
      -self.cfg.position_delta, self.cfg.position_delta
    )
    angles = torch.empty(count, 2, 3, device=self.device).uniform_(
      -self.cfg.rotation_delta, self.cfg.rotation_delta
    )
    delta = quat_from_euler_xyz(angles[..., 0], angles[..., 1], angles[..., 2])
    pose = torch.cat(
      (current[..., :3] + position_noise, quat_mul(delta, current[..., 3:])), dim=-1
    )
    self._command[env_ids] = pose.flatten(start_dim=1)

  def reset(self, env_ids):
    result = super().reset(env_ids)
    self.was_standing[env_ids] = False
    return result

  def _update_command(self):
    twist = self._env.command_manager.get_command("twist")
    standing = torch.linalg.vector_norm(twist, dim=-1) < 0.05
    capture = standing & ~self.was_standing
    positions = self.robot.data.body_link_pos_w[:, self.foot_ids, :]
    self.foot_anchor = torch.where(capture[:, None, None], positions, self.foot_anchor)
    self.was_standing = standing

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
