"""Single-clip compatibility for the public mjlab tracking command.

This adapter does not provide research-mjlab multi-clip or FK support.
"""

from dataclasses import dataclass

import torch
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg


class SingleMotionAdapter:
  def __init__(self, motion, anchor_index: int):
    self._motion = motion
    self.motion_ends = torch.tensor(
      [motion.time_step_total], device=motion.joint_pos.device
    )
    self.root_pos_w = motion.body_pos_w[:, anchor_index]
    self.root_quat_w = motion.body_quat_w[:, anchor_index]
    self.root_lin_vel_w = motion.body_lin_vel_w[:, anchor_index]
    self.root_ang_vel_w = motion.body_ang_vel_w[:, anchor_index]

  def __getattr__(self, name):
    return getattr(self._motion, name)

  def body_state(self, frames):
    return tuple(
      getattr(self._motion, key)[frames]
      for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    )


class CompatibleMotionCommand(MotionCommand):
  def __init__(self, cfg, env):
    if not isinstance(cfg.motion_file, str):
      raise ValueError(
        "Public mjlab compatibility currently supports exactly one motion file"
      )
    if cfg.sampling_mode == "external":
      raise ValueError(
        "Public mjlab does not support return/external sampling; use uniform, start or adaptive"
      )
    super().__init__(cfg, env)
    if cfg.anchor_body_name != "pelvis":
      raise ValueError("Compatibility adapter requires pelvis as anchor")
    self.motion = SingleMotionAdapter(self.motion, self.motion_anchor_body_index)


@dataclass(kw_only=True)
class CompatibleMotionCommandCfg(MotionCommandCfg):
  def build(self, env):
    return CompatibleMotionCommand(self, env)
