"""Differentiable dynamics and joint-task scoring for imagined transitions."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_conjugate, quat_mul

from supermimic.tasks.joint_command import JointCommand, joint_tracking_errors
from supermimic.utils.gencoord_helpers import decompose_rep_state


def joint_imagination_step(
  dynamics,
  fk,
  aux: torch.Tensor,
  state: torch.Tensor,
  action: torch.Tensor,
  sim_ctx: torch.Tensor,
  command: torch.Tensor,
  foot_anchor: torch.Tensor,
  standing: torch.Tensor,
):
  """Predict one step and calculate differentiable task rewards.

  FK outputs must be left/right wrist followed by left/right ankle. Foot
  anchors and aux position share an environment-local world frame. This is an
  initial reward subset, not a substitute for real contact or safety checks.
  """
  next_aux, next_state = dynamics(state, action, aux, sim_ctx)
  _, gravity, lin_vel, ang_vel, qj, _ = decompose_rep_state(next_state)
  pos, quat = fk(next_aux[..., :3], next_aux[..., 3:7], qj)
  root_pos = next_aux[..., :3].unsqueeze(-2).expand_as(pos[..., :2, :])
  root_quat = next_aux[..., 3:7].unsqueeze(-2).expand_as(quat[..., :2, :])
  hand_pos = quat_apply_inverse(root_quat, pos[..., :2, :] - root_pos)
  hand_quat = quat_mul(quat_conjugate(root_quat), quat[..., :2, :])
  hands = torch.cat((hand_pos, hand_quat), dim=-1)
  velocity = torch.cat((lin_vel[..., :2], ang_vel[..., 2:3]), dim=-1)
  errors = joint_tracking_errors(
    JointCommand.from_tensor(command), velocity, hands[..., 0, :], hands[..., 1, :]
  )
  reward = (
    2.0 * torch.exp(-errors["linear_velocity"] / 0.25)
    + 2.0 * torch.exp(-errors["yaw_rate"] / 0.5)
    + torch.exp(-errors["left_position"] / 0.04)
    + torch.exp(-errors["right_position"] / 0.04)
    + 0.5 * torch.exp(-errors["left_orientation"] / 0.25)
    + 0.5 * torch.exp(-errors["right_orientation"] / 0.25)
  )
  feet_error = pos[..., 2:4, :] - foot_anchor.reshape(*pos.shape[:-2], 2, 3)
  feet_error = feet_error.square().sum(dim=(-1, -2))
  reward = reward + standing.to(reward.dtype) * torch.exp(-feet_error / 0.01)
  # A smooth proxy for severe tilt and very low root height. Thresholds are
  # provisional and independent of any reference-motion pelvis height.
  tilt_done = torch.sigmoid((gravity[..., 2] + 0.342) / 0.05)
  low_done = torch.sigmoid((0.3 - next_aux[..., 2]) / 0.05)
  done = 1.0 - (1.0 - tilt_done) * (1.0 - low_done)
  return next_aux, next_state, reward, done, errors
