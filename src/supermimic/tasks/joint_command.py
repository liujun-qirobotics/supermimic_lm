"""Joint locomotion and bimanual pose commands, independent of motion clips.

Hand poses are expressed in the same body-fixed frame, using wxyz
quaternions. The choice of body frame and hand TCP is made by the caller.
These functions do not establish reachability or foot-contact constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_conjugate, quat_mul


def interpolate_reference_joints(
  start: torch.Tensor, end: torch.Tensor, fraction: torch.Tensor
) -> torch.Tensor:
  """Interpolate joint references with zero endpoint velocities."""
  u = fraction.clamp(0.0, 1.0)
  u = (u * u * (3.0 - 2.0 * u))[..., None]
  return start + u * (end - start)


@dataclass(frozen=True)
class JointCommand:
  """A simultaneous velocity command and two body-relative hand poses.

  ``velocity`` is (..., 3), ordered vx, vy, yaw rate in the body frame;
  each hand pose is (..., 7), ordered xyz and quaternion wxyz.
  """

  velocity: torch.Tensor
  left_hand: torch.Tensor
  right_hand: torch.Tensor

  def as_tensor(self) -> torch.Tensor:
    return torch.cat((self.velocity, self.left_hand, self.right_hand), dim=-1)

  @classmethod
  def from_tensor(cls, command: torch.Tensor) -> JointCommand:
    if command.shape[-1] != 17:
      raise ValueError("joint command must have 17 components")
    return cls(command[..., :3], command[..., 3:10], command[..., 10:17])


def hand_pose_in_body_frame(
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  hand_pos_w: torch.Tensor,
  hand_quat_w: torch.Tensor,
) -> torch.Tensor:
  """Transform a world-frame TCP pose to the chosen body frame."""
  body_quat_w = F.normalize(body_quat_w, dim=-1)
  hand_quat_w = F.normalize(hand_quat_w, dim=-1)
  pos = quat_apply_inverse(body_quat_w, hand_pos_w - body_pos_w)
  quat = quat_mul(quat_conjugate(body_quat_w), hand_quat_w)
  return torch.cat((pos, quat), dim=-1)


def joint_tracking_errors(
  command: JointCommand,
  velocity_b: torch.Tensor,
  left_hand_b: torch.Tensor,
  right_hand_b: torch.Tensor,
) -> dict[str, torch.Tensor]:
  """Return per-sample squared errors, suitable for differentiable rewards.

  Quaternion error is 1 - dot(q, q_target)^2: sign-invariant and smooth
  near matching orientations. Inputs must contain valid, nonzero quaternions.
  """

  def pose_error(actual: torch.Tensor, target: torch.Tensor):
    position = (actual[..., :3] - target[..., :3]).square().sum(dim=-1)
    qa = F.normalize(actual[..., 3:7], dim=-1)
    qt = F.normalize(target[..., 3:7], dim=-1)
    orientation = 1.0 - (qa * qt).sum(dim=-1).square().clamp(max=1.0)
    return position, orientation

  left_pos, left_rot = pose_error(left_hand_b, command.left_hand)
  right_pos, right_rot = pose_error(right_hand_b, command.right_hand)
  velocity = (velocity_b - command.velocity).square()
  return {
    "linear_velocity": velocity[..., :2].sum(dim=-1),
    "yaw_rate": velocity[..., 2],
    "left_position": left_pos,
    "left_orientation": left_rot,
    "right_position": right_pos,
    "right_orientation": right_rot,
  }
