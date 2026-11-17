"""Differentiable dynamics and joint-task scoring for imagined transitions."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
)

from supermimic.tasks.joint_command import JointCommand, joint_tracking_errors
from supermimic.utils.differentiable_fk import DifferentiableFKWithVelocity
from supermimic.utils.gencoord_helpers import decompose_rep_state


def foot_contact_proxy(height: torch.Tensor) -> torch.Tensor:
  """Smooth flat-ground proximity, not a physical contact measurement."""
  return torch.sigmoid((0.025 - height) / 0.01)


def foot_slip_proxy(
  site_pos: torch.Tensor, site_vel: torch.Tensor, command: torch.Tensor,
  height_threshold: float = 0.025, width: float = 0.01,
):
  """Squared horizontal speed weighted by flat-ground proximity during motion."""
  active = (command[..., :2].norm(dim=-1) + command[..., 2].abs()) > 0.05
  slip = (
    torch.sigmoid((height_threshold - site_pos[..., 2]) / width)
    * site_vel[..., :2].square().sum(-1)
  ).sum(-1)
  return active.to(slip.dtype) * slip


def foot_clearance_proxy(site_pos, site_vel, command, target_height=0.1):
  """Flat-ground counterpart of mjlab's velocity-weighted clearance cost."""
  active = (command[..., :2].norm(dim=-1) + command[..., 2].abs()) > 0.05
  error = (site_pos[..., 2] - target_height).abs()
  return active.to(error.dtype) * (error * site_vel[..., :2].norm(dim=-1)).sum(-1)


def landing_speed_proxy(site_pos, site_vel, command):
  """Downward speed near the ground; not a measurement of impact force."""
  active = (command[..., :2].norm(dim=-1) + command[..., 2].abs()) > 0.05
  cost = (foot_contact_proxy(site_pos[..., 2]) * site_vel[..., 2].clamp(max=0).square()).sum(-1)
  return active.to(cost.dtype) * cost


def ankle_collision_proxy(ankle_pos, min_distance=0.10):
  """One conservative ankle pair only, not full-body collision detection."""
  distance = (ankle_pos[..., 0, :] - ankle_pos[..., 1, :]).norm(dim=-1)
  return (min_distance - distance).clamp_min(0).square()


def body_collision_proxy(pos, quat):
  """Selected ankle, wrist and torso point pairs; not mesh collision detection."""
  if pos.shape[-2] < 5:
    raise ValueError("Collision proxy requires wrists, ankles and torso FK outputs")
  torso_offset = pos.new_tensor([0.0, 0.0, 0.12]).expand_as(pos[..., 4, :])
  torso = pos[..., 4, :] + quat_apply(quat[..., 4, :], torso_offset)
  wrist_distance = (pos[..., 0, :] - pos[..., 1, :]).norm(dim=-1)
  torso_distance = (pos[..., :2, :] - torso.unsqueeze(-2)).norm(dim=-1)
  return (
    ankle_collision_proxy(pos[..., 2:4, :])
    + (0.08 - wrist_distance).clamp_min(0).square()
    + (0.12 - torso_distance).clamp_min(0).square().sum(-1)
  )


def imagined_done(aux: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
  """Existing smooth tilt and root-height termination proxy."""
  _, gravity, *_ = decompose_rep_state(state)
  tilt_done = torch.sigmoid((gravity[..., 2] + 0.342) / 0.05)
  low_done = torch.sigmoid((0.3 - aux[..., 2]) / 0.05)
  return 1.0 - (1.0 - tilt_done) * (1.0 - low_done)


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
  last_action: torch.Tensor | None = None,
  joint_limits: torch.Tensor | None = None,
  foot_site_offsets: torch.Tensor | None = None,
  clearance_weight: float = 2.0,
  landing_weight: float = 0.0,
  collision_weight: float = 0.0,
  slip_contact_height: float = 0.025,
  slip_contact_width: float = 0.01,
  capsule_collision=None,
  capsule_collision_weight: float = 0.0,
):
  """Predict one step and calculate differentiable task rewards.

  FK outputs must be left/right wrist followed by left/right ankle. Foot
  anchors and aux position share an environment-local world frame. This is an
  initial reward subset, not a substitute for real contact or safety checks.
  """
  next_aux, next_state = dynamics(state, action, aux, sim_ctx)
  _, gravity, lin_vel, ang_vel, qj, vj = decompose_rep_state(next_state)
  if foot_site_offsets is None:
    pos, quat = fk(next_aux[..., :3], next_aux[..., 3:7], qj)
  else:
    root_q = next_aux[..., 3:7]
    pos, quat, link_vel, link_omega = DifferentiableFKWithVelocity(fk)(
      next_aux[..., :3], root_q, qj, quat_apply(root_q, lin_vel),
      quat_apply(root_q, ang_vel), vj
    )
    offset = quat_apply(quat[..., 2:4, :], foot_site_offsets.expand_as(pos[..., 2:4, :]))
    site_pos = pos[..., 2:4, :] + offset
    site_vel = link_vel[..., 2:4, :] + torch.cross(
      link_omega[..., 2:4, :], offset, dim=-1
    )
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
    2.0 * torch.exp(-(errors["linear_velocity"] + lin_vel[..., 2].square()) / 0.25)
    + 2.0 * torch.exp(-(errors["yaw_rate"] + ang_vel[..., :2].square().sum(-1)) / 0.5)
    + torch.exp(-errors["left_position"] / 0.04)
    + torch.exp(-errors["right_position"] / 0.04)
    + 0.5 * torch.exp(-errors["left_orientation"] / 0.25)
    + 0.5 * torch.exp(-errors["right_orientation"] / 0.25)
  )
  feet_error = pos[..., 2:4, :] - foot_anchor.reshape(*pos.shape[:-2], 2, 3)
  feet_error = feet_error.square().sum(dim=(-1, -2))
  reward = reward + standing.to(reward.dtype) * torch.exp(-feet_error / 0.01)
  if foot_site_offsets is not None:
    slip = foot_slip_proxy(site_pos, site_vel, command, slip_contact_height, slip_contact_width)
    clearance = foot_clearance_proxy(site_pos, site_vel, command)
    landing = landing_speed_proxy(site_pos, site_vel, command)
    collision = body_collision_proxy(pos, quat) if pos.shape[-2] >= 5 else ankle_collision_proxy(pos[..., 2:4, :])
    reward = reward - 0.1 * slip
    reward = reward - clearance_weight * clearance
    reward = reward - landing_weight * landing - collision_weight * collision
    errors["foot_slip_proxy"] = slip
    errors["foot_clearance_proxy"] = clearance
    errors["landing_speed_proxy"] = landing
    errors["body_collision_proxy"] = collision
  if capsule_collision_weight and capsule_collision is None:
    raise ValueError("capsule collision model required when its weight is nonzero")
  if capsule_collision is not None and capsule_collision_weight:
    separation = capsule_collision(next_aux[..., :3], next_aux[..., 3:7], qj)
    capsule_cost = (-separation).clamp_min(0).square().sum(-1)
    reward = reward - capsule_collision_weight * capsule_cost
    errors["capsule_collision_proxy"] = capsule_cost
  # Root tilt is a differentiable proxy; the real task scores torso tilt.
  reward = reward + 0.3 * torch.exp(-gravity[..., :2].square().sum(-1) / 0.2)
  if joint_limits is not None:
    violation = (joint_limits[:, 0] - qj).clamp_min(0) + (
      qj - joint_limits[:, 1]
    ).clamp_min(0)
    reward = reward - violation.sum(dim=-1)
  if last_action is not None:
    reward = reward - 0.1 * (action - last_action).square().sum(-1)
  reward = reward * dynamics.dt
  # A smooth proxy for severe tilt and very low root height. Thresholds are
  # provisional and independent of any reference-motion pelvis height.
  done = imagined_done(next_aux, next_state)
  return next_aux, next_state, reward, done, errors
