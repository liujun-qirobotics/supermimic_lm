"""Differentiable distances for selected G1 capsule collision geometries."""

from __future__ import annotations

import mujoco
import torch
import torch.nn as nn

from supermimic.utils.differentiable_fk import DifferentiableFK, _quat_apply


def _point_segment_distance_sq(point, a, b):
  direction = b - a
  length_sq = direction.square().sum(-1).clamp_min(1e-12)
  t = ((point - a) * direction).sum(-1).div(length_sq).clamp(0, 1)
  return (point - (a + t.unsqueeze(-1) * direction)).square().sum(-1)


def segment_distance(a, b, c, d):
  """Shortest distance between two finite segments, including parallel cases."""
  u, v, w = b - a, d - c, a - c
  aa, bb, cc = (u*u).sum(-1), (u*v).sum(-1), (v*v).sum(-1)
  dd, ee = (u*w).sum(-1), (v*w).sum(-1)
  denom = aa * cc - bb.square()
  safe = denom.clamp_min(1e-12)
  s = (bb*ee - cc*dd) / safe
  t = (aa*ee - bb*dd) / safe
  valid = (denom > 1e-12) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
  interior = (w + s.unsqueeze(-1)*u - t.unsqueeze(-1)*v).square().sum(-1)
  interior = torch.where(valid, interior, torch.full_like(interior, float("inf")))
  distances = torch.stack((
    _point_segment_distance_sq(a, c, d), _point_segment_distance_sq(b, c, d),
    _point_segment_distance_sq(c, a, b), _point_segment_distance_sq(d, a, b), interior,
  ), dim=-1)
  # Clamp avoids undefined sqrt gradients for intersecting axes.
  return distances.amin(dim=-1).clamp_min(1e-16).sqrt()


class G1CapsuleCollision(nn.Module):
  """Same-side wrist/hand against hip/thigh, using compiled model capsules.

  The result is geometric signed surface separation, not a contact force.
  It covers only the explicitly selected pairs, not the whole robot.
  """

  def __init__(self, model, robot):
    super().__init__()
    geom_names = [f"{side}_{part}_collision" for side in ("left", "right")
                  for part in ("hip", "thigh", "wrist", "hand")]
    geom_ids = [int(robot.indexing.geom_ids[robot.geom_names.index(name)]) for name in geom_names]
    if any(model.geom_type[i] != mujoco.mjtGeom.mjGEOM_CAPSULE for i in geom_ids):
      raise ValueError("Expected G1 hip, thigh, wrist and hand collision capsules")
    body_names = [model.body(int(model.geom_bodyid[i])).name for i in geom_ids]
    joint_names = [model.joint(int(i)).name for i in robot.indexing.joint_ids]
    root = model.body(int(robot.indexing.root_body_id)).name
    self.fk = DifferentiableFK(model, root, joint_names, body_names)
    geom_q = torch.as_tensor(model.geom_quat[geom_ids].copy(), dtype=torch.float64)
    local_axis = torch.zeros(len(geom_ids), 3, dtype=torch.float64)
    local_axis[:, 2] = 1.0
    axis = _quat_apply(geom_q, local_axis)
    center = torch.as_tensor(model.geom_pos[geom_ids].copy(), dtype=torch.float64)
    size = torch.as_tensor(model.geom_size[geom_ids].copy(), dtype=torch.float64)
    self.register_buffer("endpoint_a", center - axis * size[:, 1:2])
    self.register_buffer("endpoint_b", center + axis * size[:, 1:2])
    self.register_buffer("radius", size[:, 0])
    self.geom_names = tuple(geom_names)
    self.pairs = tuple((base + limb, base + arm)
                       for base in (0, 4) for limb in (0, 1) for arm in (2, 3))
    self.pair_names = tuple((geom_names[a], geom_names[b]) for a, b in self.pairs)

  def forward(self, root_pos, root_quat, joint_pos):
    body_pos, body_quat = self.fk(root_pos, root_quat, joint_pos)
    a = body_pos + _quat_apply(body_quat, self.endpoint_a.to(joint_pos).expand_as(body_pos))
    b = body_pos + _quat_apply(body_quat, self.endpoint_b.to(joint_pos).expand_as(body_pos))
    first = [p[0] for p in self.pairs]
    second = [p[1] for p in self.pairs]
    distance = segment_distance(a[..., first, :], b[..., first, :],
                                a[..., second, :], b[..., second, :])
    return distance - (self.radius[first] + self.radius[second]).to(joint_pos)
