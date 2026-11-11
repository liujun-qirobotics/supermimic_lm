"""Differentiable link poses for a free-root MuJoCo hinge-joint robot.

Joint and body names are taken from the actual compiled model. Joint positions
must be supplied in the explicit ``joint_names`` order; body poses are link
frame origins (not COM frames). This does not model contacts or dynamics.
"""

from __future__ import annotations

import mujoco
import torch
import torch.nn as nn
from mjlab.utils.lab_api.math import quat_apply, quat_mul


class DifferentiableFK(nn.Module):
  def __init__(
    self,
    model: mujoco.MjModel,
    root_body: str,
    joint_names: list[str] | tuple[str, ...],
    body_names: list[str] | tuple[str, ...],
  ) -> None:
    super().__init__()
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    if root_id < 1:
      raise ValueError(f"Unknown root body: {root_body}")
    if len(set(joint_names)) != len(joint_names):
      raise ValueError("Joint names must be unique")
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, x) for x in joint_names]
    if any(i < 0 for i in ids):
      raise ValueError("Some joint names do not exist in the MuJoCo model")
    if any(model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE for i in ids):
      raise ValueError("Only hinge joints are supported")
    outputs = [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, x) for x in body_names
    ]
    if not outputs or any(i < 0 for i in outputs):
      raise ValueError("At least one valid output body is required")

    joint_lookup = {j: i for i, j in enumerate(ids)}
    required = {root_id}
    for body_id in outputs:
      current = body_id
      while current != root_id and current != 0:
        required.add(current)
        current = int(model.body_parentid[current])
      if current != root_id:
        raise ValueError("All output bodies must descend from root_body")
    self.root_id = root_id
    self.body_ids = tuple(sorted(required))
    self.output_ids = tuple(outputs)
    self.num_joints = len(ids)
    self.body_parent: dict[int, int] = {}
    self.body_joint: dict[int, tuple[int, int] | None] = {}
    for body_id in self.body_ids:
      if body_id == root_id:
        continue
      self.body_parent[body_id] = int(model.body_parentid[body_id])
      count = int(model.body_jntnum[body_id])
      if count > 1:
        raise ValueError("Only one hinge per body is supported")
      if count == 0:
        self.body_joint[body_id] = None
        continue
      joint_id = int(model.body_jntadr[body_id])
      if joint_id not in joint_lookup:
        raise ValueError(f"Missing joint {model.joint(joint_id).name} in joint_names")
      self.body_joint[body_id] = (joint_id, joint_lookup[joint_id])

    def buffer(name: str, array) -> None:
      self.register_buffer(name, torch.as_tensor(array.copy(), dtype=torch.float64))

    buffer("body_pos", model.body_pos)
    buffer("body_quat", model.body_quat)
    buffer("joint_pos", model.jnt_pos)
    buffer("joint_axis", model.jnt_axis)
    buffer("qpos0", model.qpos0[model.jnt_qposadr[ids]])

  def forward(
    self, root_pos: torch.Tensor, root_quat: torch.Tensor, joint_pos: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if joint_pos.shape[-1] != self.num_joints:
      raise ValueError(f"Expected {self.num_joints} joint positions")
    positions = {self.root_id: root_pos}
    rotations = {self.root_id: torch.nn.functional.normalize(root_quat, dim=-1)}
    for body_id in self.body_ids:
      if body_id == self.root_id:
        continue
      parent = self.body_parent[body_id]
      parent_p, parent_q = positions[parent], rotations[parent]
      local_p = self.body_pos[body_id].to(joint_pos).expand_as(root_pos)
      local_q = self.body_quat[body_id].to(joint_pos).expand_as(root_quat)
      joint = self.body_joint[body_id]
      if joint is not None:
        joint_id, index = joint
        half = (joint_pos[..., index] - self.qpos0[index].to(joint_pos)) * 0.5
        axis = self.joint_axis[joint_id].to(joint_pos)
        rotation = torch.cat(
          (half.cos().unsqueeze(-1), half.sin().unsqueeze(-1) * axis), dim=-1
        )
        anchor = self.joint_pos[joint_id].to(joint_pos).expand_as(root_pos)
        local_p = local_p + quat_apply(local_q, anchor - quat_apply(rotation, anchor))
        local_q = quat_mul(local_q, rotation)
      positions[body_id] = parent_p + quat_apply(parent_q, local_p)
      rotations[body_id] = quat_mul(parent_q, local_q)
    return (
      torch.stack([positions[i] for i in self.output_ids], dim=-2),
      torch.stack([rotations[i] for i in self.output_ids], dim=-2),
    )
