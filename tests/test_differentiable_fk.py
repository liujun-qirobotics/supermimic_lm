import mujoco
import numpy as np
import pytest
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from supermimic.utils.differentiable_fk import DifferentiableFK


def test_g1_fk_matches_mujoco_and_has_waist_gradients():
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  data = mujoco.MjData(model)
  joint_ids = [
    i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
  ]
  joint_names = [model.joint(i).name for i in joint_ids]
  body_names = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ]
  fk = DifferentiableFK(model, "pelvis", joint_names, body_names)
  rng = np.random.default_rng(19)
  angles = rng.uniform(-0.3, 0.3, size=len(joint_ids))
  data.qpos[model.jnt_qposadr[joint_ids]] = angles
  data.qpos[:7] = [0.2, -0.1, 0.75, np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
  mujoco.mj_forward(model, data)
  q = torch.tensor(angles, dtype=torch.float64, requires_grad=True)
  pos, quat = fk(
    torch.tensor(data.qpos[:3].copy()), torch.tensor(data.qpos[3:7].copy()), q
  )
  expected_pos = np.stack([data.body(name).xpos.copy() for name in body_names])
  expected_quat = np.stack([data.body(name).xquat.copy() for name in body_names])
  np.testing.assert_allclose(pos.detach().numpy(), expected_pos, atol=1e-9)
  np.testing.assert_allclose(
    np.abs((quat.detach().numpy() * expected_quat).sum(-1)), 1, atol=1e-9
  )
  pos[0, 0].backward()
  waist = joint_names.index("waist_yaw_joint")
  assert torch.isfinite(q.grad).all() and abs(q.grad[waist]) > 1e-5
  eps = 1e-5
  plus = q.detach().clone()
  minus = q.detach().clone()
  plus[waist] += eps
  minus[waist] -= eps
  args = (torch.tensor(data.qpos[:3].copy()), torch.tensor(data.qpos[3:7].copy()))
  finite_diff = (fk(*args, plus)[0][0, 0] - fk(*args, minus)[0][0, 0]) / (2 * eps)
  torch.testing.assert_close(q.grad[waist], finite_diff, atol=1e-7, rtol=1e-5)


def test_g1_fk_velocities_match_mujoco():
  from supermimic.utils.differentiable_fk import DifferentiableFKWithVelocity

  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  data = mujoco.MjData(model)
  ids = [
    i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
  ]
  names = [model.joint(i).name for i in ids]
  bodies = ["pelvis", "left_wrist_yaw_link", "right_ankle_roll_link"]
  fk = DifferentiableFKWithVelocity(DifferentiableFK(model, "pelvis", names, bodies))
  rng = np.random.default_rng(4)
  data.qpos[:7] = [0.2, -0.1, 0.75, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
  data.qpos[model.jnt_qposadr[ids]] = rng.uniform(-0.3, 0.3, len(ids))
  data.qvel[:] = rng.uniform(-0.2, 0.2, model.nv)
  mujoco.mj_forward(model, data)
  # Free joint angular velocity in MuJoCo qvel is local; read world velocity.
  velocity = np.zeros(6)
  mujoco.mj_objectVelocity(
    model, data, mujoco.mjtObj.mjOBJ_XBODY, model.body("pelvis").id, velocity, 0
  )
  args = [
    torch.tensor(a.copy())
    for a in (
      data.qpos[:3],
      data.qpos[3:7],
      data.qpos[model.jnt_qposadr[ids]],
      velocity[3:],
      velocity[:3],
      data.qvel[model.jnt_dofadr[ids]],
    )
  ]
  _, _, linear, angular = fk(*args)
  for i, name in enumerate(bodies):
    mujoco.mj_objectVelocity(
      model, data, mujoco.mjtObj.mjOBJ_XBODY, model.body(name).id, velocity, 0
    )
    np.testing.assert_allclose(linear[i].detach().numpy(), velocity[3:], atol=1e-8)
    np.testing.assert_allclose(angular[i].detach().numpy(), velocity[:3], atol=1e-8)
  args[2].requires_grad_()
  args[5].requires_grad_()
  _, _, linear, angular = fk(*args)
  (linear.square().sum() + angular.square().sum()).backward()
  assert torch.isfinite(args[2].grad).all()
  assert torch.isfinite(args[5].grad).all()
  assert args[5].grad.abs().sum() > 0


@pytest.mark.parametrize(
  "device",
  ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
  ))],
)
def test_velocity_jvp_and_reverse_gradient_on_device(device):
  from supermimic.utils.differentiable_fk import DifferentiableFKWithVelocity

  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  joint_names = [
    model.joint(i).name for i in range(model.njnt)
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
  ]
  fk = DifferentiableFKWithVelocity(DifferentiableFK(
    model, "pelvis", joint_names, ["left_wrist_yaw_link", "right_ankle_roll_link"]
  )).to(device)
  q = (torch.randn(4, len(joint_names), device=device) * 0.1).requires_grad_()
  dq = (torch.randn_like(q) * 0.1).requires_grad_()
  root = torch.zeros(4, 3, device=device)
  orientation = torch.tensor([1., 0., 0., 0.], device=device).expand(4, -1)
  pos, quat, linear, angular = fk(root, orientation, q, root, root, dq)
  loss = pos.square().sum() + quat.square().sum() + linear.square().sum() + angular.square().sum()
  loss.backward()
  assert torch.isfinite(loss)
  assert torch.isfinite(q.grad).all() and torch.isfinite(dq.grad).all()
  assert dq.grad.abs().sum() > 0
