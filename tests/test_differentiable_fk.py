import mujoco
import numpy as np
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
