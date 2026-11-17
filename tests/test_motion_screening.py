"""Numerical checks for provisional motion filtering."""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.scripts.csv_to_npz import JOINT_NAMES


def test_frame_filter_rejects_excessive_joint_speed():
  script = Path(__file__).resolve().parents[1] / "scripts/motion_data/screen_clips.py"
  spec = importlib.util.spec_from_file_location("motion_screen_clips", script)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  ids = [model.joint(name).id for name in JOINT_NAMES]
  q = np.tile(model.qpos0[model.jnt_qposadr[ids]], (3, 1))
  qvel = np.zeros_like(q)
  qvel[1, 0] = 21.0
  data = {
    "joint_names": np.array(JOINT_NAMES),
    "joint_pos": q,
    "joint_vel": qvel,
    "root_pos_w": np.tile([0.0, 0.0, 0.8], (3, 1)),
    "root_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
    "root_lin_vel_w": np.zeros((3, 3)),
    "root_ang_vel_w": np.zeros((3, 3)),
  }
  mask = module.frame_mask(data, model)
  assert mask.tolist() == [True, False, True]
