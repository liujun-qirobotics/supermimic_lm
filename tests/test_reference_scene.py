import importlib.util
from pathlib import Path

import mujoco


def test_reference_scene_matches_tracking_environment():
  path = Path(__file__).resolve().parents[1] / 'scripts/motion_data/view_reference.py'
  spec = importlib.util.spec_from_file_location('view_reference', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  model = module.scene_model()
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'terrain') >= 0
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot/pelvis') >= 0
  assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'robot/left_knee_joint') >= 0
