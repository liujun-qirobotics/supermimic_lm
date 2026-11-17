"""Compatibility metadata for fixed joint-task evaluation datasets."""

import hashlib
import json


def static_model_signature(runner):
  """Hash static geometry, kinematics, joint order and action scaling."""
  model = runner.env.sim.mj_model
  digest = hashlib.sha256()
  digest.update(json.dumps(list(runner.env.scene["robot"].joint_names)).encode())
  for name in ("body_parentid", "body_pos", "body_quat", "jnt_type", "jnt_axis",
               "jnt_pos", "geom_bodyid", "geom_type", "geom_pos", "geom_quat", "geom_size"):
    array = getattr(model, name)
    digest.update(name.encode())
    digest.update(array.tobytes())
  for tensor in (runner.wm.dynamics.action_scale, runner.wm.dynamics.action_offset):
    digest.update(tensor.detach().cpu().numpy().tobytes())
  return digest.hexdigest()
