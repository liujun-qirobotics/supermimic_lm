import importlib.util
from pathlib import Path
import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML


def test_arm_fit_preserves_body_and_has_clearance_and_lateral_palms():
  path = Path(__file__).resolve().parents[1] / 'scripts/motion_data/arm_ik_100style.py'
  spec = importlib.util.spec_from_file_location('arm_ik_100style',path)
  module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
  model = mujoco.MjModel.from_xml_path(str(G1_XML)); solver=module.ArmDirectionIK(model)
  state=mujoco.MjData(model).qpos.copy(); state[2]=0.8; state[7]=0.2
  targets={'left':np.array([[0,-.25,-1],[0,-.1,-1]]),
           'right':np.array([[0,.25,-1],[0,.1,-1]])}
  result=solver.fit(state,targets)
  indices=np.concatenate([solver.sides[s]['all_qadr'] for s in ('left','right')])
  np.testing.assert_array_equal(np.delete(result,indices),np.delete(state,indices))
  solver.data.qpos[:]=result; mujoco.mj_kinematics(model,solver.data)
  torso=solver.data.xmat[solver.torso].reshape(3,3)
  for side in ('left','right'):
    d=solver.sides[side]
    assert any(model.geom_type[a] == mujoco.mjtGeom.mjGEOM_MESH and
               model.geom_type[b] == mujoco.mjtGeom.mjGEOM_MESH for a,b in d['pairs'])
    assert any(model.geom(b).name == 'left_thigh_collision' for a,b in d['pairs'])
    for a,b in d['pairs']:
      assert mujoco.mj_geomDistance(model,solver.data,a,b,.2,None)>0.005
    hand=solver.data.xmat[d['body'][-1]].reshape(3,3)
    assert hand[:,1]@torso[:,1]>.95
    assert np.all(result[d['all_qadr']]>=d['lower'])
    assert np.all(result[d['all_qadr']]<=d['upper'])
  changed={'left':np.array([[0,.5,-1],[.5,.5,-1]]),
           'right':np.array([[0,-.5,-1],[.5,-.5,-1]])}
  next_result=solver.fit(state,changed)
  assert np.max(abs(next_result[indices]-result[indices])) <= 3.0/60 + 1e-7
