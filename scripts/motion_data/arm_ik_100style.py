"""Collision-aware 100STYLE arm fit with an explicit neutral palm convention."""
from __future__ import annotations
import numpy as np
import mujoco
from scipy.optimize import least_squares


class ArmDirectionIK:
  """Keep locomotion fixed; fit all seven arm DOFs with torso clearance."""
  def __init__(self, model, fps=60.0, max_speed=3.0):
    self.model = model
    if fps <= 0 or max_speed <= 0:
      raise ValueError('fps and max_speed must be positive')
    self.max_step = max_speed / fps
    self.data = mujoco.MjData(model)
    self.torso = model.body('torso_link').id
    self.sides = {}
    obstacles = [i for i in range(model.ngeom) if model.geom(i).name in
                 ('torso_collision','pelvis_collision')]
    if len(obstacles) != 2:
      raise ValueError('Expected G1 torso and pelvis collision geometry')
    obstacles += [i for i in range(model.ngeom) if model.geom(i).name in
                  tuple(f'{s}_{part}_collision' for s in ('left','right')
                        for part in ('hip','thigh','shin'))]
    def mesh_geoms(names):
      mesh_ids = {model.mesh(n).id for n in names}
      return [i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
              and model.geom_dataid[i] in mesh_ids]
    leg_meshes = mesh_geoms([f'{s}_{part}_link' for s in ('left','right')
                            for part in ('hip_roll','hip_yaw','knee')])
    for side in ('left','right'):
      joints=[model.joint(f'{side}_{name}_joint') for name in
              ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow')]
      wrist=[model.joint(f'{side}_wrist_{name}_joint') for name in ('roll','pitch','yaw')]
      all_joints=joints+wrist
      geoms=[i for i in range(model.ngeom) if model.geom(i).name in
             (f'{side}_shoulder_yaw_collision',f'{side}_elbow_yaw_collision',
              f'{side}_wrist_collision',f'{side}_hand_collision')]
      if len(geoms)!=4:
        raise ValueError('Expected G1 arm collision geometry')
      hand_meshes = mesh_geoms([f'{side}_rubber_hand', f'{side}_wrist_pitch_link', f'{side}_wrist_yaw_link'])
      self.sides[side]={
        'qadr':np.array([j.qposadr[0] for j in joints]),
        'wrist_qadr':np.array([j.qposadr[0] for j in wrist]),
        'all_qadr':np.array([j.qposadr[0] for j in all_joints]),
        'lower':np.array([j.range[0] for j in all_joints])+0.02,
        'upper':np.array([j.range[1] for j in all_joints])-0.02,
        'body':[model.body(f'{side}_{name}_link').id for name in
                ('shoulder_pitch','elbow','wrist_yaw')],
        'pairs':[(a,b) for a in geoms for b in obstacles] +
                [(a,b) for a in hand_meshes for b in leg_meshes]}
    self.previous={}

  def fit(self, qpos, targets):
    self.data.qpos[:]=qpos
    mujoco.mj_kinematics(self.model,self.data)
    torso_R=self.data.xmat[self.torso].reshape(3,3).copy()
    torso_pos=self.data.xpos[self.torso].copy()
    for side in ('left','right'):
      d=self.sides[side]; sign=1 if side=='left' else -1
      target=np.asarray(targets[side],dtype=float)
      lengths=np.linalg.norm(target,axis=1)
      if np.any(lengths<1e-6): raise ValueError('zero-length arm vector')
      target=target/lengths[:,None]
      neutral=np.array([0.,sign*0.18,0.,1.45,0.,0.,0.])
      prior=self.previous.get(side,neutral)
      lower, upper = d['lower'], d['upper']
      if side in self.previous:
        lower = np.maximum(lower, prior-self.max_step)
        upper = np.minimum(upper, prior+self.max_step)
      initial=np.clip(prior,lower+1e-6,upper-1e-6)
      def residual(x):
        self.data.qpos[d['all_qadr']]=x
        mujoco.mj_kinematics(self.model,self.data)
        points=self.data.xpos[d['body']]
        vec=np.diff(points,axis=0)
        unit=vec/np.maximum(np.linalg.norm(vec,axis=1,keepdims=True),1e-9)
        hand=self.data.xmat[d['body'][-1]].reshape(3,3)
        # The rubber-hand mesh points along +X, its broad face is XZ.
        # Align its lateral axis with the torso's lateral axis, keeping
        # the paired palms on their inward-facing neutral side.
        palm_lateral=hand[:,1]-torso_R[:,1]
        distal=hand[:,0]-unit[1]
        gaps=np.array([mujoco.mj_geomDistance(self.model,self.data,a,b,0.2,None)
                       for a,b in d['pairs']])
        lateral=(points[1:]-torso_pos)@torso_R[:,1]*sign
        return np.concatenate((0.7*(unit-target).ravel(),0.8*palm_lateral,
                               0.45*distal,65*np.maximum(0.025-gaps,0),
                               15*np.maximum(np.array([0.17,0.23])-lateral,0),
                               0.06*(x-prior),np.array([.02,.02,.12,.02,.02,.02,.02])*(x-neutral)))
      result=least_squares(residual,initial,bounds=(lower,upper),
                           max_nfev=25,ftol=1e-4,xtol=1e-4,gtol=1e-4)
      if not np.isfinite(result.x).all(): raise ValueError('non-finite IK')
      self.data.qpos[d['all_qadr']]=result.x
      self.previous[side]=result.x.copy()
    return self.data.qpos.copy()
