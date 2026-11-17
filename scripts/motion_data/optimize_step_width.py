"""Gently widen narrow G1 ankle trajectories with leg-only IK."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--min-lateral',type=float,default=.085);p.add_argument('--max-shift',type=float,default=.035);a=p.parse_args()
 with np.load(a.source,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
 q=data['qpos'].copy();m=mujoco.MjModel.from_xml_path(str(data['robot_xml']));d=mujoco.MjData(m);sides={}
 for side in ('left','right'):
  joints=[m.joint(f'{side}_{n}_joint') for n in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
  sides[side]={'q':np.array([j.qposadr[0] for j in joints]),'lower':np.array([j.range[0] for j in joints])+.001,'upper':np.array([j.range[1] for j in joints])-.001,'body':m.body(f'{side}_ankle_roll_link').id}
 local=np.empty((len(q),2))
 for i,state in enumerate(q):
  d.qpos[:]=state;mujoco.mj_kinematics(m,d);R=d.body('pelvis').xmat.reshape(3,3);root=d.body('pelvis').xpos
  for k,side in enumerate(('left','right')):local[i,k]=(R.T@(d.xpos[sides[side]['body']]-root))[1]*(1 if k==0 else -1)
 shift=np.clip(a.min_lateral-local,0,a.max_shift);shift=gaussian_filter1d(shift,8,axis=0,mode='nearest')
 prev={}
 for i in range(len(q)):
  d.qpos[:]=q[i];mujoco.mj_kinematics(m,d);R=d.body('pelvis').xmat.reshape(3,3).copy()
  for k,side in enumerate(('left','right')):
   if shift[i,k]<.0005:continue
   x=sides[side];bid=x['body'];origin=d.xpos[bid].copy();rot=d.xmat[bid].reshape(3,3).copy();target=origin+R[:,1]*shift[i,k]*(1 if k==0 else -1);base=q[i,x['q']].copy();previous=prev.get(side,base)
   def fun(v):
    d.qpos[x['q']]=v;mujoco.mj_kinematics(m,d);ori=Rotation.from_matrix(rot.T@d.xmat[bid].reshape(3,3)).as_rotvec()
    return np.r_[10*(d.xpos[bid]-target),1.0*ori,.05*(v-base),.02*(v-previous)]
   result=least_squares(fun,np.clip(base,x['lower'],x['upper']),bounds=(x['lower'],x['upper']),max_nfev=15,ftol=1e-5,xtol=1e-5,gtol=1e-5)
   q[i,x['q']]=result.x;prev[side]=result.x
  if (i+1)%1000==0:print(f'{i+1}/{len(q)}',flush=True)
 data['qpos']=q;data['step_width_target_m']=a.min_lateral;data['step_width_max_shift_m']=a.max_shift;data['step_width_method']='smoothed lateral ankle target, leg-only IK'
 np.savez_compressed(a.output,**data)
 print(json.dumps({'mean_requested_shift_m':shift.mean(axis=0).tolist(),'max_requested_shift_m':shift.max(axis=0).tolist(),'frames_changed':(shift>.0005).sum(axis=0).tolist()}))
if __name__=='__main__':main()
