"""Reduce outward foot yaw with leg IK while approximately retaining foot position."""
import argparse
from pathlib import Path
import mujoco,numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--max-outward-deg',type=float,default=20);p.add_argument('--max-correction-deg',type=float,default=20);a=p.parse_args()
 if not 0<a.max_outward_deg<90 or not 0<a.max_correction_deg<=45:p.error('invalid angles')
 with np.load(a.source,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
 q=data['qpos'].copy();m=mujoco.MjModel.from_xml_path(str(G1_XML));d=mujoco.MjData(m); groups={}
 for s in ('left','right'):
  js=[m.joint(s+'_'+n+'_joint') for n in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
  groups[s]={'q':np.array([j.qposadr[0] for j in js]),'lo':np.array([j.range[0] for j in js])+.001,'hi':np.array([j.range[1] for j in js])-.001,'foot':m.body(s+'_ankle_roll_link').id}
 yaw=np.empty((len(q),2))
 for i,state in enumerate(q):
  d.qpos[:]=state;mujoco.mj_kinematics(m,d);R=d.body('pelvis').xmat.reshape(3,3)
  for k,s in enumerate(('left','right')):
   F=R.T@d.xmat[groups[s]['foot']].reshape(3,3);yaw[i,k]=np.arctan2(F[1,0],F[0,0])*(1 if k==0 else -1)
 excess=np.clip(yaw-np.deg2rad(a.max_outward_deg),0,np.deg2rad(a.max_correction_deg));delta=gaussian_filter1d(excess,12,axis=0,mode='nearest')
 for i in range(len(q)):
  d.qpos[:]=q[i];mujoco.mj_kinematics(m,d);R=d.body('pelvis').xmat.reshape(3,3).copy()
  for k,s in enumerate(('left','right')):
   if delta[i,k]<1e-4:continue
   g=groups[s];b=g['foot'];xyz=d.xpos[b].copy();old=d.xmat[b].reshape(3,3).copy();target=Rotation.from_rotvec(R[:,2]*(-1 if k==0 else 1)*delta[i,k]).as_matrix()@old;base=q[i,g['q']].copy()
   def res(x):
    d.qpos[g['q']]=x;mujoco.mj_kinematics(m,d);rot=Rotation.from_matrix(target.T@d.xmat[b].reshape(3,3)).as_rotvec()
    return np.r_[15*(d.xpos[b]-xyz),rot,.10*(x-base)]
   z=least_squares(res,np.clip(base,g['lo'],g['hi']),bounds=(g['lo'],g['hi']),max_nfev=20,ftol=1e-5,gtol=1e-5,xtol=1e-5)
   q[i,g['q']]=z.x;d.qpos[g['q']]=z.x
 data['qpos']=q;data['toe_heading_adjustment']='relative pelvis outward correction, soft target';data['max_outward_target_deg']=a.max_outward_deg
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**data)
 print(a.source.name,'saved',a.output,'max target correction deg',np.degrees(delta.max()),flush=True)
if __name__=='__main__':main()
