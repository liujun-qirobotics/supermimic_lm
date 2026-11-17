"""Compare approximate footfall minima in 100STYLE BVH and G1 kinematic replay."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
from general_motion_retargeting.utils.lafan_vendor import utils

def main():
 p=argparse.ArgumentParser();p.add_argument('motion',type=Path);p.add_argument('bvh',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with np.load(a.motion,allow_pickle=False) as f:q=f['qpos'];names=f['joint_names'].tolist();fps=float(f['fps']);start=int(f['start_frame']);xml=str(f['robot_xml'])
 anim=read_bvh(str(a.bvh)); quat,pos=utils.quat_fk(anim.quats[start:start+len(q)],anim.pos[start:start+len(q)],anim.parents)
 T=Rotation.from_euler('x',90,degrees=True).as_matrix();pos=pos@T.T/100
 rootR=Rotation.from_quat(quat[:,0],scalar_first=True); C=Rotation.from_matrix(T); human_R=C*rootR*C.inv()*Rotation.from_euler('z',-90,degrees=True)
 h=np.stack([pos[:,anim.bones.index(s+'Ankle')] for s in ('Left','Right')],axis=1);hp=pos[:,0]
 for_side=lambda v: np.stack([human_R.inv().apply(v[:,i]) for i in range(2)],axis=1)
 hlocal=for_side(h-hp[:,None,:]); model=mujoco.MjModel.from_xml_path(xml);d=mujoco.MjData(model);ids=[model.joint(n).qposadr[0] for n in names];r=np.empty((len(q),2,3));rlocal=np.empty_like(r)
 for i,state in enumerate(q):
  d.qpos[:7]=state[:7];d.qpos[ids]=state[7:];mujoco.mj_kinematics(model,d);R=d.body('pelvis').xmat.reshape(3,3);root=d.body('pelvis').xpos
  for k,s in enumerate(('left','right')):
   r[i,k]=d.body(s+'_ankle_roll_link').xpos;rlocal[i,k]=R.T@(r[i,k]-root)
 out={'method':'ankle height local minima, smoothed 3 frames, prominence 1 cm, separation 0.35s; kinematic proxy, not contact forces', 'fps':fps,'sides':{}}
 for k,s in enumerate(('left','right')):
  z=gaussian_filter1d(h[:,k,2],3); ix,_=find_peaks(-z,prominence=.01,distance=int(.35*fps));sgn=1 if k==0 else -1
  entry={'count':len(ix),'human_ankle_signed_lateral_median_m':float(np.median(sgn*hlocal[ix,k,1])),'robot_ankle_signed_lateral_median_m':float(np.median(sgn*rlocal[ix,k,1])), 'human_lateral_p10_m':float(np.percentile(sgn*hlocal[ix,k,1],10)),'robot_lateral_p10_m':float(np.percentile(sgn*rlocal[ix,k,1],10)), 'robot_width_median_m':float(np.median(rlocal[ix,0,1]-rlocal[ix,1,1])), 'robot_ankle_z_median_m':float(np.median(r[ix,k,2])), 'events':[{'frame':int(i),'time_s':round(i/fps,3),'human_lateral':float(sgn*hlocal[i,k,1]),'robot_lateral':float(sgn*rlocal[i,k,1]),'robot_width':float(rlocal[i,0,1]-rlocal[i,1,1])} for i in ix]}
  out['sides'][s]=entry
 a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:{x:y for x,y in v.items() if x!='events'} for k,v in out['sides'].items()},indent=2))
if __name__=='__main__':main()
