"""Iterative full-sequence smoothing and collision projection for G1 arms."""
import argparse
from pathlib import Path
import numpy as np
import mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from arm_ik_100style import ArmDirectionIK


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('source',type=Path);p.add_argument('output',type=Path)
 p.add_argument('--sigma',type=float,default=4.0);p.add_argument('--iterations',type=int,default=12)
 p.add_argument('--clearance',type=float,default=.015)
 args=p.parse_args()
 if args.sigma<=0 or args.iterations<1 or not 0<args.clearance<.1:p.error('invalid optimization parameters')
 with np.load(args.source,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
 model=mujoco.MjModel.from_xml_path(str(data['robot_xml']));solver=ArmDirectionIK(model)
 indices=np.concatenate([solver.sides[s]['all_qadr'] for s in ('left','right')]);q=data['qpos'].copy()
 for iteration in range(args.iterations):
  q[:,indices]=gaussian_filter1d(q[:,indices],sigma=args.sigma if iteration==0 else 1.5,axis=0,mode='nearest')
  corrected=0; mingap=.2
  for i in range(len(q)):
   solver.data.qpos[:]=q[i];mujoco.mj_kinematics(model,solver.data)
   for side in ('left','right'):
    d=solver.sides[side]; pairs=d['pairs']
    def gaps():return np.array([mujoco.mj_geomDistance(model,solver.data,a,b,.15,None) for a,b in pairs])
    g=gaps();mingap=min(mingap,float(g.min()))
    if g.min() >= args.clearance:continue
    # Only nearby geometry pairs can become active during this local projection.
    active=[pair for pair,dist in zip(pairs,g,strict=True) if dist < .06]
    prior=q[i,d['all_qadr']].copy()
    def residual(x):
     solver.data.qpos[d['all_qadr']]=x;mujoco.mj_kinematics(model,solver.data)
     dist=np.array([mujoco.mj_geomDistance(model,solver.data,a,b,.15,None) for a,b in active])
     return np.r_[x-prior,60*np.maximum(args.clearance+.003-dist,0)]
    res=least_squares(residual,prior,bounds=(d['lower'],d['upper']),max_nfev=18,ftol=1e-4,xtol=1e-4,gtol=1e-4)
    q[i,d['all_qadr']]=res.x;solver.data.qpos[d['all_qadr']]=res.x;corrected+=1
  print(f'iteration={iteration+1} minimum_before_projection={mingap:.5f} corrected_sides={corrected}',flush=True)
  if corrected==0: break
 data['qpos']=q;data['arm_trajectory_optimization']='global_smooth_iterative_collision_projection_v6'
 args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,**data)
 print(f'Saved {len(q)} frames to {args.output}',flush=True)
if __name__=='__main__':main()
