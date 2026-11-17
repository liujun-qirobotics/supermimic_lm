"""Smooth arm joint trajectories offline while preserving root and leg poses."""
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
import mujoco
from arm_ik_100style import ArmDirectionIK


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('source',type=Path); p.add_argument('output',type=Path)
  p.add_argument('--sigma-frames',type=float,default=3.0)
  args=p.parse_args()
  if args.sigma_frames <= 0: p.error('sigma-frames must be positive')
  with np.load(args.source,allow_pickle=False) as f:
    data={key:f[key].copy() for key in f.files}
  names=data['joint_names'].tolist()
  indices=[7+i for i,n in enumerate(names) if 'shoulder_' in n or 'elbow_' in n or 'wrist_' in n]
  if len(indices)!=14: raise ValueError('Expected 14 arm joints')
  data['qpos'][:,indices]=gaussian_filter1d(data['qpos'][:,indices],sigma=args.sigma_frames,axis=0,mode='nearest')
  original=data['qpos'].copy()
  with np.load(args.source,allow_pickle=False) as f:
    original[:,indices]=f['qpos'][:,indices]
  model=mujoco.MjModel.from_xml_path(str(data['robot_xml']))
  solver=ArmDirectionIK(model)
  pairs=[pair for s in ('left','right') for pair in solver.sides[s]['pairs']]
  # Expand blends around close approaches to avoid abrupt transitions.
  for iteration in range(4):
    gaps=[]
    for state in data['qpos']:
      solver.data.qpos[:]=state; mujoco.mj_kinematics(model,solver.data)
      gaps.append(min(mujoco.mj_geomDistance(model,solver.data,a,b,.2,None) for a,b in pairs))
    gaps=np.asarray(gaps)
    if gaps.min() >= .019: break
    protect=np.clip((.022-gaps)/.012,0,1)
    protect=maximum_filter1d(protect,size=11,mode='nearest')
    protect=gaussian_filter1d(protect,sigma=3,mode='nearest')
    data['qpos'][:,indices]=(1-protect[:,None])*data['qpos'][:,indices]+protect[:,None]*original[:,indices]
  print(f'Final minimum checked gap: {gaps.min():.4f} m',flush=True)
  data['arm_smoothing_sigma_frames']=np.asarray(args.sigma_frames)
  args.output.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(args.output,**data)
  print(f'Saved {len(data["qpos"])} frames to {args.output}')
if __name__=='__main__': main()
