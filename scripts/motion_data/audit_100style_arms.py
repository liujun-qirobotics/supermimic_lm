"""Check every frame for arm/torso clearance, palm axes and preserved locomotion."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from arm_ik_100style import ArmDirectionIK


def main():
  p=argparse.ArgumentParser(description=__doc__); p.add_argument('motion',type=Path); p.add_argument('--baseline',type=Path,required=True)
  args=p.parse_args()
  with np.load(args.motion,allow_pickle=False) as f:
    q=f['qpos']; xml=str(f['robot_xml']); fps=float(f['fps'])
  with np.load(args.baseline,allow_pickle=False) as f: baseline=f['qpos']
  m=mujoco.MjModel.from_xml_path(xml);solver=ArmDirectionIK(m); gaps=[];palms=[]
  pairs=[pair for s in ('left','right') for pair in solver.sides[s]['pairs']]
  for state in q:
    solver.data.qpos[:]=state;mujoco.mj_kinematics(m,solver.data)
    gaps.append([mujoco.mj_geomDistance(m,solver.data,a,b,.2,None) for a,b in pairs])
    torso=solver.data.xmat[solver.torso].reshape(3,3)
    palms.append([solver.data.xmat[solver.sides[s]['body'][-1]].reshape(3,3)[:,1]@torso[:,1] for s in ('left','right')])
  gaps=np.array(gaps);palms=np.array(palms)
  worst_frame, worst_pair = np.unravel_index(np.argmin(gaps), gaps.shape)
  def geom_name(i):
    return m.geom(i).name or ('mesh:' + m.mesh(int(m.geom_dataid[i])).name)
  indices=np.concatenate([solver.sides[s]['all_qadr'] for s in ('left','right')])
  lower=np.concatenate([solver.sides[s]['lower'] for s in ('left','right')]);upper=np.concatenate([solver.sides[s]['upper'] for s in ('left','right')])
  report={'frames':len(q),'duration_s':len(q)/fps,
          'minimum_checked_geom_distance_m':float(gaps.min()),
          'checked_pair_count':len(pairs),
          'worst_frame':int(worst_frame),
          'worst_pair':[geom_name(i) for i in pairs[worst_pair]],
          'penetration_frames':int(np.any(gaps<0,axis=1).sum()),
          'clearance_below_5mm_frames':int(np.any(gaps<.005,axis=1).sum()),
          'max_lateral_hand_axis_error_deg':float(np.degrees(np.arccos(np.clip(palms,-1,1))).max()),
          'body_leg_max_change':float(np.abs(np.delete(q,indices,axis=1)-np.delete(baseline,indices,axis=1)).max()),
          'arm_joint_limit_violation_frames':int(np.any((q[:,indices]<lower-1e-6)|(q[:,indices]>upper+1e-6),axis=1).sum()),
          'arm_speed_p99_9_rad_s':np.percentile(abs(np.diff(q[:,indices],axis=0))*fps,99.9,axis=0).tolist(),
          'limitation':'Checks arm vs torso/pelvis/leg primitives and wrist/rubber-hand vs leg visual-mesh convex hulls; not all self-collision pairs.'}
  args.motion.with_suffix('.audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
