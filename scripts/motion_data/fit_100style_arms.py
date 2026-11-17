"""Refine arms in an existing 100STYLE G1 motion while preserving body/legs."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
from general_motion_retargeting.utils.lafan_vendor import utils
from arm_ik_100style import ArmDirectionIK


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('source',type=Path,help='previous GMR qpos npz')
  p.add_argument('bvh',type=Path)
  p.add_argument('output',type=Path)
  p.add_argument('--limit-frames',type=int,default=None)
  p.add_argument('--robot-xml',type=Path,default=Path(__file__).resolve().parents[3] / 'mjlab_research/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml')
  args=p.parse_args()
  with np.load(args.source,allow_pickle=False) as f:
    saved={k:f[k].copy() for k in f.files}
  q=saved['qpos']; start=int(saved['start_frame']); stop=int(saved['stop_frame_exclusive'])
  if args.limit_frames is not None:
    stop=min(stop,start+args.limit_frames); q=q[:stop-start].copy()
  anim=read_bvh(str(args.bvh))
  if stop>len(anim.pos) or len(q)!=stop-start:
    raise ValueError('source motion does not match BVH frame range')
  _, pos=utils.quat_fk(anim.quats[start:stop],anim.pos[start:stop],anim.parents)
  pos=pos @ Rotation.from_euler('x',90,degrees=True).as_matrix().T / 100
  ids={name:anim.bones.index(name) for name in ('LeftShoulder','LeftElbow','LeftWrist','RightShoulder','RightElbow','RightWrist')}
  model=mujoco.MjModel.from_xml_path(str(args.robot_xml))
  names=[model.joint(i).name for i in range(1,model.njnt)]
  if names != saved['joint_names'].tolist():
    raise ValueError('reference joint order differs from model')
  solver=ArmDirectionIK(model, fps=float(saved['fps']))
  output=[]; before=[]; after=[]
  def direction_errors(qpos,targets):
    solver.data.qpos[:]=qpos; mujoco.mj_kinematics(model,solver.data)
    errors=[]
    for side in ('left','right'):
      pts=solver.data.xpos[solver.sides[side]['body']]; v=np.diff(pts,axis=0); v/=np.linalg.norm(v,axis=1)[:,None]
      t=targets[side]/np.linalg.norm(targets[side],axis=1)[:,None]
      errors.extend(np.degrees(np.arccos(np.clip((v*t).sum(axis=1),-1,1))).tolist())
    return errors
  for i,state in enumerate(q):
    targets={}
    for side in ('left','right'):
      prefix=side.title(); s,e,w=(pos[i,ids[prefix+x]] for x in ('Shoulder','Elbow','Wrist'))
      targets[side]=np.stack((e-s,w-e))
    before.append(direction_errors(state,targets))
    result=solver.fit(state,targets)
    after.append(direction_errors(result,targets)); output.append(result)
    if (i+1)%300==0:
      print(f'Arm refinement {i+1}/{len(q)}',flush=True)
  saved['qpos']=np.stack(output); saved['stop_frame_exclusive']=np.asarray(stop)
  saved['arm_fit']='arm_body_leg_and_hand_mesh_clearance_v4'
  saved['robot_xml']=str(args.robot_xml.resolve())
  args.output.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(args.output,**saved)
  report={'samples':len(q),'angle_errors_deg_mean_before':np.mean(before,axis=0).tolist(),
          'angle_errors_deg_mean_after':np.mean(after,axis=0).tolist(),
          'order':['left_upper','left_forearm','right_upper','right_forearm'],
          'wrist':'seven-DOF fit; neutral inward palm convention, not captured human wrist orientation',
          'max_arm_joint_speed_rad_s':3.0,
          'body_legs':'copied unchanged from source motion'}
  args.output.with_suffix('.arm_fit.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
