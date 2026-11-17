"""Retarget trimmed 100STYLE BVH with GMR's Xsens G1 IK configuration."""
import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from general_motion_retargeting import GeneralMotionRetargeting, IK_CONFIG_DICT
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
from general_motion_retargeting.utils.lafan_vendor import utils


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('source', type=Path)
  p.add_argument('cuts', type=Path)
  p.add_argument('output', type=Path)
  p.add_argument('--height', type=float, default=1.75, help='assumed human height in metres')
  p.add_argument('--ground-offset', type=float, default=0.06, help='constant downward offset in metres')
  p.add_argument('--limit-frames', type=int, default=None, help='only for pipeline testing')
  p.add_argument('--offset-frames', type=int, default=0, help='offset from the trimmed start')
  p.add_argument('--ik-config', type=Path, default=Path(__file__).with_name('100style_g1_ik.json'))
  args = p.parse_args()
  style, movement = args.source.stem.rsplit('_', 1)
  with args.cuts.open(newline='', encoding='utf-8-sig') as stream:
    row = next((x for x in csv.DictReader(stream) if x['STYLE_NAME'] == style), None)
  if row is None:
    raise ValueError('Style missing from frame cuts')
  start, stop = int(row[f'{movement}_START']), int(row[f'{movement}_STOP'])
  start += args.offset_frames
  if args.limit_frames is not None:
    stop = min(stop, start + args.limit_frames)
  text = args.source.read_text()
  match = __import__('re').search(r'^Frame Time:\s*([0-9.]+)', text, __import__('re').M)
  if not match:
    raise ValueError('BVH frame time missing')
  fps = 1.0 / float(match.group(1))
  anim = read_bvh(str(args.source))
  if not 0 <= start < stop <= len(anim.pos) or stop-start < 3:
    raise ValueError('invalid frame range')
  global_quat, global_pos = utils.quat_fk(anim.quats[start:stop], anim.pos[start:stop], anim.parents)
  names = anim.bones
  required = {'Hips','Chest4','LeftHip','RightHip','LeftKnee','RightKnee','LeftAnkle','RightAnkle','LeftShoulder','RightShoulder','LeftElbow','RightElbow','LeftWrist','RightWrist'}
  if not required.issubset(names):
    raise ValueError(f'Missing expected bones: {required - set(names)}')
  transform = Rotation.from_euler('x', 90, degrees=True)
  world_q = transform.as_quat(scalar_first=True)
  world_q_inverse = world_q * np.array([1, -1, -1, -1])
  world_rot = transform.as_matrix()
  # In 100STYLE the local forward axis is -Y after conversion to Z-up;
  # G1 faces +X. Right-multiplication changes the body frame, not the path.
  heading_alignment = Rotation.from_euler('z', -90, degrees=True).as_quat(scalar_first=True)
  IK_CONFIG_DICT['bvh_xsens']['unitree_g1'] = args.ik_config.resolve()
  retarget = GeneralMotionRetargeting(src_human='bvh_xsens', tgt_robot='unitree_g1', actual_human_height=args.height)
  model = mujoco.MjModel.from_xml_path(str(retarget.xml_file))
  joints = [model.joint(i).name for i in range(1, model.njnt)]
  if model.nq != 36 or len(joints) != 29:
    raise ValueError('Expected G1 with 29 single-coordinate joints')
  results=[]
  for index in range(stop-start):
    frame={name: [global_pos[index, j] @ world_rot.T / 100.0,
                  utils.quat_mul(utils.quat_mul(world_q, global_quat[index,j]), world_q_inverse)]
           for j,name in enumerate(names)}
    for name in ('Hips', 'Chest4', 'LeftAnkle', 'RightAnkle'):
      frame[name][1] = utils.quat_mul(frame[name][1], heading_alignment)
    frame['LeftFootMod'] = frame['LeftAnkle']
    frame['RightFootMod'] = frame['RightAnkle']
    results.append(retarget.retarget(frame).copy())
    if (index+1)%300 == 0:
      print(f'Retargeted {index+1}/{stop-start}', flush=True)
  qpos=np.stack(results)
  if not np.isfinite(qpos).all():
    raise ValueError('non-finite retargeted motion')
  qpos[:, 2] -= args.ground_offset
  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(args.output, qpos=qpos, joint_names=np.asarray(joints), fps=fps,
                      source=str(args.source.resolve()), start_frame=start, stop_frame_exclusive=stop,
                      robot_xml=str(retarget.xml_file), quaternion_order='wxyz', assumed_human_height_m=args.height,
                      ground_offset_m=args.ground_offset, ik_config=str(args.ik_config.resolve()))
  print(f'Saved {len(qpos)} frames, {len(qpos)/fps:.3f}s at {fps:.5f}fps to {args.output}', flush=True)


if __name__ == '__main__':
  main()
