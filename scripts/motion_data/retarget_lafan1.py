"""Retarget LAFAN1 BVH to named G1 generalized positions with the GMR environment.

Run with GMR's Python; output is an intermediate, not a SuperMimic motion file.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("source", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--start", type=int, default=0)
  parser.add_argument("--frames", type=int, default=None)
  args = parser.parse_args()
  if args.start < 0 or (args.frames is not None and args.frames < 2):
    parser.error("start must be nonnegative and frames must be at least 2")
  frames, height = load_bvh_file(str(args.source))
  selected = frames[args.start : args.start + args.frames if args.frames else None]
  if len(selected) < 2:
    parser.error("selection must contain at least two frames")
  retarget = GeneralMotionRetargeting(
    src_human="bvh_lafan1", tgt_robot="unitree_g1", actual_human_height=height
  )
  model = mujoco.MjModel.from_xml_path(str(retarget.xml_file))
  joints = [model.joint(i).name for i in range(1, model.njnt)]
  if model.nq != 7 + len(joints) or len(joints) != 29:
    raise ValueError("Expected G1 free root and 29 single-coordinate joints")
  results = []
  for index, frame in enumerate(selected, start=1):
    results.append(retarget.retarget(frame).copy())
    if index % 300 == 0:
      print(f"Retargeted {index}/{len(selected)} frames", flush=True)
  qpos = np.stack(results)
  if not np.isfinite(qpos).all():
    raise ValueError("Non-finite retarget result")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    args.output,
    qpos=qpos,
    joint_names=np.asarray(joints),
    fps=30.0,
    source=str(args.source.resolve()),
    start_frame=args.start,
    robot_xml=str(retarget.xml_file),
    quaternion_order="wxyz",
  )
  print(f"Saved {len(qpos)} frames to {args.output}")


if __name__ == "__main__":
  main()
