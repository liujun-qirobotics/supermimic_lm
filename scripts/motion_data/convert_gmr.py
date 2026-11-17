"""Convert named GMR qpos to research mjlab's reference motion format."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
from mjlab.scripts.csv_to_npz import JOINT_NAMES, MotionLoader


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("source", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--fps", type=int, default=50)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()
  with np.load(args.source, allow_pickle=False) as data:
    qpos = data["qpos"].copy()
    names = data["joint_names"].tolist()
    fps = float(data["fps"])
    if str(data["quaternion_order"]) != "wxyz":
      raise ValueError("Expected wxyz input")
  if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 3:
    raise ValueError("Expected at least three frames of free root and 29 joints")
  if len(names) != 29 or len(set(names)) != 29 or set(names) != set(JOINT_NAMES):
    raise ValueError("GMR joint names do not match research mjlab G1 joint names")
  if not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0 or args.fps <= 0:
    raise ValueError("Motion and frame rates must be finite and valid")
  if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-3):
    raise ValueError("Expected normalized root quaternions")
  indices = [names.index(name) for name in JOINT_NAMES]
  csv = np.concatenate(
    (qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:][:, indices]), axis=1
  )
  with tempfile.TemporaryDirectory() as temp:
    path = Path(temp) / "motion.csv"
    np.savetxt(path, csv, delimiter=",")
    motion = MotionLoader(str(path), fps, args.fps, device=args.device)
    result = motion.arrays()
  if len(result["joint_pos"]) < 3:
    raise ValueError("Output is too short for velocity computation")
  if not all(np.isfinite(result[key]).all() for key in result if key != "joint_names"):
    raise ValueError("Converted motion contains non-finite values")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(args.output, **result)
  print(f"Saved {len(result['joint_pos'])} frames to {args.output}")


if __name__ == "__main__":
  main()
