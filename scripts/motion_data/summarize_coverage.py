"""Summarize provisional motion command coverage in the pelvis frame."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("motion_dir", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  totals = dict.fromkeys(
    ("stationary", "forward", "backward", "left", "right", "turn_left", "turn_right"),
    0.0,
  )
  durations = []
  for path in sorted(args.motion_dir.glob("*.npz")):
    with np.load(path, allow_pickle=False) as data:
      quat = data["root_quat_w"]
      rot = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
      velocity = rot.inv().apply(data["root_lin_vel_w"])
      omega = rot.inv().apply(data["root_ang_vel_w"])
      fps = float(data["fps"][0])
      vx, vy, wz = velocity[:, 0], velocity[:, 1], omega[:, 2]
      conditions = {
        "stationary": (np.hypot(vx, vy) < 0.1) & (np.abs(wz) < 0.2),
        "forward": vx > 0.1,
        "backward": vx < -0.1,
        "left": vy > 0.1,
        "right": vy < -0.1,
        "turn_left": wz > 0.2,
        "turn_right": wz < -0.2,
      }
      for name, mask in conditions.items():
        totals[name] += float(mask.sum()) / fps
      durations.append(len(vx) / fps)
  result = {
    "clips": len(durations),
    "total_duration_s": sum(durations),
    "command_component_seconds": totals,
    "note": "Non-exclusive thresholds; does not establish command-response quality or visual validity.",
  }
  args.output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
