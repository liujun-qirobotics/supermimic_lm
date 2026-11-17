"""Measure candidate reference motions; thresholds flag issues for manual review."""

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML


def audit(path: Path, model: mujoco.MjModel, stride: int) -> dict:
  with np.load(path, allow_pickle=False) as m:
    q = m["joint_pos"].copy()
    qv = m["joint_vel"].copy()
    root = m["root_pos_w"].copy()
    quat = m["root_quat_w"].copy()
    vel = m["root_lin_vel_w"].copy()
    ang = m["root_ang_vel_w"].copy()
    names = m["joint_names"].tolist()
    fps = float(m["fps"][0])
  n = len(q)
  if (
    q.shape != (n, 29)
    or qv.shape != q.shape
    or root.shape != (n, 3)
    or quat.shape != (n, 4)
    or vel.shape != (n, 3)
    or ang.shape != (n, 3)
    or fps <= 0
    or n < 3
    or len(set(names)) != 29
  ):
    raise ValueError("invalid dimensions or frame rate")
  if not all(np.isfinite(x).all() for x in (q, qv, root, quat, vel, ang)):
    raise ValueError("non-finite values")
  if not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-3):
    raise ValueError("non-unit root quaternion")
  ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
  if any(i < 0 for i in ids):
    raise ValueError("unknown joints")
  data = mujoco.MjData(model)
  foot_ids = [
    [
      i
      for i in range(model.ngeom)
      if (model.geom(i).name or "").startswith(side + "_foot")
      and (model.geom(i).name or "").endswith("_collision")
    ]
    for side in ("left", "right")
  ]
  if any(not ids for ids in foot_ids):
    raise ValueError("missing foot collision geoms")
  if any(
    model.geom_type[i] != mujoco.mjtGeom.mjGEOM_CAPSULE
    for side in foot_ids
    for i in side
  ):
    raise ValueError("foot height calculation requires capsule collision geoms")
  minimum = np.full(2, np.inf)
  for t in range(0, n, stride):
    data.qpos[:3] = root[t]
    data.qpos[3:7] = quat[t]
    data.qpos[model.jnt_qposadr[ids]] = q[t]
    mujoco.mj_kinematics(model, data)
    for side, geoms in enumerate(foot_ids):
      for geom in geoms:
        bottom = (
          data.geom_xpos[geom, 2]
          - model.geom_size[geom, 0]
          - model.geom_size[geom, 1] * abs(data.geom_xmat[geom].reshape(3, 3)[2, 2])
        )
        minimum[side] = min(minimum[side], bottom)
  limits = model.jnt_range[ids]
  limited = model.jnt_limited[ids].astype(bool)
  violation = np.maximum(limits[:, 0] - q, q - limits[:, 1])[:, limited]
  excess = float(max(0.0, violation.max(initial=0)))
  return {
    "name": path.stem,
    "frames": n,
    "fps": fps,
    "duration_s": (n - 1) / fps,
    "root_speed_max_mps": float(np.linalg.norm(vel[:, :2], axis=1).max()),
    "yaw_rate_abs_max_radps": float(np.abs(ang[:, 2]).max()),
    "joint_speed_abs_max_radps": float(np.abs(qv).max()),
    "joint_limit_excess_max_rad": excess,
    "left_foot_min_z_m": float(minimum[0]),
    "right_foot_min_z_m": float(minimum[1]),
    "foot_sample_stride": stride,
    "flag_joint_limits": excess > 0.05,
    "flag_ground_penetration": bool((minimum < -0.03).any()),
    "flag_joint_speed": bool(np.abs(qv).max() > 20),
    "review_status": "needs_visual_review",
  }


def main():
  p = argparse.ArgumentParser()
  p.add_argument("motion_dir", type=Path)
  p.add_argument("--output", type=Path, required=True)
  p.add_argument("--stride", type=int, default=5)
  args = p.parse_args()
  if args.stride < 1:
    p.error("stride must be positive")
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  rows = []
  for path in sorted(args.motion_dir.glob("*.npz")):
    try:
      row = audit(path, model, args.stride)
    except (ValueError, KeyError, OSError) as exc:
      row = {"name": path.stem, "review_status": "invalid", "error": str(exc)}
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
  if rows:
    keys = sorted({key for row in rows for key in row})
    with args.output.with_suffix(".csv").open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=keys)
      writer.writeheader()
      writer.writerows(rows)


if __name__ == "__main__":
  main()
