"""Generate provisional clips from low-anomaly reference spans.

Heuristics are not physical or visual validation. All output requires review.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML


def digest(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_mask(data, model):
  n = len(data["joint_pos"])
  names = data["joint_names"].tolist()
  if len(names) != 29 or len(set(names)) != 29:
    raise ValueError("expected 29 unique joint names")
  ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
  if any(i < 0 for i in ids):
    raise ValueError("unknown joint")
  q, qv = data["joint_pos"], data["joint_vel"]
  root, quat = data["root_pos_w"], data["root_quat_w"]
  vel, ang = data["root_lin_vel_w"], data["root_ang_vel_w"]
  if (
    q.shape != (n, 29)
    or qv.shape != q.shape
    or root.shape != (n, 3)
    or quat.shape != (n, 4)
    or vel.shape != (n, 3)
    or ang.shape != (n, 3)
  ):
    raise ValueError("inconsistent shapes")
  finite = np.isfinite(np.concatenate((q, qv, root, quat, vel, ang), axis=1)).all(
    axis=1
  )
  limits = model.jnt_range[ids]
  limited = model.jnt_limited[ids].astype(bool)
  excess = np.maximum(limits[:, 0] - q, q - limits[:, 1])[:, limited].max(axis=1)
  gravity_z = 2 * np.square(quat[:, 1:3]).sum(axis=1) - 1
  good = finite & (np.abs(np.linalg.norm(quat, axis=1) - 1) < 1e-3)
  good &= (excess <= 0.05) & (np.abs(qv).max(axis=1) <= 20)
  good &= (np.linalg.norm(vel[:, :2], axis=1) <= 2) & (np.abs(ang[:, 2]) <= 4)
  good &= (root[:, 2] >= 0.55) & (root[:, 2] <= 1.05) & (gravity_z <= -0.707)
  foot = [
    i
    for i in range(model.ngeom)
    if (model.geom(i).name or "").startswith(("left_foot", "right_foot"))
    and (model.geom(i).name or "").endswith("_collision")
  ]
  if len(foot) != 14 or any(
    model.geom_type[i] != mujoco.mjtGeom.mjGEOM_CAPSULE for i in foot
  ):
    raise ValueError("expected G1 foot capsule collision geoms")
  mj = mujoco.MjData(model)
  for t in np.flatnonzero(good):
    mj.qpos[:3], mj.qpos[3:7] = root[t], quat[t]
    mj.qpos[model.jnt_qposadr[ids]] = q[t]
    mujoco.mj_kinematics(model, mj)
    # Lowest point of each capsule relative to the assumed z=0 ground plane.
    bottom = [
      mj.geom_xpos[i, 2]
      - model.geom_size[i, 0]
      - model.geom_size[i, 1] * abs(mj.geom_xmat[i].reshape(3, 3)[2, 2])
      for i in foot
    ]
    if min(bottom) < -0.03 or min(bottom) > 0.10:
      good[t] = False
  return good


def main():
  p = argparse.ArgumentParser()
  p.add_argument("motion_dir", type=Path)
  p.add_argument("--output-dir", required=True, type=Path)
  p.add_argument("--min-seconds", type=float, default=2.0)
  p.add_argument("--max-seconds", type=float, default=20.0)
  p.add_argument("--guard-seconds", type=float, default=0.1)
  args = p.parse_args()
  if not 0 < args.min_seconds <= args.max_seconds or args.guard_seconds < 0:
    p.error("invalid durations")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  records = []
  keys = (
    "root_pos_w",
    "root_quat_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "joint_pos",
    "joint_vel",
  )
  for source in sorted(args.motion_dir.glob("*.npz")):
    with np.load(source, allow_pickle=False) as data:
      fps = float(data["fps"][0])
      good = frame_mask(data, model)
      guard = round(args.guard_seconds * fps)
      if guard:
        bad = np.pad((~good).astype(np.int32), (guard, guard))
        good = (
          np.convolve(bad, np.ones(2 * guard + 1, dtype=np.int32), mode="valid") == 0
        )
      edges = np.diff(np.pad(good.astype(np.int8), (1, 1)))
      spans = zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
      minimum, maximum = round(args.min_seconds * fps), round(args.max_seconds * fps)
      source_hash = digest(source)
      for start, end in spans:
        for lo in range(int(start), int(end), maximum):
          hi = min(lo + maximum, int(end))
          if hi - lo < minimum:
            continue
          out = args.output_dir / f"{source.stem}__{lo:06d}_{hi:06d}.npz"
          if out.exists():
            raise FileExistsError(f"Refusing to overwrite {out}")
          arrays = {key: data[key][lo:hi].copy() for key in keys}
          arrays.update(fps=data["fps"].copy(), joint_names=data["joint_names"].copy())
          np.savez_compressed(out, **arrays)
          records.append(
            {
              "file": out.name,
              "source": source.name,
              "source_sha256": source_hash,
              "start_frame": lo,
              "end_frame_exclusive": hi,
              "duration_s": (hi - lo) / fps,
              "status": "provisional_requires_visual_review",
            }
          )
    print(
      f"{source.name}: {len(good)} frames, {good.sum()} pass frame checks", flush=True
    )
  manifest = args.output_dir / "manifest.json"
  if manifest.exists():
    raise FileExistsError(f"Refusing to overwrite {manifest}")
  manifest.write_text(
    json.dumps(
      {
        "criteria": {
          "max_joint_speed_radps": 20,
          "max_root_xy_speed_mps": 2,
          "max_yaw_rate_radps": 4,
          "root_height_m": [0.55, 1.05],
          "max_tilt_deg": 45,
          "foot_min_z_m": [-0.03, 0.10],
          "max_joint_limit_excess_rad": 0.05,
          "guard_s": args.guard_seconds,
          "min_s": args.min_seconds,
          "max_s": args.max_seconds,
        },
        "clips": records,
      },
      indent=2,
    )
    + "\n"
  )
  print(f"{len(records)} provisional clips written; require visual review", flush=True)


if __name__ == "__main__":
  main()
