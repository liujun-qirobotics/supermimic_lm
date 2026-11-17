"""Inspect a reference trajectory in MuJoCo's native GUI (kinematic replay)."""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from mjlab.scene import Scene
from supermimic.tasks.env_cfgs import gencoord_env_cfg


def scene_model():
  """Compile the actual G1 tracking training scene with a single robot."""
  cfg = gencoord_env_cfg()
  cfg.scene.num_envs = 1
  return Scene(cfg.scene, device="cpu").compile()


def main():
  p = argparse.ArgumentParser()
  p.add_argument("motion", type=Path)
  p.add_argument("--speed", type=float, default=1.0)
  p.add_argument("--start", type=float, default=0.0, help="start time in seconds")
  args = p.parse_args()
  if args.speed <= 0 or args.start < 0:
    p.error("speed must be positive and start must be nonnegative")
  with np.load(args.motion, allow_pickle=False) as f:
    if "qpos" in f:
      qpos = f["qpos"].copy()
      data = {"joint_pos": qpos[:, 7:], "root_pos_w": qpos[:, :3],
              "root_quat_w": qpos[:, 3:7], "joint_names": f["joint_names"].copy(),
              "fps": f["fps"].copy()}
    else:
      data = {name: f[name].copy() for name in ("joint_pos", "joint_names", "root_pos_w", "root_quat_w", "fps")}
  fps = float(np.asarray(data["fps"]).reshape(-1)[0])
  model = scene_model()
  state = mujoco.MjData(model)
  ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + name)
    for name in data["joint_names"].tolist()
  ]
  if any(i < 0 for i in ids):
    raise ValueError("unknown joint names")
  start = int(args.start * fps)
  if start >= len(data["joint_pos"]):
    raise ValueError("start is beyond the motion")
  print(f"Playing {len(data['joint_pos'])} frames ({len(data['joint_pos']) / fps:.2f}s), looping at {args.speed:g}x", flush=True)
  with mujoco.viewer.launch_passive(model, state) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = model.body("robot/pelvis").id
    viewer.cam.distance = 3.0
    begin = time.monotonic()
    while viewer.is_running():
      frame = start + int((time.monotonic() - begin) * fps * args.speed) % (
        len(data["joint_pos"]) - start
      )
      with viewer.lock():
        state.qpos[:3] = data["root_pos_w"][frame]
        state.qpos[3:7] = data["root_quat_w"][frame]
        state.qpos[model.jnt_qposadr[ids]] = data["joint_pos"][frame]
        mujoco.mj_forward(model, state)
      viewer.sync()
      time.sleep(0.005)


if __name__ == "__main__":
  main()
