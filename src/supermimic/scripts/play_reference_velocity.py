"""Play a reference-trained velocity Encoder with a preset or Linux joystick."""

import argparse

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from supermimic.runners.linux_joystick import LinuxJoystick
from supermimic.runners.reference_velocity_playback import (
  SCENARIOS,
  ReferenceVelocityPolicy,
  velocity_env_cfg,
)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("checkpoint", help="Trusted velocity-stage checkpoint")
  p.add_argument("--scenario", choices=tuple(SCENARIOS), default="stand")
  p.add_argument("--viewer", choices=("native", "viser"), default="native")
  p.add_argument("--joystick", default=None, help="Linux device, for example /dev/input/js0")
  p.add_argument("--forward-axis", type=int, default=1)
  p.add_argument("--lateral-axis", type=int, default=0)
  p.add_argument("--yaw-axis", type=int, default=3)
  p.add_argument("--deadzone", type=float, default=0.12)
  p.add_argument("--device", default=None)
  p.add_argument("--seed", type=int, default=42)
  args = p.parse_args()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  command = SCENARIOS[args.scenario]
  env = ManagerBasedRlEnv(velocity_env_cfg(command, 1, 3600, args.seed), device=device)
  joystick = None
  try:
    wrapped = RslRlVecEnvWrapper(env)
    if args.joystick:
      joystick = LinuxJoystick(args.joystick, args.forward_axis, args.lateral_axis,
                               args.yaw_axis, args.deadzone)
    policy = ReferenceVelocityPolicy(env, checkpoint, command, joystick)
    if args.viewer == "native":
      from mjlab.viewer.native import NativeMujocoViewer
      viewer = NativeMujocoViewer(wrapped, policy)
    else:
      from mjlab.viewer.viser.viewer import ViserPlayViewer
      viewer = ViserPlayViewer(wrapped, policy)
    print(f"Command: {command}; joystick: {args.joystick or 'disabled'}", flush=True)
    viewer.run()
  finally:
    if joystick is not None:
      joystick.close()
    env.close()


if __name__ == "__main__":
  main()
