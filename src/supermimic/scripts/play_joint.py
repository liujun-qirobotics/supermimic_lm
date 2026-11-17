"""Play a joint-command checkpoint in MuJoCo or Viser with fixed commands."""

import argparse

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from supermimic.runners.joint_playback import SCENARIOS, FixedJointPolicy, fixed_env_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", help="Trusted joint checkpoint")
  parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="forward")
  parser.add_argument("--viewer", choices=("native", "viser"), default="native")
  parser.add_argument("--device", default=None)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--shoulder", type=float, default=0.0)
  parser.add_argument("--elbow", type=float, default=0.8)
  parser.add_argument("--perturbations", action="store_true")
  args = parser.parse_args()
  command = SCENARIOS[args.scenario]
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  cfg = fixed_env_cfg(command, 1, 3600, args.seed, args.perturbations)
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    wrapped = RslRlVecEnvWrapper(env)
    policy = FixedJointPolicy(env, checkpoint, command, args.shoulder, args.elbow)
    print(f"Fixed command {args.scenario}: {command}; pelvis-frame hand targets: {policy.hand_target[0].tolist()}")
    if args.viewer == "native":
      from mjlab.viewer.native import NativeMujocoViewer
      viewer = NativeMujocoViewer(wrapped, policy)
    else:
      from mjlab.viewer.viser.viewer import ViserPlayViewer
      viewer = ViserPlayViewer(wrapped, policy)
    viewer.run()
  finally:
    env.close()


if __name__ == "__main__":
  main()
