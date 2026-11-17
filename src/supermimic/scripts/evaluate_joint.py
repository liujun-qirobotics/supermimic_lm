"""Evaluate one joint-command checkpoint on fixed locomotion commands."""

import argparse
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from supermimic.runners.joint_playback import (
  SCENARIOS,
  FixedJointPolicy,
  evaluate_fixed,
  fixed_env_cfg,
)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", help="Trusted joint checkpoint")
  parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS))
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--duration", type=float, default=10.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--shoulder", type=float, default=0.0, help="Both shoulder pitch joint angles in radians")
  parser.add_argument("--elbow", type=float, default=0.8, help="Both elbow joint angles in radians")
  parser.add_argument("--perturbations", action="store_true")
  parser.add_argument("--device", default=None)
  parser.add_argument("--output", default=None)
  args = parser.parse_args()
  if args.num_envs < 1 or args.duration <= 0:
    parser.error("num-envs and duration must be positive")
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  report = {
    "checkpoint": str(args.checkpoint), "seed": args.seed,
    "perturbations": args.perturbations, "observation_noise": False,
    "startup_domain_randomization": True,
    "arm_reference_joint_angles_rad": {"shoulder_pitch": args.shoulder, "elbow": args.elbow},
    "metrics_exclude_terminal_post_reset_state": True, "scenarios": {},
  }
  for name in args.scenarios:
    command = SCENARIOS[name]
    cfg = fixed_env_cfg(command, args.num_envs, args.duration, args.seed, args.perturbations)
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
      policy = FixedJointPolicy(env, checkpoint, command, args.shoulder, args.elbow)
      report["scenarios"][name] = evaluate_fixed(env, policy, args.duration)
    finally:
      env.close()
    print(f"Evaluated: {name}", flush=True)
  payload = json.dumps(report, indent=2)
  if args.output:
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n")
    print(f"Saved evaluation: {path}")
  print(payload)


if __name__ == "__main__":
  main()
