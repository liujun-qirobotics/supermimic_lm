"""Evaluate a reference-trained velocity Encoder on fixed velocity commands."""

import argparse
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from supermimic.runners.reference_velocity_playback import (
  SCENARIOS,
  ReferenceVelocityPolicy,
  evaluate_velocity,
  velocity_env_cfg,
)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("checkpoint", help="Trusted velocity-stage checkpoint")
  p.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS))
  p.add_argument("--num-envs", type=int, default=16)
  p.add_argument("--duration", type=float, default=10.0)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--device", default=None)
  p.add_argument("--output", default=None)
  args = p.parse_args()
  if args.num_envs < 1 or args.duration <= 0:
    p.error("num-envs and duration must be positive")
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  report = {"seed": args.seed, "duration_s": args.duration,
            "noise": False, "pushes": False, "startup_domain_randomization": True,
            "scenarios": {}}
  for name in args.scenarios:
    command = SCENARIOS[name]
    env = ManagerBasedRlEnv(velocity_env_cfg(command, args.num_envs, args.duration, args.seed), device=device)
    try:
      policy = ReferenceVelocityPolicy(env, checkpoint, command)
      report["scenarios"][name] = evaluate_velocity(env, policy, args.duration)
    finally:
      env.close()
    print(f"Evaluated: {name}", flush=True)
  result = json.dumps(report, indent=2)
  if args.output:
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result + "\n")
  print(result)


if __name__ == "__main__":
  main()
