"""Experimental joint-command world-model training without motion files."""

import argparse
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from supermimic.learners.joint_policy_learner import JointPolicyCfg
from supermimic.learners.wm_learner import WorldModelLearnerCfg
from supermimic.runners.joint_runner import JointRunner, JointRunnerCfg
from supermimic.tasks.joint_env import joint_g1_env_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--profile", choices=("smoke", "probe", "dev"), default="smoke")
  parser.add_argument("--iterations", type=int, default=3)
  parser.add_argument("--num-envs", type=int, default=2)
  parser.add_argument("--history-length", type=int, choices=(10,), default=10)
  parser.add_argument("--device", default=None)
  parser.add_argument("--checkpoint", default="logs/joint/checkpoint.pt")
  parser.add_argument("--save-every", type=int, default=0,
                      help="Save the checkpoint every N iterations; 0 disables periodic saving")
  parser.add_argument("--resume", default=None)
  parser.add_argument("--allow-horizon-change", action="store_true",
                      help="Allow only world-model and policy horizon differences when resuming")
  parser.add_argument("--allow-slip-proxy-change", action="store_true",
                      help="Allow slip proxy parameter changes when resuming")
  parser.add_argument("--allow-capsule-weight-change", action="store_true",
                      help="Allow the capsule collision weight to change when resuming")
  parser.add_argument("--eager", action="store_true")
  parser.add_argument("--clearance-weight", type=float, default=2.0)
  parser.add_argument("--landing-weight", type=float, default=0.0)
  parser.add_argument("--collision-weight", type=float, default=0.0)
  parser.add_argument("--capsule-collision-weight", type=float, default=0.0)
  parser.add_argument("--slip-contact-height", type=float, default=0.025)
  parser.add_argument("--slip-contact-width", type=float, default=0.01)
  args = parser.parse_args()
  if args.iterations < 1 or args.num_envs < 1 or args.history_length < 1:
    parser.error("iterations, num-envs and history-length must be positive")
  if min(args.clearance_weight, args.landing_weight, args.collision_weight, args.capsule_collision_weight) < 0:
    parser.error("proxy weights must be nonnegative")
  if args.save_every < 0:
    parser.error("save-every must be nonnegative")
  if not (0.0 <= args.slip_contact_height <= 0.1 and 0.0001 <= args.slip_contact_width <= 0.1):
    parser.error("slip height must be in [0, 0.1] and width in [0.0001, 0.1] metres")
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  if args.profile == "smoke":
    cfg = JointRunnerCfg(
      buffer_size=16,
      history_length=args.history_length,
      collect_steps=4,
      warmup_iterations=1,
      actor_hidden_dims=(32,),
      dynamics_hidden_dims=(32,),
      wm=WorldModelLearnerCfg(horizon=8, batch_size=4, num_batches=1),
      policy=JointPolicyCfg(horizon=32, batch_size=4),
      eager=True,
    )
  elif args.profile == "probe":
    cfg = JointRunnerCfg(
      buffer_size=256,
      history_length=args.history_length,
      collect_steps=16,
      warmup_iterations=10,
      actor_hidden_dims=(128, 128),
      dynamics_hidden_dims=(128, 128),
      wm=WorldModelLearnerCfg(horizon=8, batch_size=128, num_batches=2),
      policy=JointPolicyCfg(horizon=32, batch_size=32),
      eager=True,
    )
  else:
    cfg = JointRunnerCfg(eager=args.eager, history_length=args.history_length)
  cfg.policy.clearance_weight = args.clearance_weight
  cfg.policy.landing_weight = args.landing_weight
  cfg.policy.collision_weight = args.collision_weight
  cfg.policy.capsule_collision_weight = args.capsule_collision_weight
  cfg.policy.slip_contact_height = args.slip_contact_height
  cfg.policy.slip_contact_width = args.slip_contact_width
  env_cfg = joint_g1_env_cfg()
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    runner = JointRunner(env, cfg)
    if args.resume:
      runner.load(args.resume, allow_horizon_change=args.allow_horizon_change,
                  allow_slip_proxy_change=args.allow_slip_proxy_change,
                  allow_capsule_weight_change=args.allow_capsule_weight_change)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(args.iterations):
      metrics = runner.step()
      print(f"iteration={runner.iteration} metrics={metrics}", flush=True)
      if args.save_every and runner.iteration % args.save_every == 0:
        runner.save(checkpoint)
        print(f"Saved checkpoint: {checkpoint}", flush=True)
    runner.save(checkpoint)
    print(f"Saved checkpoint: {checkpoint}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
