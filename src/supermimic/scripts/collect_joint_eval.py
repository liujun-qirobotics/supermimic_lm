"""Collect a fresh, frozen-policy evaluation dataset outside training Replay."""

import argparse
import hashlib
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from supermimic.learners.joint_policy_learner import (
  JointPolicyCfg,
  sensor_from_real_obs,
)
from supermimic.learners.wm_learner import WorldModelLearnerCfg
from supermimic.runners.joint_collection import JointReplayCollector
from supermimic.runners.joint_eval_dataset import static_model_signature
from supermimic.runners.joint_runner import JointRunner, JointRunnerCfg
from supermimic.tasks.joint_env import joint_g1_env_cfg


def collect(runner, steps, seed, action_noise=0.0):
  if steps < 2 or action_noise < 0:
    raise ValueError("steps must be at least 2 and action_noise nonnegative")
  env = runner.env
  runner.obs, _ = env.reset(seed=seed)
  sensor = sensor_from_real_obs(runner.obs["actor"])
  runner.history = sensor.unsqueeze(1).repeat(1, runner.cfg.history_length, 1)
  collector = JointReplayCollector(env, steps)
  runner.actor.eval()
  with torch.no_grad():
    for _ in range(steps):
      action = runner.actor(runner.actor_inputs())
      if action_noise:
        action = action + torch.randn_like(action) * action_noise
      runner.obs, _, terminated, truncated, _ = collector.step(action, runner.obs["actor"])
      sensor = sensor_from_real_obs(runner.obs["actor"])
      shifted = torch.cat((runner.history[:, 1:], sensor.unsqueeze(1)), dim=1)
      runner.history = torch.where((terminated | truncated).view(-1, 1, 1),
                                   sensor.unsqueeze(1), shifted)
  buffer = collector.buffer
  return {
    "kind": "supermimic_joint_eval_v1",
    "seed": seed,
    "steps": steps,
    "num_envs": env.num_envs,
    "action_noise": action_noise,
    "step_dt": env.step_dt,
    "joint_names": list(env.scene["robot"].joint_names),
    "static_model_sha256": static_model_signature(runner),
    "replay": {k: v.detach().cpu().clone() for k, v in buffer._storage.items()},
    "replay_step": buffer._step,
    "replay_count": buffer.num_transitions,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", help="Trusted checkpoint supplying the frozen Actor")
  parser.add_argument("--output", required=True)
  parser.add_argument("--steps", type=int, default=256)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--seed", type=int, default=20261113)
  parser.add_argument("--action-noise", type=float, default=0.0)
  parser.add_argument("--device", default=None)
  args = parser.parse_args()
  if args.steps < 2 or args.num_envs < 1 or args.action_noise < 0:
    parser.error("steps >= 2, num-envs >= 1 and action-noise >= 0 are required")
  target = Path(args.output)
  if target.exists():
    raise FileExistsError(f"Evaluation dataset already exists: {target}")
  data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  if data.get("version") not in (4, 5):
    raise ValueError("Expected a version-4 or version-5 checkpoint")
  config = dict(data["config"])
  config["wm"] = WorldModelLearnerCfg(**config["wm"])
  config["policy"] = JointPolicyCfg(**config["policy"])
  cfg = JointRunnerCfg(**config)
  env_cfg = joint_g1_env_cfg()
  env_cfg.scene.num_envs, env_cfg.seed = args.num_envs, args.seed
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env = ManagerBasedRlEnv(env_cfg, device=device)
  try:
    runner = JointRunner(env, cfg)
    runner.actor.load_state_dict(data["actor"])
    result = collect(runner, args.steps, args.seed, args.action_noise)
    result["policy_checkpoint_sha256"] = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
      torch.save(result, stream)
    print(f"Saved independent evaluation dataset: {target}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
