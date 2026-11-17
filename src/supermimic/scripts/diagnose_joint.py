"""Evaluate a joint-training checkpoint's replay without changing the checkpoint."""

import argparse
import json

import torch
from mjlab.envs import ManagerBasedRlEnv

from supermimic.learners.joint_policy_learner import JointPolicyCfg
from supermimic.learners.wm_learner import WorldModelLearnerCfg
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.runners.joint_diagnostics import (
  evaluate_joint_replay,
  inspect_collision_geometry,
)
from supermimic.runners.joint_eval_dataset import static_model_signature
from supermimic.runners.joint_runner import JointRunner, JointRunnerCfg
from supermimic.tasks.joint_env import joint_g1_env_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", help="Trusted version-4 or version-5 joint-training checkpoint")
  parser.add_argument("--horizon", type=int, default=8)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--same-state-limit", type=int, default=32768)
  parser.add_argument("--replay-checkpoint", default=None,
                      help="Use fixed replay from this compatible checkpoint for comparison")
  parser.add_argument("--eval-set", default=None,
                      help="Use a separately collected fixed evaluation dataset")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--collision-samples", type=int, default=0,
                      help="Reconstruct this many states per self-collision class on CPU")
  parser.add_argument("--device", default=None)
  args = parser.parse_args()
  if args.eval_set and args.replay_checkpoint:
    parser.error("choose either --eval-set or --replay-checkpoint")
  data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  if data.get("version") not in (4, 5):
    raise ValueError("Expected a version-4 or version-5 checkpoint")
  config = dict(data["config"])
  config["wm"] = WorldModelLearnerCfg(**config["wm"])
  config["policy"] = JointPolicyCfg(**config["policy"])
  runner_cfg = JointRunnerCfg(**config)
  env_cfg = joint_g1_env_cfg()
  env_cfg.scene.num_envs = data["replay"]["state"].shape[0]
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env = ManagerBasedRlEnv(env_cfg, device=device)
  try:
    runner = JointRunner(env, runner_cfg)
    runner.load(args.checkpoint)
    if args.eval_set:
      evaluation = torch.load(args.eval_set, map_location=device, weights_only=False)
      if evaluation.get("kind") != "supermimic_joint_eval_v1":
        raise ValueError("Invalid evaluation dataset format")
      if evaluation["joint_names"] != list(env.scene["robot"].joint_names):
        raise ValueError("Evaluation dataset joint order does not match")
      if abs(evaluation["step_dt"] - env.step_dt) > 1e-9:
        raise ValueError("Evaluation dataset control timestep does not match")
      if evaluation.get("static_model_sha256") != static_model_signature(runner):
        raise ValueError("Evaluation dataset static model/action signature does not match")
      current = runner.collector.buffer
      stored = evaluation["replay"]
      if set(stored) != set(current._storage.keys()):
        raise ValueError("Evaluation dataset replay schema does not match")
      eval_envs, eval_steps = stored["state"].shape[:2]
      schema = {}
      for name, old in current._storage.items():
        if stored[name].shape[2:] != old.shape[2:] or stored[name].dtype != old.dtype:
          raise ValueError(f"Evaluation dataset field mismatch: {name}")
        schema[name] = FieldSpec(old.shape[2] if old.ndim == 3 else 0, old.dtype)
      fixed = ReplayBuffer(schema, eval_steps, eval_envs, device)
      for name, value in stored.items():
        fixed._storage[name].copy_(value)
      fixed._step, fixed.num_transitions = evaluation["replay_step"], evaluation["replay_count"]
      if not (0 <= fixed._step < eval_steps and 0 < fixed.num_transitions <= eval_steps):
        raise ValueError("Invalid evaluation dataset replay counters")
      runner.collector.buffer = fixed
    if args.replay_checkpoint:
      reference = torch.load(args.replay_checkpoint, map_location=device, weights_only=False)
      buffer = runner.collector.buffer
      legacy = reference.get("version") == 4
      saved_fields = set(reference["replay"])
      expected_fields = set(buffer._storage.keys())
      if reference.get("version") not in (4, 5) or saved_fields != (expected_fields - {"wm_invalid"} if legacy else expected_fields):
        raise ValueError("Reference replay version or schema does not match")
      for name, value in reference["replay"].items():
        if value.shape != buffer._storage[name].shape:
          raise ValueError(f"Reference replay shape mismatch: {name}")
        buffer._storage[name].copy_(value)
      if legacy:
        buffer._storage["wm_invalid"].fill_(True)
      buffer._step = reference["replay_step"]
      buffer.num_transitions = reference["replay_count"]
    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
      torch.cuda.manual_seed_all(args.seed)
    report = evaluate_joint_replay(runner, args.horizon, args.batch_size, args.same_state_limit,
                                   strict_disturbances=bool(args.eval_set))
    if args.eval_set:
      report["evaluation_dataset"] = {
        "seed": evaluation["seed"], "steps": evaluation["steps"],
        "num_envs": evaluation["num_envs"],
        "policy_checkpoint_sha256": evaluation["policy_checkpoint_sha256"],
      }
    if args.collision_samples:
      report["collision_geometry"] = inspect_collision_geometry(
        runner, args.collision_samples, args.seed
      )
    print(json.dumps(report, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
