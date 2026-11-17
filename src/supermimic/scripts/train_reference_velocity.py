"""Train original SuperMimic motion anchor, then motion-supervised velocity encoder."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg
from supermimic.learners.wm_learner import WorldModelLearnerCfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import create_env, setup_device
from supermimic.tasks.env_cfgs import gencoord_env_cfg


def save_checkpoint(runner, path, stage, iteration, updates):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temp = path.with_suffix(path.suffix + ".tmp")
  try:
    torch.save({"kind": "reference_velocity_v1", "stage": stage, "iteration": iteration,
                "updates": updates,
                "config": asdict(runner.cfg), "num_goals": runner._num_goals,
                "world": runner._wm.bundle(),
                "actor": runner.policy_learner.actor.state_dict(),
                "critic": runner.policy_learner.critic.state_dict()}, temp)
    temp.replace(path)
  finally:
    temp.unlink(missing_ok=True)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--motion", required=True, help="Research-format NPZ path, glob, or packed library")
  p.add_argument("--stage", choices=("motion", "velocity"), required=True)
  p.add_argument("--init", default=None, help="Trusted motion-stage checkpoint for velocity stage")
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--profile", choices=("smoke", "train"), default="train")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--iterations", type=int, default=1000)
  p.add_argument("--collect-steps", type=int, default=16)
  p.add_argument("--save-every", type=int, default=100)
  p.add_argument("--device", default=None)
  args = p.parse_args()
  if min(args.num_envs, args.iterations, args.collect_steps) < 1 or args.save_every < 0:
    p.error("num-envs, iterations and collect-steps must be positive; save-every nonnegative")
  if args.stage == "velocity" and not args.init:
    p.error("velocity stage requires --init with a trained motion-stage checkpoint")
  if args.stage == "motion" and args.init:
    p.error("--init is intended for motion-to-velocity stage transfer")
  device = setup_device(args.device)
  env_cfg = gencoord_env_cfg()
  env_cfg.commands["motion"].motion_file = args.motion
  env_cfg.commands["motion"].sampling_mode = "uniform"
  if args.profile == "smoke":
    cfg = SuperMimicRunnerCfg(
      sampling_mode="uniform", buffer_size=32, actor_obs="proprio", history_length=10,
      dynamics_hidden_dims=(32,), actor_hidden_dims=(32,), critic_hidden_dims=(32,),
      wm=WorldModelLearnerCfg(batch_size=2, horizon=8, num_batches=1),
      policy=ActorCriticPolicyLearnerCfg(batch_size=2, horizon=32, num_batches=1),
    )
  else:
    cfg = SuperMimicRunnerCfg(
      sampling_mode="uniform", actor_obs="proprio", history_length=10,
      dynamics_hidden_dims=(512, 512, 512), actor_hidden_dims=(512, 256, 128),
      wm=WorldModelLearnerCfg(batch_size=512, horizon=8, num_batches=2),
      policy=ActorCriticPolicyLearnerCfg(batch_size=256, horizon=32, num_batches=1),
    )
  cfg.slot_mix = {"motion": 1.0, "velocity": 0.0} if args.stage == "motion" else {"motion": 0.5, "velocity": 0.5}
  cfg.max_iterations = args.iterations
  env = create_env(env_cfg, args.num_envs, device)
  try:
    runner = SuperMimicRunner(env, cfg, task_id="SuperMimic-GenCoord-G1", device=device)
    if args.init:
      saved = torch.load(args.init, map_location=device, weights_only=False)
      if saved.get("kind") != "reference_velocity_v1" or saved.get("stage") != "motion":
        raise ValueError("--init must be a motion-stage reference_velocity_v1 checkpoint")
      if saved.get("updates", 0) < 1:
        raise ValueError("motion checkpoint must contain at least one training update")
      previous = saved["config"]
      for name in ("actor_obs", "history_length", "latent_dim", "actor_hidden_dims", "critic_hidden_dims", "dynamics_hidden_dims", "representation"):
        if previous[name] != asdict(cfg)[name]:
          raise ValueError(f"motion checkpoint configuration mismatch: {name}")
      if saved["num_goals"] != runner._num_goals:
        raise ValueError("motion checkpoint goal dimension mismatch")
      runner._wm.load_bundle(saved["world"])
      runner.policy_learner.actor.load_state_dict(saved["actor"])
      runner.policy_learner.critic.load_state_dict(saved["critic"])
    learner = runner.world_model_learner
    if not device.startswith("cuda"):
      learner._compiled_rollout = learner._rollout
      learner._compiled_compute_losses = learner._compute_losses
      learner._compiled_represent = learner.representation.represent
    env.reset()
    runner._seed_sim_history()
    updates = 0
    for iteration in range(1, args.iterations + 1):
      for _ in range(args.collect_steps):
        runner.step()
      if runner.replay_buffer.num_transitions >= cfg.wm.horizon:
        try:
          batches = [runner.replay_buffer.sample(cfg.wm.horizon, cfg.wm.batch_size)
                     for _ in range(cfg.wm.num_batches)]
        except RuntimeError as exc:
          if str(exc).startswith("no valid sequences"):
            print(f"iteration={iteration} waiting for continuous reference transitions", flush=True)
            continue
          raise
        wm = learner.update(batches)
        policy = runner.policy_learner.update()
        updates += 1
        logs = {k: float(v) for k, v in (wm | policy).items()}
        print(f"iteration={iteration} metrics={logs}", flush=True)
      if updates and args.save_every and iteration % args.save_every == 0:
        save_checkpoint(runner, args.checkpoint, args.stage, iteration, updates)
    if not updates:
      raise RuntimeError("No model or policy update completed; gather longer valid reference episodes")
    save_checkpoint(runner, args.checkpoint, args.stage, args.iterations, updates)
    print(f"Saved checkpoint: {args.checkpoint}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
