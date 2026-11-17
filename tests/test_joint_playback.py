import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer.native import NativeMujocoViewer

from supermimic.joint_actor import JointCommandActor
from supermimic.runners.joint_playback import (
  FixedJointPolicy,
  evaluate_fixed,
  fixed_env_cfg,
)


def _checkpoint():
  actor = JointCommandActor(93, 10, 29, hidden_dims=(16,))
  return {"version": 5, "config": {"history_length": 10, "actor_hidden_dims": (16,)},
          "actor": actor.state_dict()}


def test_fixed_command_history_and_metrics():
  command = (0.3, 0.0, 0.0)
  env = ManagerBasedRlEnv(fixed_env_cfg(command, 2, 1.0, 7), device="cpu")
  try:
    obs, _ = env.reset()
    policy = FixedJointPolicy(env, _checkpoint(), command)
    action = policy(obs)
    assert action.shape == (2, 29)
    torch.testing.assert_close(env.command_manager.get_command("twist"), torch.tensor(command).expand(2, -1))
    torch.testing.assert_close(env.command_manager.get_command("hands"), policy.hand_target)
    assert torch.allclose(policy.history[:, 0], policy.history[:, -1])
    report = evaluate_fixed(env, policy, 0.12)
    assert report["trials"] == 2 and report["valid_post_step_samples"] > 0
    assert report["velocity_xy_rmse_mps"] is not None
    assert all(torch.isfinite(torch.tensor(v)) for v in report.values() if isinstance(v, float))
  finally:
    env.close()


def test_terminal_state_is_excluded_even_when_environment_auto_resets():
  cfg = fixed_env_cfg((0., 0., 0.), 2, 1.0, 8)
  cfg.terminations["forced"] = TerminationTermCfg(
    func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  )
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    policy = FixedJointPolicy(env, _checkpoint(), (0., 0., 0.))
    report = evaluate_fixed(env, policy, 0.12)
    assert report["terminated_trials"] == 2
    assert report["valid_post_step_samples"] == 0
    assert report["velocity_xy_rmse_mps"] is None
    assert report["survived_fraction"] == 0.0
  finally:
    env.close()


def test_native_viewer_step_without_opening_a_window():
  command = (0.3, 0.0, 0.0)
  env = ManagerBasedRlEnv(fixed_env_cfg(command, 1, 1.0, 9), device="cpu")
  try:
    wrapped = RslRlVecEnvWrapper(env)
    policy = FixedJointPolicy(env, _checkpoint(), command)
    viewer = NativeMujocoViewer(wrapped, policy)
    assert viewer._execute_step()
    assert viewer._execute_step()
    assert policy.history.shape == (1, 10, 93)
  finally:
    env.close()
