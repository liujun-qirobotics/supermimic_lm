"""Run real collection, BPTT, critic and checkpoint recovery on CPU."""

import math

import mujoco
import numpy as np
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply

from supermimic.envs.joint_imagination import (
  ankle_collision_proxy,
  body_collision_proxy,
  foot_clearance_proxy,
  foot_slip_proxy,
  landing_speed_proxy,
)
from supermimic.learners.joint_policy_learner import (
  JointPolicyCfg,
  sensor_from_real_obs,
)
from supermimic.learners.wm_learner import WorldModelLearnerCfg
from supermimic.runners.joint_diagnostics import (
  evaluate_joint_replay,
  inspect_collision_geometry,
  slip_contact_breakdown,
)
from supermimic.runners.joint_runner import JointRunner, JointRunnerCfg
from supermimic.scripts.collect_joint_eval import collect
from supermimic.tasks.joint_env import joint_g1_env_cfg
from supermimic.utils.differentiable_fk import DifferentiableFKWithVelocity


def test_joint_training_and_resume(tmp_path):
  env_cfg = joint_g1_env_cfg()
  env_cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(env_cfg, device="cpu")
  cfg = JointRunnerCfg(
    buffer_size=16,
    collect_steps=2,
    warmup_iterations=0,
    actor_hidden_dims=(16,),
    dynamics_hidden_dims=(16,),
    eager=True,
    wm=WorldModelLearnerCfg(horizon=2, batch_size=2, num_batches=1),
    policy=JointPolicyCfg(horizon=2, batch_size=2, capsule_collision_weight=1.0),
  )
  try:
    runner = JointRunner(env, cfg)
    robot = env.scene["robot"]
    data = robot.data
    model = env.sim.mj_model
    mjdata = mujoco.MjData(model)
    root_joint = int(model.body_jntadr[int(robot.indexing.root_body_id)])
    root_qpos = int(model.jnt_qposadr[root_joint])
    joint_qpos = [int(model.jnt_qposadr[int(i)]) for i in robot.indexing.joint_ids]
    q = data.joint_pos[0].detach().clone().requires_grad_()
    root_pos = data.root_link_pos_w[0].detach()
    root_quat = data.root_link_quat_w[0].detach()
    mjdata.qpos[root_qpos:root_qpos+7] = torch.cat((root_pos, root_quat)).numpy()
    mjdata.qpos[joint_qpos] = q.detach().numpy()
    mujoco.mj_forward(model, mjdata)
    actual_dist = runner.capsule_collision(root_pos, root_quat, q)
    geom_ids = {name: int(robot.indexing.geom_ids[robot.geom_names.index(name)])
                for name in runner.capsule_collision.geom_names}
    expected_dist = [
      mujoco.mj_geomDistance(model, mjdata, geom_ids[a], geom_ids[b], 10.0, np.zeros(6))
      for a, b in runner.capsule_collision.pair_names
    ]
    np.testing.assert_allclose(actual_dist.detach().numpy(), expected_dist, atol=2e-5)
    actual_dist.sum().backward()
    assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
    generator = torch.Generator().manual_seed(7)
    for _ in range(4):
      q_sample = q.detach() + 0.3 * torch.randn(q.shape, generator=generator)
      mjdata.qpos[joint_qpos] = q_sample.numpy()
      mujoco.mj_forward(model, mjdata)
      expected = [
        mujoco.mj_geomDistance(model, mjdata, geom_ids[a], geom_ids[b], 10.0, np.zeros(6))
        for a, b in runner.capsule_collision.pair_names
      ]
      distances = runner.capsule_collision(root_pos, root_quat, q_sample)
      np.testing.assert_allclose(distances.detach().numpy(), expected, atol=2e-5)
    pos, quat, vel, omega = DifferentiableFKWithVelocity(runner.fk)(
      data.root_link_pos_w,
      data.root_link_quat_w,
      data.joint_pos,
      data.root_link_lin_vel_w,
      data.root_link_ang_vel_w,
      data.joint_vel,
    )
    offset = quat_apply(
      quat[:, 2:4], runner.policy.foot_site_offsets.expand(2, -1, -1)
    )
    torch.testing.assert_close(
      pos[:, 2:4] + offset, data.site_pos_w[:, runner.foot_site_ids], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
      vel[:, 2:4] + torch.cross(omega[:, 2:4], offset, dim=-1),
      data.site_lin_vel_w[:, runner.foot_site_ids], atol=1e-4, rtol=1e-4
    )
    assert runner.history.shape == (2, 10, 93)
    assert torch.equal(runner.history[:, 0], runner.history[:, -1])
    before = next(runner.actor.network.parameters()).detach().clone()
    metrics = runner.step()
    assert torch.equal(runner.history[:, -1], sensor_from_real_obs(runner.obs["actor"]))
    assert "left_hand_error_m" in metrics and "velocity_xy_error_mps" in metrics
    assert "foot_contact_proxy_disagreement" in metrics
    assert all(math.isfinite(x) for x in metrics.values())
    assert "critic_loss" in metrics and metrics["actor_grad"] > 0
    assert not torch.equal(before, next(runner.actor.network.parameters()))
    path = tmp_path / "checkpoint.pt"
    runner.save(path)
    saved = next(runner.actor.network.parameters()).detach().clone()
    cfg.policy.horizon = 3
    cfg.wm.horizon = 3
    with pytest.raises(ValueError, match="configuration"):
      runner.load(path)
    cfg.action_noise = 0.2
    with pytest.raises(ValueError, match="configuration"):
      runner.load(path, allow_horizon_change=True)
    cfg.action_noise = 0.1
    cfg.policy.capsule_collision_weight = 2.0
    with pytest.raises(ValueError, match="configuration"):
      runner.load(path, allow_horizon_change=True)
    runner.load(path, allow_horizon_change=True, allow_capsule_weight_change=True)
    cfg.policy.capsule_collision_weight = 1.0
    runner.load(path, allow_horizon_change=True)
    # The restored replay is truncated and only two new rows are collected below.
    cfg.wm.horizon = 2
    assert torch.equal(saved, next(runner.actor.network.parameters()))
    last = (runner.collector.buffer._step - 1) % cfg.buffer_size
    assert runner.collector.buffer._storage["truncated"][:, last].all()
    metrics = runner.step()
    assert runner.iteration == 2 and all(math.isfinite(x) for x in metrics.values())
    report = evaluate_joint_replay(runner, horizon=1, batch_size=4)
    assert report["same_state"]["samples"] == 8
    assert report["same_state"]["available_unique_states"] == 8
    assert report["slip_candidate_comparison"]["available"]
    assert "contact_changed_within_window" in report["open_loop_by_scenario"]
    collision_report = inspect_collision_geometry(runner, sample_per_class=2, seed=1)
    assert "positive_history" in collision_report
    assert report["open_loop"][0]["step"] == 1
    assert math.isfinite(report["open_loop"][0]["ankle_position_error_m"])
    assert report["one_step_termination"]["valid_samples"] <= 4
    runner.save(path)
    legacy = torch.load(path, weights_only=False)
    legacy["version"] = 4
    legacy["replay"].pop("wm_invalid")
    legacy["config"]["policy"].pop("slip_contact_height")
    legacy["config"]["policy"].pop("slip_contact_width")
    torch.save(legacy, path)
    runner.load(path)
    assert runner.collector.buffer._storage["wm_invalid"].all()
    before_replay = runner.collector.buffer._storage["state"].clone()
    before_params = next(runner.actor.network.parameters()).detach().clone()
    evaluation = collect(runner, steps=4, seed=23)
    assert evaluation["kind"] == "supermimic_joint_eval_v1"
    assert evaluation["replay"]["state"].shape == (2, 4, 71)
    assert torch.equal(before_replay, runner.collector.buffer._storage["state"])
    assert torch.equal(before_params, next(runner.actor.network.parameters()))
  finally:
    env.close()


def test_slip_proxy_gating_and_gradient():
  positions = torch.tensor([[[0., 0., 0.0], [0., 0., 0.2]]], requires_grad=True)
  velocities = torch.tensor([[[1., 0., 0.], [1., 0., 0.]]], requires_grad=True)
  moving = torch.tensor([[0.2, 0., 0.]])
  cost = foot_slip_proxy(positions, velocities, moving)
  assert 0.9 < cost.item() < 1.0
  cost.backward()
  assert velocities.grad[0, 0, 0] > 1.8
  assert velocities.grad[0, 1, 0] < 1e-5
  assert foot_slip_proxy(positions.detach(), velocities.detach(), torch.zeros_like(moving)).item() == 0
  groups = slip_contact_breakdown(
    positions.detach(), velocities.detach(), torch.tensor([[1., 0.]]), moving
  )
  assert groups["contact"]["foot_samples"] == 1
  assert groups["no_contact"]["foot_samples"] == 1
  total = sum(group["proxy_slip_per_environment"] for group in groups.values())
  torch.testing.assert_close(torch.tensor(total), foot_slip_proxy(
    positions.detach(), velocities.detach(), moving
  ).squeeze())


def test_clearance_landing_collision_proxies():
  pos = torch.tensor([[[0., 0., 0.], [0., 0., 0.1]]], requires_grad=True)
  vel = torch.tensor([[[1., 0., -1.], [1., 0., 0.]]], requires_grad=True)
  moving = torch.tensor([[0.2, 0., 0.]])
  assert torch.allclose(foot_clearance_proxy(pos, vel, moving), torch.tensor([0.1]))
  assert landing_speed_proxy(pos, vel, moving).item() > 0.9
  assert landing_speed_proxy(pos, vel, torch.zeros_like(moving)).item() == 0
  ankles = torch.tensor([[[0., 0., 0.], [0., 0.05, 0.]]], requires_grad=True)
  cost = ankle_collision_proxy(ankles)
  torch.testing.assert_close(cost, torch.tensor([0.0025]))
  cost.backward()
  assert ankles.grad[0, 0, 1] > 0
  body = torch.tensor([[[0., 0., 0.12], [1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 0.]]])
  quat = torch.tensor([[[1., 0., 0., 0.]]]).expand(1, 5, 4)
  torch.testing.assert_close(body_collision_proxy(body, quat), torch.tensor([0.0144]))
