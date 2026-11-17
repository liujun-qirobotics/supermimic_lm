"""Integration check of the reference-free task against the real simulator."""

import mujoco
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.envs import ManagerBasedRlEnv
from tensordict import TensorDict

from supermimic.envs.joint_imagination import joint_imagination_step
from supermimic.joint_actor import JointCommandActor
from supermimic.learners.wm_learner import WorldModelLearnerCfg
from supermimic.runners.joint_collection import JointReplayCollector, full_state
from supermimic.runners.joint_world_model import JointWorldModel
from supermimic.tasks.joint_command import interpolate_reference_joints
from supermimic.tasks.joint_env import joint_g1_env_cfg
from supermimic.utils.differentiable_fk import DifferentiableFK


def test_joint_environment_reset_and_step():
  cfg = joint_g1_env_cfg()
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    assert env.action_manager.total_action_dim == 29
    assert obs["actor"].shape == (2, 127)
    assert obs["critic"].shape == (2, 139)
    targets = env.command_manager.get_command("hands").reshape(2, 2, 7)
    assert torch.allclose(targets[..., 3:].norm(dim=-1), torch.ones(2, 2), atol=1e-5)
    hands = env.command_manager.get_term("hands")
    hands.foot_anchor[1] = 123.0
    hands.was_standing[1] = True
    hands._update_command(torch.tensor([0], device=env.device))
    assert torch.all(hands.foot_anchor[1] == 123.0)
    assert hands.was_standing[1]
    collector = JointReplayCollector(env, buffer_size=8)
    first_state = full_state(env).clone()
    first_obs = obs["actor"].clone()
    start = hands.transition_start.clone()
    target = hands.transition_target.clone()
    elapsed = (hands.transition_time_left_at_start - hands.time_left).clone()
    obs, reward, terminated, truncated, _ = collector.step(
      torch.zeros(2, 29), first_obs
    )
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(reward).all()
    assert terminated.shape == truncated.shape == (2,)
    assert not (terminated | truncated).any()
    expected_q = interpolate_reference_joints(
      start, target, (elapsed + env.step_dt) / hands.cfg.transition_duration
    )
    root = torch.zeros(2, 3)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    p, q = hands.fk(root, quat, expected_q)
    expected = torch.cat((p, q), dim=-1).flatten(start_dim=1)
    torch.testing.assert_close(hands.command, expected, atol=1e-6, rtol=1e-6)
    stored = collector.buffer._storage
    torch.testing.assert_close(stored["hand_transition_start"][:, 0], start)
    torch.testing.assert_close(stored["hand_transition_target"][:, 0], target)
    next_state = full_state(env).clone()
    collector.step(torch.zeros(2, 29), obs["actor"])
    torch.testing.assert_close(stored["state"][:, 1], next_state)
    torch.testing.assert_close(stored["action"][:, 0], torch.zeros(2, 29))
    sample = collector.buffer.sample(sequence_length=2, batch_size=4)
    assert torch.isfinite(sample["state"]).all()
    assert sample["command"].shape == (4, 2, 17)
    assert sample["sim_ctx"].shape[-1] == collector.sim_context.output_dim
    assert torch.equal(collector.buffer._storage["actor_obs"][:, 0], first_obs)
    assert torch.equal(collector.buffer._storage["state"][:, 0], first_state)
    wm = JointWorldModel(
      collector,
      WorldModelLearnerCfg(horizon=2, batch_size=2, num_batches=1),
      hidden_dims=(16,),
    )
    # Exercise the same update eagerly; compilation is a separate runtime concern.
    wm.learner._compiled_rollout = wm.learner._rollout
    wm.learner._compiled_compute_losses = wm.learner._compute_losses
    wm.learner._compiled_represent = wm.representation.represent
    before = next(wm.dynamics.mlp.parameters()).detach().clone()
    stats = wm.update()
    assert torch.isfinite(stats["Loss/state"])
    assert not torch.equal(before, next(wm.dynamics.mlp.parameters()))
    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    fk = DifferentiableFK(
      model,
      "pelvis",
      env.scene["robot"].joint_names,
      [
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
      ],
    )
    state = sample["state"][:, 0]
    rep = wm.representation.represent(state)
    aux = wm.representation.extract_aux_state(state)
    command = sample["command"][:, 0]
    actor = JointCommandActor(93, 1, 29, hidden_dims=(16,))
    actor_obs = TensorDict({"history": torch.randn(4, 1, 93), "command": command}, [4])
    actions = actor(actor_obs)
    _, _, imagined_reward, done, errors = joint_imagination_step(
      wm.dynamics,
      fk,
      aux,
      rep,
      actions,
      sample["sim_ctx"][:, 0],
      command,
      sample["foot_anchor"][:, 0],
      sample["standing"][:, 0],
    )
    hand_grads = torch.autograd.grad(
      errors["left_position"].sum(),
      tuple(actor.parameters()),
      retain_graph=True,
      allow_unused=True,
    )
    assert any(g is not None and g.abs().sum() > 0 for g in hand_grads)
    (-imagined_reward.mean()).backward()
    assert torch.isfinite(done).all()
    assert all(
      torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None
    )
    assert any(
      p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters()
    )
    events = env.event_manager
    index = events._mode_term_names["interval"].index("push_robot")
    events._interval_term_time_left[index][:] = env.step_dt * 0.5
    slot = collector.buffer._step
    collector.step(torch.zeros(2, 29), obs["actor"])
    assert collector.buffer._storage["wm_invalid"][:, slot].all()
  finally:
    env.close()
