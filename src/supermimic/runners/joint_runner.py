"""Experimental reference-free joint-task world-model training runner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from tensordict import TensorDict

from supermimic.envs.joint_imagination import foot_contact_proxy
from supermimic.joint_actor import JointCommandActor
from supermimic.learners.joint_policy_learner import (
  JointPolicyCfg,
  JointPolicyLearner,
  sensor_from_real_obs,
)
from supermimic.learners.wm_learner import WorldModelLearnerCfg
from supermimic.runners.joint_collection import JointReplayCollector
from supermimic.runners.joint_world_model import JointWorldModel
from supermimic.utils.capsule_collision import G1CapsuleCollision
from supermimic.utils.differentiable_fk import DifferentiableFK


@dataclass
class JointRunnerCfg:
  buffer_size: int = 256
  history_length: int = 10
  collect_steps: int = 16
  warmup_iterations: int = 10
  action_noise: float = 0.1
  actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
  dynamics_hidden_dims: tuple[int, ...] = (512, 512, 512)
  wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)
  policy: JointPolicyCfg = field(default_factory=JointPolicyCfg)
  eager: bool = False


class JointRunner:
  def __init__(self, env, cfg: JointRunnerCfg):
    if cfg.history_length != 10 or cfg.history_length > cfg.buffer_size:
      raise ValueError(
        "joint training requires 10 history frames and buffer_size >= 10"
      )
    self.env, self.cfg = env, cfg
    self.obs, _ = env.reset()
    initial_sensor = sensor_from_real_obs(self.obs["actor"])
    self.history = initial_sensor.unsqueeze(1).repeat(1, cfg.history_length, 1)
    self.collector = JointReplayCollector(env, cfg.buffer_size)
    self.wm = JointWorldModel(self.collector, cfg.wm, cfg.dynamics_hidden_dims)
    if cfg.eager:
      self.wm.learner._compiled_rollout = self.wm.learner._rollout
      self.wm.learner._compiled_compute_losses = self.wm.learner._compute_losses
      self.wm.learner._compiled_represent = self.wm.representation.represent
    self.actor = JointCommandActor(
      93, cfg.history_length, env.action_manager.total_action_dim, cfg.actor_hidden_dims
    ).to(env.device)
    robot = env.scene["robot"]
    model = env.sim.mj_model
    body_names = [
      model.body(int(robot.indexing.body_ids[robot.body_names.index(name)])).name
      for name in (
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "torso_link",
      )
    ]
    root_name = model.body(robot.indexing.root_body_id).name
    joint_names = [model.joint(int(i)).name for i in robot.indexing.joint_ids]
    self.fk = DifferentiableFK(model, root_name, joint_names, body_names).to(env.device)
    self.foot_site_ids, sites = robot.find_sites(
      ("left_foot", "right_foot"), preserve_order=True
    )
    if sites != ["left_foot", "right_foot"]:
      raise ValueError(f"Expected two foot sites, found {sites}")
    site_ids = [int(robot.indexing.site_ids[i]) for i in self.foot_site_ids]
    ankle_ids = [int(robot.indexing.body_ids[robot.body_names.index(n)]) for n in (
      "left_ankle_roll_link", "right_ankle_roll_link"
    )]
    if any(int(model.site_bodyid[s]) != b for s, b in zip(site_ids, ankle_ids, strict=True)):
      raise ValueError("Foot sites must be attached to their respective ankle links")
    offsets = torch.as_tensor(model.site_pos[site_ids].copy(), device=env.device, dtype=torch.float32)
    self.capsule_collision = G1CapsuleCollision(model, robot).to(env.device)
    self.policy = JointPolicyLearner(
      self.actor, self.wm, self.fk, cfg.policy, offsets, self.capsule_collision
    )
    self.iteration = 0

  def actor_inputs(self) -> TensorDict:
    hand = self.env.command_manager.get_command("hands")
    twist = self.env.command_manager.get_command("twist")
    return TensorDict(
      {
        "history": self.history,
        "command": torch.cat((twist, hand), dim=-1),
      },
      batch_size=[self.env.num_envs],
      device=self.env.device,
    )

  def step(self) -> dict[str, float]:
    reward_sum = 0.0
    velocity_error_sum = 0.0
    yaw_error_sum = 0.0
    hand_error_sum = torch.zeros(2, device=self.env.device)
    hand_angle_sum = torch.zeros(2, device=self.env.device)
    termination_sum = 0.0
    contact_sum = 0.0
    contact_disagree_sum = 0.0
    slip_sum = 0.0
    push_sum = 0.0
    for _ in range(self.cfg.collect_steps):
      with torch.no_grad():
        inputs = self.actor_inputs()
        self.actor.train()
        self.actor.update_normalization(inputs)
        action = (
          self.actor(inputs)
          + torch.randn(
            self.env.num_envs,
            self.env.action_manager.total_action_dim,
            device=self.env.device,
          )
          * self.cfg.action_noise
        )
        self.obs, reward, terminated, truncated, _ = self.collector.step(
          action, self.obs["actor"]
        )
        push_sum += float(self.collector.last_push_event.float().mean())
        sensor = sensor_from_real_obs(self.obs["actor"])
        shifted = torch.cat((self.history[:, 1:], sensor.unsqueeze(1)), dim=1)
        reset = (terminated | truncated).view(-1, 1, 1)
        self.history = torch.where(reset, sensor.unsqueeze(1), shifted)
        robot = self.env.scene["robot"]
        contact = self.env.scene["feet_ground_contact"].data.found > 0
        site_pos = robot.data.site_pos_w[:, self.foot_site_ids]
        site_velocity = robot.data.site_lin_vel_w[:, self.foot_site_ids, :2]
        proxy = foot_contact_proxy(site_pos[..., 2] - self.env.scene.env_origins[:, None, 2])
        contact_sum += float(contact.float().mean())
        contact_disagree_sum += float(((proxy >= 0.5) != contact).float().mean())
        slip_sum += float((site_velocity.norm(dim=-1) * contact).sum() / contact.sum().clamp_min(1))
        twist = self.env.command_manager.get_command("twist")
        actual_v = robot.data.root_link_lin_vel_b
        actual_w = robot.data.root_link_ang_vel_b
        velocity_error_sum += float(
          (actual_v[:, :2] - twist[:, :2]).norm(dim=-1).mean()
        )
        yaw_error_sum += float((actual_w[:, 2] - twist[:, 2]).abs().mean())
        hands = self.env.command_manager.get_term("hands")
        actual = hands.current_poses()
        target = hands.command.reshape(-1, 2, 7)
        hand_error_sum += (actual[..., :3] - target[..., :3]).norm(dim=-1).mean(0)
        dots = (actual[..., 3:] * target[..., 3:]).sum(-1).abs().clamp(0, 1)
        hand_angle_sum += (2 * dots.acos()).mean(0)
        termination_sum += float(terminated.float().mean())
        reward_sum += float(reward.mean())
    n = self.cfg.collect_steps
    result = {
      "sim_reward": reward_sum / n,
      "velocity_xy_error_mps": velocity_error_sum / n,
      "yaw_error_radps": yaw_error_sum / n,
      "left_hand_error_m": float(hand_error_sum[0] / n),
      "right_hand_error_m": float(hand_error_sum[1] / n),
      "left_hand_angle_rad": float(hand_angle_sum[0] / n),
      "right_hand_angle_rad": float(hand_angle_sum[1] / n),
      "terminated_fraction_per_step": termination_sum / n,
      "foot_contact_fraction": contact_sum / n,
      "foot_contact_proxy_disagreement": contact_disagree_sum / n,
      "contact_foot_slip_speed_mps": slip_sum / n,
      "push_fraction_per_step": push_sum / n,
    }
    if self.collector.buffer.num_transitions >= self.cfg.wm.horizon:
      wm_stats = self.wm.update()
      result.update({key: float(value) for key, value in wm_stats.items()})
      if not wm_stats:
        result["wm_skipped_no_valid_window"] = 1.0
      elif self.iteration >= self.cfg.warmup_iterations:
        result.update(self.policy.update())
    self.iteration += 1
    return result

  def save(self, path: str | Path) -> None:
    buffer = self.collector.buffer
    path = Path(path)
    with NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
      temporary_path = Path(stream.name)
    try:
      torch.save(
        {
        "version": 5,
        "config": asdict(self.cfg),
        "iteration": self.iteration,
        "actor": self.actor.state_dict(),
        "critic": self.policy.critic.state_dict(),
        "dynamics": self.wm.dynamics.state_dict(),
        "actor_optimizer": self.policy.actor_optimizer.state_dict(),
        "critic_optimizer": self.policy.critic_optimizer.state_dict(),
        "wm_optimizers": [x.state_dict() for x in self.wm.learner.optimizers],
        "replay": {k: v.clone() for k, v in buffer._storage.items()},
        "replay_step": buffer._step,
        "replay_count": buffer.num_transitions,
        "torch_rng": torch.get_rng_state(),
        },
        temporary_path,
      )
      temporary_path.replace(path)
    finally:
      temporary_path.unlink(missing_ok=True)

  def load(self, path: str | Path, *, allow_horizon_change: bool = False,
           allow_slip_proxy_change: bool = False,
           allow_capsule_weight_change: bool = False) -> None:
    data = torch.load(path, map_location=self.env.device, weights_only=False)
    saved_cfg = deepcopy(data["config"])
    current_cfg = asdict(self.cfg)
    # Version-4 checkpoints predating configurable slip use these exact values.
    saved_cfg["policy"].setdefault("slip_contact_height", 0.025)
    saved_cfg["policy"].setdefault("slip_contact_width", 0.01)
    saved_cfg["policy"].setdefault("capsule_collision_weight", 0.0)
    changes = {}
    if allow_horizon_change:
      for name in ("wm", "policy"):
        old, new = saved_cfg[name]["horizon"], current_cfg[name]["horizon"]
        if old != new:
          changes[name] = (old, new)
          saved_cfg[name]["horizon"] = new
    if allow_slip_proxy_change:
      for name in ("slip_contact_height", "slip_contact_width"):
        old, new = saved_cfg["policy"][name], current_cfg["policy"][name]
        if old != new:
          changes[name] = (old, new)
          saved_cfg["policy"][name] = new
    if allow_capsule_weight_change:
      name = "capsule_collision_weight"
      old, new = saved_cfg["policy"][name], current_cfg["policy"][name]
      if old != new:
        changes[name] = (old, new)
        saved_cfg["policy"][name] = new
    if data["version"] not in (4, 5) or saved_cfg != current_cfg:
      raise ValueError("Checkpoint version or runner configuration does not match")
    buffer = self.collector.buffer
    legacy = data["version"] == 4
    saved_fields = set(data["replay"])
    expected_fields = set(buffer._storage.keys())
    if saved_fields != (expected_fields - {"wm_invalid"} if legacy else expected_fields):
      raise ValueError("Checkpoint replay schema does not match")
    for key, value in data["replay"].items():
      if value.shape != buffer._storage[key].shape:
        raise ValueError(f"Checkpoint replay shape mismatch: {key}")
    self.actor.load_state_dict(data["actor"])
    self.policy.critic.load_state_dict(data["critic"])
    self.wm.dynamics.load_state_dict(data["dynamics"])
    self.policy.actor_optimizer.load_state_dict(data["actor_optimizer"])
    self.policy.critic_optimizer.load_state_dict(data["critic_optimizer"])
    for optim, state in zip(
      self.wm.learner.optimizers, data["wm_optimizers"], strict=True
    ):
      optim.load_state_dict(state)
    for key, value in data["replay"].items():
      buffer._storage[key].copy_(value)
    if legacy:
      buffer._storage["wm_invalid"].fill_(True)
      print("Version-4 replay has unknown push events; old transitions are excluded from world-model fitting.", flush=True)
    buffer._step, buffer.num_transitions = data["replay_step"], data["replay_count"]
    # The current simulator was freshly reset; do not connect its first state
    # to the last transition from the saved run.
    if buffer.num_transitions:
      buffer._storage["truncated"][:, (buffer._step - 1) % buffer.buffer_size] = True
    self.iteration = data["iteration"]
    if changes:
      print(f"Resuming with changed settings: {changes}; optimizer and critic state retained.", flush=True)
    torch.set_rng_state(data["torch_rng"].cpu())
    sensor = sensor_from_real_obs(self.obs["actor"])
    self.history = sensor.unsqueeze(1).repeat(1, self.cfg.history_length, 1)
