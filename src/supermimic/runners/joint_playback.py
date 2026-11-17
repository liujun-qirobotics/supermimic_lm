"""Fixed-command policy playback and first-episode evaluation for G1."""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from supermimic.joint_actor import JointCommandActor
from supermimic.learners.joint_policy_learner import sensor_from_real_obs
from supermimic.tasks.joint_env import joint_g1_env_cfg

SCENARIOS = {
  "stand": (0.0, 0.0, 0.0),
  "forward": (0.3, 0.0, 0.0),
  "backward": (-0.2, 0.0, 0.0),
  "left": (0.0, 0.2, 0.0),
  "right": (0.0, -0.2, 0.0),
  "turn_left": (0.0, 0.0, 0.3),
  "turn_right": (0.0, 0.0, -0.3),
}


def fixed_env_cfg(command, num_envs, duration, seed, perturbations=False):
  cfg = joint_g1_env_cfg()
  cfg.scene.num_envs, cfg.seed = num_envs, seed
  cfg.episode_length_s = max(20.0, duration + 2.0)
  cfg.observations["actor"].enable_corruption = False
  if not perturbations:
    cfg.events.pop("push_robot", None)
  cmd = cfg.commands["twist"]
  cmd.ranges.lin_vel_x = (command[0], command[0])
  cmd.ranges.lin_vel_y = (command[1], command[1])
  cmd.ranges.ang_vel_z = (command[2], command[2])
  cmd.resampling_time_range = (10000.0, 10000.0)
  cmd.rel_standing_envs = float(math.sqrt(sum(x*x for x in command)) < 0.05)
  cmd.rel_forward_envs = 0.0
  cmd.init_velocity_prob = 0.0
  cfg.commands["hands"].resampling_time_range = (10000.0, 10000.0)
  return cfg


class FixedJointPolicy:
  """10-frame Actor wrapper with fixed absolute pelvis-frame hand poses."""

  def __init__(self, env, checkpoint, command, shoulder=0.0, elbow=0.8):
    self.env = env
    config = checkpoint["config"]
    if config["history_length"] != 10 or checkpoint.get("version") not in (4, 5):
      raise ValueError("Expected a trusted v4/v5 joint checkpoint with 10-frame history")
    self.actor = JointCommandActor(
      93, 10, env.action_manager.total_action_dim, tuple(config["actor_hidden_dims"])
    ).to(env.device)
    self.actor.load_state_dict(checkpoint["actor"])
    self.actor.eval()
    self.command = torch.tensor(command, device=env.device, dtype=torch.float32)
    self.hands = env.command_manager.get_term("hands")
    robot = env.scene["robot"]
    q = robot.data.default_joint_pos.clone()
    for side in ("left", "right"):
      q[:, robot.joint_names.index(f"{side}_shoulder_pitch_joint")] = shoulder
      q[:, robot.joint_names.index(f"{side}_elbow_joint")] = elbow
    limits = robot.data.soft_joint_pos_limits
    if ((q < limits[..., 0]) | (q > limits[..., 1])).any():
      raise ValueError("Requested fixed arm reference exceeds a joint soft limit")
    root = torch.zeros(env.num_envs, 3, device=env.device)
    quat = torch.zeros(env.num_envs, 4, device=env.device)
    quat[:, 0] = 1.0
    pos, orientation = self.hands.fk(root, quat, q)
    self.hand_target = torch.cat((pos, orientation), dim=-1).flatten(start_dim=1)
    self.history = None
    self.pending_reset = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, dones=None):
    self.pending_reset = (torch.ones_like(self.pending_reset) if dones is None
                          else dones.to(device=self.env.device, dtype=torch.bool))

  def __call__(self, obs):
    sensor = sensor_from_real_obs(obs["actor"])
    if self.history is None:
      self.history = sensor.unsqueeze(1).repeat(1, 10, 1)
    else:
      shifted = torch.cat((self.history[:, 1:], sensor.unsqueeze(1)), dim=1)
      self.history = torch.where(self.pending_reset[:, None, None], sensor.unsqueeze(1), shifted)
    self.pending_reset.zero_()
    # Commands are restored before every Actor call, including after auto-reset.
    twist = self.env.command_manager.get_term("twist")
    twist.vel_command_b[:] = self.command
    self.hands._command[:] = self.hand_target
    cmd = torch.cat((self.command.expand(self.env.num_envs, -1), self.hand_target), -1)
    return self.actor(TensorDict({"history": self.history, "command": cmd}, [self.env.num_envs]))


@torch.no_grad()
def evaluate_fixed(env, policy, duration):
  """First episode per environment, omitting post-reset states from metrics."""
  steps = int(round(duration / env.step_dt))
  if steps < 1:
    raise ValueError("duration must contain at least one simulation step")
  n, device = env.num_envs, env.device
  alive = torch.ones(n, device=device, dtype=torch.bool)
  termination = torch.zeros(n, device=device, dtype=torch.bool)
  timeout = torch.zeros_like(termination)
  first_end = torch.full((n,), float(steps * env.step_dt), device=device)
  totals = torch.zeros(10, device=device)
  valid_count = torch.zeros((), device=device)
  slip_sum = torch.zeros((), device=device)
  slip_count = torch.zeros((), device=device)
  obs, _ = env.reset()
  policy.reset()
  for step in range(steps):
    action = policy(obs)
    obs, _, term, trunc, _ = env.step(action)
    ended = term | trunc
    first = alive & ended
    termination |= alive & term
    timeout |= alive & trunc
    first_end = torch.where(first, (step + 1) * env.step_dt, first_end)
    valid = alive & ~ended
    robot = env.scene["robot"]
    dv = robot.data.root_link_lin_vel_b[:, :2] - policy.command[:2]
    dw = robot.data.root_link_ang_vel_b[:, 2] - policy.command[2]
    actual = policy.hands.current_poses()
    target = policy.hand_target.reshape(n, 2, 7)
    hand_error = (actual[..., :3] - target[..., :3]).norm(dim=-1)
    angle = 2 * (actual[..., 3:] * target[..., 3:]).sum(-1).abs().clamp(0, 1).acos()
    collision = env.scene["self_collision"].data.force_history
    hit = (collision.norm(dim=-1) > 10.0).any(dim=(-1, -2))
    values = torch.stack((dv.square().sum(-1), dw.square(), hand_error[:, 0],
                          hand_error[:, 1], angle[:, 0], angle[:, 1], hit.float(),
                          robot.data.root_link_lin_vel_b[:, 0],
                          robot.data.root_link_lin_vel_b[:, 1],
                          robot.data.root_link_ang_vel_b[:, 2]), -1)
    totals += values[valid].sum(0)
    valid_count += valid.sum()
    feet_ids, _ = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
    feet_speed = robot.data.site_lin_vel_w[:, feet_ids, :2].norm(dim=-1)
    contact = env.scene["feet_ground_contact"].data.found > 0
    mask = valid[:, None] & contact
    slip_sum += feet_speed[mask].sum()
    slip_count += mask.sum()
    alive &= ~ended
    policy.reset(ended)
    if not alive.any():
      break
  count = int(valid_count)
  mean = (totals / valid_count.clamp_min(1)).tolist()
  return {
    "command_mps_radps": policy.command.tolist(), "trials": n,
    "hand_target_pelvis_pos_quat_wxyz": policy.hand_target[0].reshape(2, 7).tolist(),
    "duration_s": steps * env.step_dt, "terminated_trials": int(termination.sum()),
    "truncated_trials": int(timeout.sum()), "survived_fraction": float(alive.float().mean()),
    "mean_time_to_first_end_s": float(first_end.mean()), "valid_post_step_samples": count,
    "velocity_xy_rmse_mps": math.sqrt(mean[0]) if count else None,
    "yaw_rmse_radps": math.sqrt(mean[1]) if count else None,
    "left_hand_position_mean_m": mean[2] if count else None,
    "right_hand_position_mean_m": mean[3] if count else None,
    "left_hand_orientation_mean_rad": mean[4] if count else None,
    "right_hand_orientation_mean_rad": mean[5] if count else None,
    "self_collision_history_fraction": mean[6] if count else None,
    "mean_actual_vx_mps": mean[7] if count else None,
    "mean_actual_vy_mps": mean[8] if count else None,
    "mean_actual_yaw_radps": mean[9] if count else None,
    "contact_slip_mean_mps": float(slip_sum / slip_count) if slip_count else None,
    "contact_foot_samples": int(slip_count),
  }
