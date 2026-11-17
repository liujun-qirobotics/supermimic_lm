"""Reference-trained velocity Encoder inference without a motion reference."""

import math

import torch
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from tensordict import TensorDict

from supermimic.models import MultiEncoderAE
from supermimic.runners.joint_playback import SCENARIOS
from supermimic.representations import actor_goal_dim
from supermimic.tasks.env_cfgs import _sensor_actor_group
from supermimic.tasks.task_spec import TaskSpec


def velocity_env_cfg(command, num_envs, duration, seed):
  cfg = unitree_g1_flat_env_cfg()
  cfg.scene.num_envs, cfg.seed = num_envs, seed
  cfg.episode_length_s = max(20.0, duration + 2.0)
  cfg.observations["actor"] = _sensor_actor_group()
  cfg.events.pop("push_robot", None)
  term = cfg.commands["twist"]
  term.ranges.lin_vel_x = (command[0], command[0])
  term.ranges.lin_vel_y = (command[1], command[1])
  term.ranges.ang_vel_z = (command[2], command[2])
  term.ranges.heading = None
  term.heading_command = False
  term.rel_heading_envs = term.rel_world_envs = term.rel_forward_envs = 0.0
  term.rel_standing_envs = float(math.sqrt(sum(x*x for x in command)) < 0.05)
  term.init_velocity_prob = 0.0
  term.resampling_time_range = (10000.0, 10000.0)
  return cfg


class ReferenceVelocityPolicy:
  def __init__(self, env, checkpoint, command, joystick=None):
    if checkpoint.get("kind") != "reference_velocity_v1" or checkpoint.get("stage") != "velocity":
      raise ValueError("Expected a trained reference velocity-stage checkpoint")
    cfg = checkpoint["config"]
    if cfg["actor_obs"] != "proprio" or cfg["history_length"] != 10:
      raise ValueError("Expected the 10-frame proprioception actor")
    specs = [TaskSpec("motion", actor_goal_dim(checkpoint["num_goals"]), lambda *_: None),
             TaskSpec("velocity", 3, lambda *_: None)]
    self.actor = MultiEncoderAE(93, env.action_manager.total_action_dim, specs,
                                {"motion": 0.0, "velocity": 1.0}, cfg["latent_dim"],
                                tuple(cfg["actor_hidden_dims"]), cfg["latent_bc_weight"], 10).to(env.device)
    self.actor.load_state_dict(checkpoint["actor"])
    self.actor.eval()
    self.env = env
    self.command = torch.tensor(command, device=env.device, dtype=torch.float32)
    self.joystick = joystick
    self.history = None
    self.pending_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, dones=None):
    self.pending_reset = (torch.ones_like(self.pending_reset) if dones is None else
                          dones.to(device=self.env.device, dtype=torch.bool))

  def __call__(self, obs):
    if self.joystick is not None:
      self.command[:] = torch.tensor(self.joystick.read(), device=self.env.device)
    sensor = obs["actor"]
    if sensor.shape[-1] != 93:
      raise ValueError("Expected 93-D motion-training proprioception")
    if self.history is None:
      self.history = sensor.unsqueeze(1).repeat(1, 10, 1)
    else:
      shifted = torch.cat((self.history[:, 1:], sensor.unsqueeze(1)), dim=1)
      self.history = torch.where(self.pending_reset[:, None, None], sensor.unsqueeze(1), shifted)
    self.pending_reset.zero_()
    term = self.env.command_manager.get_term("twist")
    term.vel_command_b[:] = self.command
    if self.joystick is not None:
      term.is_standing_env.fill_(False)
    return self.actor(TensorDict({"history": self.history,
                                  "goal_velocity": self.command.expand(self.env.num_envs, -1)},
                                 [self.env.num_envs]))


@torch.no_grad()
def evaluate_velocity(env, policy, duration):
  steps = int(round(duration / env.step_dt))
  if steps < 1:
    raise ValueError("duration too short")
  n, device = env.num_envs, env.device
  alive = torch.ones(n, dtype=torch.bool, device=device)
  falls = torch.zeros_like(alive)
  timeouts = torch.zeros_like(alive)
  end_time = torch.full((n,), steps * env.step_dt, device=device)
  sums = torch.zeros(3, device=device)
  counts = torch.zeros((), device=device)
  obs, _ = env.reset()
  policy.reset()
  for t in range(steps):
    action = policy(obs)
    obs, _, term, trunc, _ = env.step(action)
    end = term | trunc
    end_time = torch.where(alive & end, (t + 1) * env.step_dt, end_time)
    falls |= alive & term
    timeouts |= alive & trunc
    valid = alive & ~end
    data = env.scene["robot"].data
    dv = data.root_link_lin_vel_b[:, :2] - policy.command[:2]
    dw = data.root_link_ang_vel_b[:, 2] - policy.command[2]
    collision = env.scene["self_collision"].data.force_history
    hit = (collision.norm(dim=-1) > 10.0).any(dim=(-1, -2)).float()
    sums += torch.stack((dv.square().sum(-1), dw.square(), hit), -1)[valid].sum(0)
    counts += valid.sum()
    alive &= ~end
    policy.reset(end)
    if not alive.any():
      break
  count = int(counts)
  mean = (sums / counts.clamp_min(1)).tolist()
  return {"trials": n, "command": policy.command.tolist(),
          "survived_fraction": float(alive.float().mean()),
          "terminated_trials": int(falls.sum()), "truncated_trials": int(timeouts.sum()),
          "mean_time_to_first_end_s": float(end_time.mean()), "valid_samples": count,
          "velocity_xy_rmse_mps": math.sqrt(mean[0]) if count else None,
          "yaw_rmse_radps": math.sqrt(mean[1]) if count else None,
          "self_collision_history_fraction": mean[2] if count else None}
