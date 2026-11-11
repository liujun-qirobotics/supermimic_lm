import math

import torch
from tensordict import TensorDict

from supermimic.joint_actor import JointCommandActor
from supermimic.tasks.joint_command import (
  JointCommand,
  hand_pose_in_body_frame,
  joint_tracking_errors,
)


def test_body_frame_tracks_robot_translation_and_yaw():
  root_pos = torch.tensor([[1.0, 2.0, 0.0]])
  root_quat = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
  hand_pos = torch.tensor([[1.0, 3.0, 0.5]])
  result = hand_pose_in_body_frame(root_pos, root_quat, hand_pos, root_quat)
  assert torch.allclose(result, torch.tensor([[1., 0., 0.5, 1., 0., 0., 0.]]), atol=1e-6)


def test_pose_error_quaternion_sign_and_gradients():
  pose = torch.tensor([[0.1, 0.2, 0.3, 1., 0., 0., 0.]])
  target = pose.clone()
  target[:, 3:] *= -1
  cmd = JointCommand(torch.zeros(1, 3), target, target)
  actual = pose.clone().requires_grad_()
  errors = joint_tracking_errors(cmd, torch.zeros(1, 3), actual, actual)
  assert torch.allclose(errors["left_orientation"], torch.zeros(1))
  shifted = actual + torch.tensor([[0.1, 0., 0., 0., 0., 0., 0.]])
  errors = joint_tracking_errors(cmd, torch.zeros(1, 3), shifted, shifted)
  (errors["left_position"] + errors["right_position"]).sum().backward()
  assert torch.isfinite(actual.grad).all() and actual.grad[0, 0] != 0


def test_actor_takes_joint_commands_and_backpropagates_both_objectives():
  torch.manual_seed(3)
  actor = JointCommandActor(93, 2, 29, hidden_dims=(32,))
  command = torch.randn(4, 17, requires_grad=True)
  obs = TensorDict({"history": torch.randn(4, 2, 93), "command": command}, [4])
  actions = actor(obs)
  assert actions.shape == (4, 29)
  actions.square().sum().backward()
  assert torch.isfinite(command.grad).all()
  assert command.grad[:, :3].abs().sum() > 0
  assert command.grad[:, 3:].abs().sum() > 0


def test_joint_command_roundtrip():
  raw = torch.randn(8, 17)
  assert torch.equal(JointCommand.from_tensor(raw).as_tensor(), raw)
