"""Actor for simultaneous locomotion and bimanual commands."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.models import _CudaGraphNormalization, build_mlp


class JointCommandActor(nn.Module):
  """One network sees all commands at once and outputs all joint actions.

  Expects ``history`` with shape [B, H, obs_dim] and ``command`` with
  shape [B, 17]. Output is a raw action; the environment maps it to joints.
  """

  def __init__(
    self,
    obs_dim: int,
    history_length: int,
    num_actions: int,
    hidden_dims: tuple[int, ...] = (512, 256, 128),
  ) -> None:
    super().__init__()
    self.obs_dim = obs_dim
    self.history_length = history_length
    self.obs_normalizer = _CudaGraphNormalization(obs_dim)
    self.command_normalizer = _CudaGraphNormalization(17)
    self.network = build_mlp(obs_dim * history_length + 17, hidden_dims, num_actions)

  def forward(self, obs: TensorDict) -> torch.Tensor:
    history = self.obs_normalizer(obs["history"]).flatten(start_dim=1)
    command = self.command_normalizer(obs["command"])
    return self.network(torch.cat((history, command), dim=-1))

  def update_normalization(self, obs: TensorDict) -> None:
    self.obs_normalizer.update(obs["history"][:, -1])
    self.command_normalizer.update(obs["command"])
