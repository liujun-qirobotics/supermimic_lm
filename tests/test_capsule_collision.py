import torch

from supermimic.utils.capsule_collision import segment_distance


def test_segment_distance_crossing_parallel_and_endpoint():
  a = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
  b = torch.tensor([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
  c = torch.tensor([[0.5, -1., 0.], [0., 1., 0.], [2., 1., 0.]])
  d = torch.tensor([[0.5, 1., 0.], [1., 1., 0.], [2., 2., 0.]])
  distance = segment_distance(a, b, c, d)
  torch.testing.assert_close(distance, torch.tensor([0., 1., 2**0.5]), atol=1e-7, rtol=1e-6)


def test_segment_distance_has_finite_nonzero_gradient_near_contact():
  a = torch.tensor([[0., 0., 0.]], requires_grad=True)
  b = torch.tensor([[1., 0., 0.]])
  c = torch.tensor([[0.5, -1., 0.03]])
  d = torch.tensor([[0.5, 1., 0.03]])
  cost = (0.1 - segment_distance(a, b, c, d)).clamp_min(0).square().sum()
  cost.backward()
  assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
