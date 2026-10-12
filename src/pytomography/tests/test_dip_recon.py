"""DIPRecon's image update is the positive root of x^2 - a x - b = 0 (the maximum of the EM surrogate plus the penalty
rho/2 |x - f + mu|^2). Where a < 0 (the EM update outweighs the network) the textbook formula (a + sqrt(a^2 + 4b)) / 2
loses most of its digits in float32; the form used now does not."""
from __future__ import annotations

import torch

from pytomography.algorithms.dip_recon import _positive_root


def _reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The root in float64, each sign of a with the formula that has no cancellation."""
    a, b = a.double(), b.double()
    root = torch.sqrt(a * a + 4 * b)
    return torch.where(a > 0, (a + root) / 2, 2 * b / (root - a))


def test_positive_root_is_accurate_where_the_terms_cancel():
    a = torch.tensor([-1e5, -1e4, -1e3, -10.0, -1.0, 1e-3, 1.0, 1e3, 1e5])
    b = torch.tensor([1e-2, 3e-2, 0.3, 1.0, 2.0, 1e-4, 3.0, 7.0, 0.5])
    x = _positive_root(a, b)
    expected = _reference(a, b)
    assert ((x.double() - expected).abs() / expected).max() < 1e-6
    naive = 0.5 * (a + torch.sqrt(a * a + 4 * b))           # the formula DIPRecon used
    assert ((naive.double() - expected).abs() / expected)[0] > 0.1   # most of its digits are gone at a = -1e5


def test_positive_root_edge_cases():
    a = torch.tensor([-3.0, 0.0, 2.0, 0.0])
    b = torch.tensor([0.0, 0.0, 0.0, 4.0])
    x = _positive_root(a, b)
    assert torch.equal(x, torch.tensor([0.0, 0.0, 2.0, 2.0]))   # x^2 = a x when b = 0; x = 2 for x^2 = 4
    assert torch.isfinite(x).all()
