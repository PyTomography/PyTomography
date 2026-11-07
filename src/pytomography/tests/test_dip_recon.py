"""DIPRecon's image update is the positive root of x^2 - a x - b = 0 (the maximum of the EM surrogate plus the penalty
rho/2 |x - f + mu|^2). Where a < 0 (the EM update outweighs the network) the textbook formula (a + sqrt(a^2 + 4b)) / 2
loses most of its digits in float32; the form used now does not. Its memory estimate is OSEM's and DIP's own images."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from pytomography.algorithms import OSEM
from pytomography.algorithms.dip_recon import DIPRecon, _positive_root
from pytomography.metadata import ObjectMeta
from pytomography.projectors import SystemMatrix
from pytomography.utils.memory import MemoryPart


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


class _ToySystemMatrix(SystemMatrix):
    """An 8^3 grid; its own estimate is a fixed 1 GB of projections, so DIP's images are what the test sees."""
    def __init__(self):
        self.object_meta = ObjectMeta(dr=(1, 1, 1), shape=(8, 8, 8))
        self.proj_meta = None

    def _memory_parts(self, n_subsets, N_splits):
        return [MemoryPart('projections', ram_bytes=1e9 / n_subsets, scope='subset')]

    def forward(self, object, **kwargs): ...
    def backward(self, proj, **kwargs): ...
    def get_subset_splits(self, n_subsets): ...


class _Network:
    def predict(self):
        return torch.ones(8, 8, 8)


def _total(estimate):
    return sum(p.ram_bytes + p.gpu_bytes for p in estimate.parts)


def _total_of(estimate, name):
    return [p.ram_bytes + p.gpu_bytes for p in estimate.parts if p.name.startswith(name)]


def test_estimate_is_the_em_updates_and_twelve_images_and_the_network():
    """DIPRecon's estimate is OSEM's (the same system matrix and data) plus DIP's twelve images, and the network's fit
    when it is given; without it, the title says the fit is not counted."""
    likelihood = SimpleNamespace(system_matrix=_ToySystemMatrix(), projections=torch.zeros(1000), additive_term=None)
    dip = DIPRecon(likelihood, _Network(), rho=1.0)
    em = OSEM(likelihood, object_initial=torch.ones(8, 8, 8)).estimate_memory(n_subsets=2)
    without = dip.estimate_memory(n_subsets_osem=2)
    assert {p.name for p in em.parts} < {p.name for p in without.parts}
    assert sum(_total_of(without, 'DIP:')) == 12 * 4 * 8 ** 3
    assert _total(without) > _total(em) + 12 * 4 * 8 ** 3                      # with the allocator's share of them
    assert "without the network's fit" in without.title and without.settings['n_subsets'] == 2
    with_network = dip.estimate_memory(n_subsets_osem=2, network_gb=0.5)
    assert sum(_total_of(with_network, "the network's fit")) == 0.5e9 and "without" not in with_network.title
    assert _total(with_network) > _total(without) + 0.5e9
    assert dict(without.alternatives)['4 subsets'].ram_gb < without.ram_gb      # fewer projections per subset
