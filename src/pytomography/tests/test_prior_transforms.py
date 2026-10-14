"""Priors apply their obj2obj_transforms to their value and gradients at every voxel: Prior.__init__ stored an empty
list whatever it was given, and the nearest neighbour priors did not take the argument, so the loop that applies them
never ran."""
from __future__ import annotations

import pytest
import torch

import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.priors import LogCoshPrior, NearestNeighbourPrior, QuadraticPrior, RelativeDifferencePrior
from pytomography.transforms.shared import GaussianFilter

OBJECT_META = ObjectMeta(dr=(4.0, 4.0, 4.0), shape=(12, 10, 8))


def _gradient(prior, x):
    prior.set_object_meta(OBJECT_META)
    prior.set_object(x)
    prior.set_beta_scale(1.0)
    return prior(derivative_order=1)


@pytest.mark.parametrize('make', [lambda **kw: RelativeDifferencePrior(0.3, gamma=2, **kw),
                                  lambda **kw: QuadraticPrior(0.3, **kw), lambda **kw: LogCoshPrior(0.3, **kw)],
                         ids=['RDP', 'quadratic', 'logcosh'])
def test_transforms_are_applied_to_the_gradient(make):
    x = torch.rand(OBJECT_META.shape, generator=torch.Generator().manual_seed(0)).to(pytomography.device)
    plain = _gradient(make(), x)
    assert make().obj2obj_transforms == []
    smoothing = GaussianFilter(8.0)
    prior = make(obj2obj_transforms=[smoothing])
    assert prior.obj2obj_transforms == [smoothing]
    smoothed = _gradient(prior, x)
    assert smoothing.object_meta is OBJECT_META                          # configured by set_object_meta
    assert torch.allclose(smoothed, smoothing.forward(plain), rtol=1e-5, atol=1e-6 * plain.abs().max())
    assert (smoothed - plain).abs().max() > 0.1 * plain.abs().max()   # the filter really changed it


def test_nearest_neighbour_prior_takes_the_transforms():
    smoothing = GaussianFilter(8.0)
    assert NearestNeighbourPrior(1.0, obj2obj_transforms=(smoothing,)).obj2obj_transforms == [smoothing]
