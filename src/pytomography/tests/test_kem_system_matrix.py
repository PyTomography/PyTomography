"""KEMSystemMatrix (#228): its subsets are those of the system matrix it wraps, and so is get_weighting_subset (the
share of the projections in a subset, which scales a prior's beta and BSREM's step). It was aliased to
get_projection_subset, so a prior, BSREM or OSMAPOSL on a KEM system matrix raised a TypeError (one argument for two)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pytomography
from pytomography.algorithms import BSREM, OSMAPOSL
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.priors import RelativeDifferencePrior
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.projectors.shared import KEMSystemMatrix
from pytomography.transforms.shared import KEMTransform

SHAPE = (16, 16, 8)
N_ANGLES = 12


def _system_matrices():
    object_meta = SPECTObjectMeta((4.0, 4.0, 4.0), SHAPE)
    proj_meta = SPECTProjMeta(SHAPE[1:], (4.0, 4.0), list(np.linspace(0, 360, N_ANGLES, endpoint=False)),
                              radii=[150.0] * N_ANGLES)
    system_matrix = SPECTSystemMatrix([], [], object_meta, proj_meta)
    support = torch.rand(SHAPE, generator=torch.Generator().manual_seed(0))
    return system_matrix, KEMSystemMatrix(system_matrix, KEMTransform([support], size=3))


def test_subsets_and_weights_are_those_of_the_wrapped_system_matrix():
    system_matrix, kem = _system_matrices()
    kem.set_n_subsets(4)
    for subset_idx in (None, 0, 3):
        assert kem.get_weighting_subset(subset_idx) == system_matrix.get_weighting_subset(subset_idx)
    assert kem.get_weighting_subset(1) == pytest.approx(3 / N_ANGLES)
    assert all(torch.equal(a, b) for a, b in zip(kem.subset_indices_array, system_matrix.subset_indices_array))
    projections = torch.rand(N_ANGLES, *SHAPE[1:]).to(pytomography.device)
    assert torch.equal(kem.get_projection_subset(projections, 2), system_matrix.get_projection_subset(projections, 2))


@pytest.mark.parametrize('algorithm', [BSREM, OSMAPOSL])
def test_priors_run_on_a_kem_system_matrix(algorithm):
    """The paths that call get_weighting_subset(subset_idx): a prior's beta scale (both), BSREM's step."""
    system_matrix, kem = _system_matrices()
    truth = torch.zeros(SHAPE)
    truth[5:11, 5:11, 2:6] = 10.0
    projections = torch.poisson(system_matrix.forward(truth.to(pytomography.device)).cpu(),
                                generator=torch.Generator().manual_seed(1)).to(pytomography.device)
    reconstruction = algorithm(PoissonLogLikelihood(kem, projections), prior=RelativeDifferencePrior(beta=0.1, gamma=2))
    image = reconstruction(n_iters=2, n_subsets=4)
    assert image.shape == SHAPE and torch.isfinite(image).all() and image.sum() > 0
