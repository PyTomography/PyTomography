"""Multiplying a system matrix by a number gives a new, scaled system matrix and leaves the original unchanged."""
from __future__ import annotations

import torch

from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.projectors import ExtendedSystemMatrix

from test_contracts_spect import tiny_spect_system


def test_scaling_leaves_the_original_unchanged():
    H, f, g = tiny_spect_system()
    forward, backward = H.forward(f), H.backward(g)
    scaled = 2.5 * H
    assert scaled is not H
    torch.testing.assert_close(H.forward(f), forward)
    torch.testing.assert_close(H.backward(g), backward)
    torch.testing.assert_close(scaled.forward(f), 2.5 * forward)
    torch.testing.assert_close(scaled.backward(g), 2.5 * backward)
    torch.testing.assert_close((H * 2.5).forward(f), 2.5 * forward)   # either side


def test_scaling_twice_multiplies_the_factors():
    H, f, _ = tiny_spect_system()
    torch.testing.assert_close((3 * (2 * H)).forward(f), 6 * H.forward(f))


def test_the_same_system_matrix_scaled_twice_is_scaled_once_each_time():
    # The joint dual-peak pattern: a peak's scaled system matrix alone, and again inside the joint one
    H, f, _ = tiny_spect_system()
    alone, in_joint = 4 * H, 4 * H
    torch.testing.assert_close(alone.forward(f), 4 * H.forward(f))
    torch.testing.assert_close(in_joint.forward(f), 4 * H.forward(f))


def test_a_scaled_system_matrix_keeps_its_own_subsets():
    H, f, _ = tiny_spect_system()
    full = H.forward(f)
    scaled = 2 * H
    scaled.set_n_subsets(3)
    H.set_n_subsets(4)   # the original's subsets don't change the scaled one's
    for m in range(3):
        torch.testing.assert_close(scaled.forward(f, m), 2 * scaled.get_projection_subset(full, m), rtol=1e-4, atol=1e-6)
    norm = scaled.compute_normalization_factor(0)
    H.set_n_subsets(3)
    torch.testing.assert_close(norm, 2 * H.compute_normalization_factor(0), rtol=1e-4, atol=1e-6)


def test_scaled_system_matrices_reconstruct_jointly():
    # [2H, 3H] stacked, with data [2Hf, 3Hf]: every OSEM update is the one of H with data Hf, so the images agree
    H, f, _ = tiny_spect_system(attenuation=False, psf=False)
    joint = ExtendedSystemMatrix(system_matrices=[2 * H, 3 * H])
    data = H.forward(f)
    recon_joint = OSEM(PoissonLogLikelihood(joint, torch.stack([2 * data, 3 * data])))(n_iters=5, n_subsets=3)
    recon_alone = OSEM(PoissonLogLikelihood(H, data))(n_iters=5, n_subsets=3)
    torch.testing.assert_close(recon_joint, recon_alone, rtol=1e-4, atol=1e-5)
    assert joint.estimate_memory(n_subsets=3) is not None
