"""External KEM (transforms/shared/external_kem.py): k-nearest-neighbour kernels from a prior, the deep-learned kernel,
and the sparse kernel transform that KEM applies."""
import numpy as np
import pytest
import torch

import pytomography
from pytomography.metadata import ObjectMeta, ProjMeta
from pytomography.transforms.shared import ExternalKEMTransform
from pytomography.transforms.shared.external_kem import (UNet3D, _construct_local_kernel_torch, construct_deep_kernel,
                                                         construct_kernel, train_deep_kernel)


def _prior(shape=(8, 9, 7), seed=0):
    rng = np.random.default_rng(seed)
    prior = rng.random(shape)
    prior[2:5, 3:6, 2:5] += 4.0  # a bright block, so that neighbours follow it
    return prior


@pytest.mark.parametrize('mode', ['local', 'global'])
def test_construct_kernel_rows_are_normalised_neighbours(mode):
    prior = _prior()
    weights, indices = construct_kernel(prior, k=6, patch=3, local_window=5, mode=mode)
    assert weights.shape == indices.shape == (prior.size, 6)
    assert weights.dtype == np.float32 and indices.dtype == np.int64
    assert np.allclose(weights.sum(axis=1), 1, atol=1e-5)
    assert (weights >= 0).all() and (indices >= 0).all() and (indices < prior.size).all()
    # the self voxel is left out by default
    assert not (indices == np.arange(prior.size)[:, None])[weights > 0].any()


def test_neighbours_of_the_bright_block_stay_in_it():
    prior = _prior()
    weights, indices = construct_kernel(prior, k=4, patch=1, sigmac=10.0, local_window=5)
    inside = np.zeros(prior.shape, bool)
    inside[2:5, 3:6, 2:5] = True
    centre = np.ravel_multi_index((3, 4, 3), prior.shape)
    assert inside.reshape(-1)[indices[centre]].all()


def test_the_tensor_backend_matches_numpy():
    prior = _prior()
    args = dict(k=5, sigmaf=1.0, sigmac=1.0, patch=3, local_window=5, include_self=False, chunk_size=64)
    w_np, i_np = construct_kernel(prior, mode='local', device='cpu', **args)
    w_t, i_t = _construct_local_kernel_torch(prior.astype(np.float64), device=torch.device('cpu'), **args)
    assert np.array_equal(i_np, i_t)
    assert np.allclose(w_np, w_t, atol=1e-6)


def test_construct_kernel_refuses_bad_arguments():
    prior = _prior()
    with pytest.raises(ValueError):
        construct_kernel(prior, patch=2)
    with pytest.raises(ValueError):
        construct_kernel(prior, k=500, local_window=3)
    with pytest.raises(ValueError):
        construct_kernel(prior, mode='other')
    with pytest.raises(ValueError):
        construct_kernel(prior[0])


def test_the_transform_back_projection_is_the_transpose():
    prior = _prior()
    weights, indices = construct_kernel(prior, k=6, patch=1, local_window=5)
    kernel = ExternalKEMTransform(weights, indices, chunk_size=97)
    kernel.configure(ObjectMeta(dr=(1.0, 1.0, 1.0), shape=prior.shape), ProjMeta([0.0]))
    g = torch.Generator().manual_seed(0)
    x = torch.rand(prior.shape, generator=g).to(pytomography.device, pytomography.dtype)
    y = torch.rand(prior.shape, generator=g).to(pytomography.device, pytomography.dtype)
    lhs = float((kernel.forward(x) * y).sum())
    rhs = float((x * kernel.backward(y)).sum())
    assert lhs == pytest.approx(rhs, rel=1e-5)
    # rows sum to one, so K maps a constant image to itself
    ones = torch.ones(prior.shape, device=pytomography.device, dtype=pytomography.dtype)
    assert torch.allclose(kernel.forward(ones), ones, atol=1e-5)


def test_the_transform_checks_its_kernel_and_grid():
    weights = np.full((10, 2), 0.5, np.float32)
    indices = np.zeros((10, 2), np.int64)
    with pytest.raises(ValueError):
        ExternalKEMTransform(weights, indices[:, :1])
    with pytest.raises(ValueError):
        ExternalKEMTransform(-weights, indices)
    with pytest.raises(ValueError):
        ExternalKEMTransform(weights, indices + 10)
    kernel = ExternalKEMTransform(weights, indices)
    with pytest.raises(ValueError):
        kernel.configure(ObjectMeta(dr=(1.0, 1.0, 1.0), shape=(3, 3, 3)), ProjMeta([0.0]))


def test_deep_kernel_trains_and_exports_a_normalised_kernel():
    torch.manual_seed(0)
    shape = (16, 16, 16)
    prior = _prior(shape).astype(np.float32)
    model = UNet3D(base_channels=2, feature_channels=3)
    history = train_deep_kernel(model, prior, prior * 0.5 + 0.1, epochs=1, patches_per_epoch=1, patch_size=16,
                                chunk_size=1024, k=4, local_window=3, device='cpu', verbose=False)
    assert len(history) == 1 and np.isfinite(history[0])
    weights, indices = construct_deep_kernel(model, prior, k=4, local_window=3, chunk_size=1024, device='cpu')
    assert weights.shape == indices.shape == (prior.size, 4)
    assert np.allclose(weights.sum(axis=1), 1, atol=1e-5)
    kernel = ExternalKEMTransform(weights, indices)
    kernel.configure(ObjectMeta(dr=(1.0, 1.0, 1.0), shape=shape), ProjMeta([0.0]))
    assert kernel.forward(torch.ones(shape)).shape == shape
