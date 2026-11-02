"""Uncertainty of the counts in a mask of an image stitched from two overlapping SPECT bed positions
(PGAAMultiBedSPECT.compute_uncertainty and the stitching weights of io.SPECT.stitch_beds).

The beds are cut from one acquisition of a small phantom, the way io.SPECT.read_dicom_beds fills each bed's
projections: the two beds hold the same counts in their overlap. The mask spans the seam of the stitched image.
References:
  * without PSF modelling every slice is reconstructed on its own, so the stitched image, and the uncertainty of its
    counts, are those of one long bed covering both
  * with PSF modelling, the exact linearised variance sum_j y_j (dS/dy_j)^2 of the stitched counts S, with the
    derivative taken by autograd through OSEM written with dense matrices of the same system matrix, through both beds
    and the stitch
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pytomography
from pytomography.algorithms import OSEM, PGAAMultiBedSPECT
from pytomography.callbacks import DataStorageCallback
from pytomography.io.SPECT import stitch_beds
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTPSFTransform

N, NA, LZ, SHIFT, DX = 8, 4, 12, 7, 4.0   # voxels across, angles, slices per bed, bed shift (slices), voxel size (mm)
LZ_LONG = SHIFT + LZ                      # slices of the stitched image
OFFSETS = [SHIFT, 0]                      # first slice of each bed in the stitched image: the upper bed is given first
N_ITERS, N_SUBSETS = 2, 2
SEAM = 10                                 # first stitched slice taken from the upper bed (midslice of the overlap)


def _system_matrix(Lz, psf, first_slice=0):
    """A bed whose first slice is slice ``first_slice`` of the stitched image."""
    centre = (0.0, 0.0, 100.0 + (first_slice + (Lz - 1) / 2) * DX)
    object_meta = SPECTObjectMeta([DX] * 3, (N, N, Lz), centre=centre, frame='patient')
    proj_meta = SPECTProjMeta((N, Lz), [DX, DX], list(np.linspace(0, 360, NA, endpoint=False)), radii=[150.0] * NA)
    transforms = [SPECTPSFTransform(psf_meta=SPECTPSFMeta((0.03, 1.0)))] if psf else []
    return SPECTSystemMatrix(obj2obj_transforms=transforms, proj2proj_transforms=[], object_meta=object_meta, proj_meta=proj_meta)


def _phantom_and_mask():
    """A cylinder with a hot ellipsoid across the seam; the mask is the ellipsoid."""
    x = torch.arange(N) - (N - 1) / 2
    X, Y, Z = torch.meshgrid(x, x, torch.arange(LZ_LONG).float(), indexing='ij')
    mask = (X - 0.5) ** 2 / 6 + (Y + 0.5) ** 2 / 6 + (Z - SEAM + 0.5) ** 2 / 12 <= 1
    phantom = 1.0 * (X ** 2 + Y ** 2 < (0.45 * N) ** 2) + 4.0 * mask
    return phantom.to(pytomography.device), mask.to(pytomography.device)


def _data(psf, shared=True, seed=0):
    """Photopeak and scatter window counts of each bed (as in the files' order), on the long grid if shared."""
    gen = torch.Generator().manual_seed(seed)
    phantom, mask = _phantom_and_mask()
    expected = _system_matrix(LZ_LONG, psf).forward(30 * phantom).cpu()
    scatter_expected = torch.full_like(expected, 0.4 * expected.mean().item())
    if shared:
        y, c = torch.poisson(expected, generator=gen), torch.poisson(scatter_expected, generator=gen)
        beds = [(y[..., o:o + LZ], c[..., o:o + LZ]) for o in OFFSETS]
    else:
        beds = [(torch.poisson(expected[..., o:o + LZ], generator=gen), torch.poisson(scatter_expected[..., o:o + LZ], generator=gen)) for o in OFFSETS]
    return beds, mask


def _likelihood(H, y, c, with_scatter_variance):
    """TEW-like scatter: half the scatter window, with variance 1/4 of its counts."""
    y, c = y.to(pytomography.device), c.to(pytomography.device)
    variance = (lambda x: x * (0.25 * c) * x) if with_scatter_variance else None
    return PoissonLogLikelihood(H, y, additive_term=0.5 * c, additive_term_variance_estimate=variance)


def _multibed(beds, psf, with_scatter_variance=False):
    algorithms = [OSEM(_likelihood(_system_matrix(LZ, psf, offset), y, c, with_scatter_variance))
                  for (y, c), offset in zip(beds, OFFSETS)]
    multibed = PGAAMultiBedSPECT(algorithms)
    callbacks = [DataStorageCallback(a.likelihood, torch.clone(a.object_prediction)) for a in algorithms]
    image = multibed(N_ITERS, N_SUBSETS, callback=callbacks)
    return multibed, callbacks, image


def _stitch_weights_z():
    """Stitched slices each bed gives the image: its own half of the overlap, without its first and last slice."""
    w = torch.zeros(2, LZ_LONG)
    w[1, 1:SEAM] = 1
    w[0, SEAM:LZ_LONG - 1] = 1
    return w


def test_stitching_weights_are_each_beds_own():
    metas = [_system_matrix(LZ, False, offset).object_meta for offset in OFFSETS]
    recons = torch.rand(2, N, N, LZ, generator=torch.Generator().manual_seed(1)).to(pytomography.device)
    stitched, meta = stitch_beds(recons, metas)
    weights, zs = stitch_beds(recons, metas, return_stitching_weights=True)
    assert meta.shape == (N, N, LZ_LONG) and np.allclose(meta.affine @ [0, 0, 0, 1], metas[1].affine @ [0, 0, 0, 1])
    assert tuple(weights.shape) == (2, N, N, LZ)
    assert list(zs) == OFFSETS
    rebuilt = torch.zeros_like(stitched)
    for k in range(2):
        rebuilt[..., zs[k]:zs[k] + LZ] += weights[k] * recons[k]
    assert torch.allclose(rebuilt, stitched)
    expected_z = _stitch_weights_z()
    for k in range(2):
        assert torch.equal(weights[k][0, 0].cpu(), expected_z[k, OFFSETS[k]:OFFSETS[k] + LZ])


def test_split_bed_without_psf_matches_one_long_bed():
    """No PSF: each slice is reconstructed on its own, so two beds cut from one acquisition give the image and the
    uncertainty of one long bed."""
    beds, mask = _data(psf=False)
    multibed, callbacks, image = _multibed(beds, psf=False)
    y_long = torch.cat([beds[1][0], beds[0][0][..., LZ - SHIFT:]], dim=-1)
    c_long = torch.cat([beds[1][1], beds[0][1][..., LZ - SHIFT:]], dim=-1)
    long_bed = OSEM(_likelihood(_system_matrix(LZ_LONG, psf=False), y_long, c_long, False))
    callback = DataStorageCallback(long_bed.likelihood, torch.clone(long_bed.object_prediction))
    image_long = long_bed(N_ITERS, N_SUBSETS, callback=callback)
    inner = slice(1, LZ_LONG - 1)   # stitch_beds leaves out the first and last slice of each bed
    assert torch.allclose(image[..., inner], image_long[..., inner], rtol=1e-4, atol=1e-4 * image_long.max().item())
    expected_abs, expected_pct = long_bed.compute_uncertainty(mask, callback, return_pct=True)
    uncertainty_abs, uncertainty_pct = multibed.compute_uncertainty(mask, callbacks, return_pct=True)
    assert uncertainty_abs == pytest.approx(expected_abs, rel=1e-3)
    assert uncertainty_pct == pytest.approx(expected_pct, rel=1e-3)


@pytest.fixture(scope='module')
def dense_bed_system():
    """The bed's system matrix with PSF as dense matrices: forward projection [n_proj, n_obj] and back projection
    [n_obj, n_proj] (each a column per unit vector; back projection is the transpose of forward projection only to
    interpolation accuracy)."""
    H = _system_matrix(LZ, psf=True)
    n_obj, n_proj = N * N * LZ, NA * N * LZ
    def unit(n, i, shape):
        e = torch.zeros(n, device=pytomography.device)
        e[i] = 1
        return e.reshape(shape)
    forward = torch.stack([H.forward(unit(n_obj, i, (N, N, LZ))).reshape(-1).cpu() for i in range(n_obj)], 1)
    backward = torch.stack([H.backward(unit(n_proj, j, (NA, N, LZ))).reshape(-1).cpu() for j in range(n_proj)], 1)
    return forward.double(), backward.double()


def _osem_dense(forward, backward, y, s):
    """OSEM as OSEM(PoissonLogLikelihood) does it, with dense matrices: differentiable."""
    angle = torch.arange(forward.shape[0]) // (N * LZ)
    f = torch.ones(forward.shape[1], dtype=torch.float64)
    for _ in range(N_ITERS):
        for m in range(N_SUBSETS):
            rows = angle % N_SUBSETS == m
            H_m, HT_m = forward[rows], backward[:, rows]
            norm = HT_m.sum(dim=1)
            ratio = y[rows] / (H_m @ f + s[rows] + pytomography.delta)
            f = torch.clamp(f + f / (norm + pytomography.delta) * (HT_m @ ratio - norm), min=0)
    return f


@pytest.mark.parametrize('shared, with_scatter_variance', [(True, False), (True, True), (False, True)])
def test_split_bed_with_psf_matches_exact_linearised_variance(tmp_path, dense_bed_system, shared, with_scatter_variance):
    beds, mask = _data(psf=True, shared=shared)
    multibed, callbacks, image = _multibed(beds, psf=True, with_scatter_variance=with_scatter_variance)
    uncertainty_abs, uncertainty_pct = multibed.compute_uncertainty(mask, callbacks, return_pct=True, include_additive_term=with_scatter_variance)

    # Reference: autograd through dense OSEM of both beds and the stitch
    forward, backward = dense_bed_system
    w = _stitch_weights_z().double()
    ys = [y.double().reshape(-1).clone().requires_grad_(True) for y, _ in beds]
    cs = [c.double().reshape(-1).clone().requires_grad_(True) for _, c in beds]
    counts = 0
    for k, offset in enumerate(OFFSETS):
        mask_k = (mask.cpu().double() * w[k])[..., offset:offset + LZ].reshape(-1)
        counts = counts + (_osem_dense(forward, backward, ys[k], 0.5 * cs[k]) * mask_k).sum()
    counts.backward()
    assert (image * mask).sum().item() == pytest.approx(counts.item(), rel=1e-4)
    def variance(values, grads, data_variance):
        """sum over measured bins of variance * (derivative)^2; shared bins once, with the beds' derivatives added"""
        if not shared:
            return sum((data_variance * v.double().reshape(-1) * g ** 2).sum().item() for v, g in zip(values, grads))
        total = torch.zeros(NA, N, LZ_LONG, dtype=torch.float64)
        data = torch.zeros(NA, N, LZ_LONG, dtype=torch.float64)
        for v, g, offset in zip(values, grads, OFFSETS):
            total[..., offset:offset + LZ] += g.reshape(NA, N, LZ)
            data[..., offset:offset + LZ] = v.double()
        return (data_variance * data * total ** 2).sum().item()
    var_reference = variance([y for y, _ in beds], [y.grad for y in ys], 1.0)
    if with_scatter_variance:
        # d/dc = 0.5 d/ds and var(c) = c: the same as var(s) = c/4 with d/ds
        var_reference += variance([c for _, c in beds], [c.grad for c in cs], 1.0)
    reference = np.sqrt(var_reference)
    assert uncertainty_abs == pytest.approx(reference, rel=1e-3)
    assert uncertainty_pct == pytest.approx(100 * reference / counts.item(), rel=1e-3)
    if shared:
        # the beds' covariance matters here: adding the beds' variances as if independent is well off
        var_independent = sum((y.double().reshape(-1) * g.grad ** 2).sum().item() for (y, _), g in zip(beds, ys))
        if with_scatter_variance:
            var_independent += sum((c.double().reshape(-1) * g.grad ** 2).sum().item() for (_, c), g in zip(beds, cs))
        assert abs(np.sqrt(var_independent) / reference - 1) > 0.02
