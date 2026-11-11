"""Rotations of the SPECT projectors (utils.spatial, plain PyTorch since kornia was dropped: bit-identical to
kornia.geometry.transform.rotate, which they replaced) against an independent rotation computed in float64, and the
cached rotation grids and the optional attenuation probability cache, which must reproduce the uncached results exactly
(they only avoid recomputing angle-dependent constants). The transposes that back projection applies
(rotate_cached_transpose and the functions built on it) must be the exact transposes of the rotations."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pytomography
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform
from pytomography.transforms.shared import RotationTransform
from pytomography.utils import rotate_detector_z, rotate_detector_z_transpose
from pytomography.utils.spatial import (_ROTATION_GRIDS, _ROTATION_TRANSPOSES, rotate_cached, rotate_cached_transpose,
                                        rotation_transpose_bytes)

DEV = pytomography.device


def _reference_rotation(x: np.ndarray, angle: float, mode: str):
    """[C, H, W] rotated by ``angle`` degrees anti-clockwise as displayed (row 0 on top) about the centre
    ((W - 1) / 2, (H - 1) / 2), zero outside, in float64: output pixel (x, y) is the input at the centre plus the
    offset rotated back. Also returns where the source position is within 1e-4 pixel of a nearest-neighbour tie."""
    C, H, W = x.shape
    a = np.radians(angle)
    cos, sin = np.cos(a), np.sin(a)
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing='ij')
    cx, cy = (W - 1) / 2, (H - 1) / 2
    sx = cx + cos * (xx - cx) - sin * (yy - cy)
    sy = cy + sin * (xx - cx) + cos * (yy - cy)
    def tap(ix, iy):
        inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        values = x[:, np.clip(iy, 0, H - 1), np.clip(ix, 0, W - 1)]
        return np.where(inside, values, 0.0)
    ties = (np.abs(sx - np.floor(sx) - 0.5) < 1e-4) | (np.abs(sy - np.floor(sy) - 0.5) < 1e-4)
    if mode == 'nearest':
        return tap(np.rint(sx).astype(int), np.rint(sy).astype(int)), ties
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    wx, wy = sx - x0, sy - y0
    out = (tap(x0, y0) * (1 - wx) * (1 - wy) + tap(x0 + 1, y0) * wx * (1 - wy)
           + tap(x0, y0 + 1) * (1 - wx) * wy + tap(x0 + 1, y0 + 1) * wx * wy)
    return out, ties


@pytest.mark.parametrize('size', [(23, 23), (24, 24), (23, 26)], ids=['odd', 'even', 'non-square'])
def test_rotate_cached_matches_an_independent_rotation(size):
    """Bilinear within 5e-5 of the float64 rotation: the grid is float32 (kornia's normalisation, kept for
    bit-identical results, is float32 even for float64 images), measured within 8e-6 on the CPU and 1.2e-5 on an
    RTX 5090, whose float32 grid (3x3 inverses and affine_grid on the GPU, the same bits as kornia's) is up to 7e-7 of
    the half-width off the CPU's; a wrong direction or a half-pixel shift is off by 0.1 to 1. Nearest identical away
    from ties; multiples of 90 degrees are torch.rot90 in nearest mode (cos 90 is -4e-8 in float32, not 0)."""
    gen = torch.Generator().manual_seed(0)
    angles = (0.0, 33.7, 90.0, 137.25, 180.0, 270.0, 359.9, -45.0, -267.1875, 3.75)     # SPECT: -(270 - beta), 360/96
    for dtype in (torch.float32, torch.float64):
        x = torch.rand(1, 6, *size, generator=gen, dtype=dtype).to(DEV)
        for mode in ('bilinear', 'nearest'):
            for angle in angles:
                new = rotate_cached(x, torch.tensor(angle, device=DEV, dtype=dtype), mode=mode)
                assert torch.equal(new, rotate_cached(x, angle, mode=mode))           # second call: from the cache
                reference, ties = _reference_rotation(x[0].cpu().double().numpy(), angle, mode)
                new = new[0].cpu().double().numpy()
                if mode == 'bilinear':
                    assert np.abs(new - reference).max() < 5e-5, (angle, dtype)
                else:
                    assert np.array_equal(new[:, ~ties], reference[:, ~ties]), (angle, dtype)
                if angle % 90 == 0 and mode == 'nearest' and size[0] == size[1]:
                    k = int(angle // 90) % 4
                    assert torch.equal(torch.tensor(new), torch.rot90(x[0].cpu().double(), k, dims=(-2, -1)))
    assert len(_ROTATION_GRIDS) >= len(angles)


def _reference_rotate_z(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """An [Lx, Ly, Lz] object rotated about the z axis by ``angle_deg`` (anti-clockwise, as rotate_cached rotates
    [Lz, Lx, Ly]), bilinear, in float64."""
    rotated, _ = _reference_rotation(x.permute(2, 0, 1).cpu().double().numpy(), float(angle_deg), 'bilinear')
    return torch.tensor(rotated).permute(1, 2, 0)


def test_rotate_detector_z_negative_semantics_and_round_trip():
    """``rotate_detector_z(x, a)`` rotates by ``-(270 - a)`` and ``negative=True`` by ``+(270 - a)``. Rotating there
    and back at a multiple of 90 degrees is a pure permutation of the odd-sized grid (round-trip error ~1e-6 from
    cos/sin rounding); at a general angle two bilinear passes blur a sharp blob (sigma ~2.4 voxels), measured peak
    error 0.06-0.07, so the round trip is only required to be within 0.1 of the object while a single rotation alone
    moves it by more than 0.3."""
    ii, jj = torch.meshgrid(torch.arange(23.0), torch.arange(23.0), indexing="ij")
    blob = torch.exp(-((ii - 8) ** 2 + (jj - 13) ** 2) / 12.0).unsqueeze(-1).repeat(1, 1, 6).to(DEV)
    for angle in (0.0, 90.0, 180.0, 270.0, 23.5, 45.0, 137.25, 300.0):
        a = torch.tensor(angle, device=DEV)
        phi = 270 - angle
        fwd = rotate_detector_z(blob, a)
        back = rotate_detector_z(fwd, a, negative=True)
        assert torch.allclose(fwd.cpu().double(), _reference_rotate_z(blob, -phi), atol=1e-5), angle
        assert torch.allclose(back.cpu().double(), _reference_rotate_z(fwd, phi), atol=1e-5), angle
        if angle % 90 == 0:
            assert torch.allclose(back, blob, atol=1e-5), angle
        else:
            assert (fwd - blob).abs().max() > 0.3, angle          # the forward rotation really moved the blob
            assert (back - blob).abs().max() < 0.1, angle         # and negative=True brought it back


def _system_matrix(cache):
    dx = 5.0
    object_meta = SPECTObjectMeta([dx] * 3, (16, 16, 4))
    angles = list(range(0, 360, 30))
    proj_meta = SPECTProjMeta((16, 4), [dx, dx], angles, radii=[100.0] * len(angles))
    gen = torch.Generator().manual_seed(2)
    mu = (0.015 * torch.rand(16, 16, 4, generator=gen)).to(DEV)
    att = SPECTAttenuationTransform(attenuation_map=mu, cache_probabilities=cache)
    sm = SPECTSystemMatrix(obj2obj_transforms=[att], proj2proj_transforms=[], object_meta=object_meta, proj_meta=proj_meta)
    return sm, att, mu


def test_attenuation_probability_cache_is_exact_and_invalidates():
    gen = torch.Generator().manual_seed(3)
    f = torch.rand(16, 16, 4, generator=gen).to(DEV)
    g = torch.rand(12, 16, 4, generator=gen).to(DEV)
    sm0, _, mu = _system_matrix(False)
    sm1, att1, _ = _system_matrix(True)
    assert torch.equal(sm0.forward(f), sm1.forward(f))
    assert torch.equal(sm0.backward(g), sm1.backward(g))
    assert len(att1._prob_cache) == 12
    assert torch.equal(sm1.forward(f), sm0.forward(f))            # served from the cache
    # a new map must invalidate the cache
    att1.attenuation_map = 0.5 * mu
    sm0.obj2obj_transforms[0].attenuation_map = 0.5 * mu
    assert torch.equal(sm0.forward(f), sm1.forward(f))
    assert torch.equal(sm0.backward(g), sm1.backward(g))
    # so must new projection angles
    sm1.proj_meta.angles = sm1.proj_meta.angles + 7.0
    sm0.proj_meta.angles = sm0.proj_meta.angles + 7.0
    assert torch.equal(sm0.forward(f), sm1.forward(f))
    assert torch.equal(sm0.backward(g), sm1.backward(g))
    assert len(att1._prob_cache) == 12


@pytest.mark.parametrize('size', [(23, 23), (24, 24), (23, 26)], ids=['odd', 'even', 'non-square'])
def test_rotate_cached_transpose_is_the_exact_transpose(size):
    """The matrix of rotate_cached_transpose is the transpose of rotate_cached's, both built column by column from
    basis images: equal to float32 rounding (measured 1.2e-7 on the CPU; the weights are computed with grid_sample's own
    arithmetic on each device), exactly in nearest mode and in float64 to 1e-15, at multiples of 90 degrees too. The
    rotation by the opposite angle, which back projection applied before, differs at other angles (by more the further
    the angle is from a multiple of 90 degrees: 0.011 at 2.8 degrees from one)."""
    H, W = size
    for dtype in (torch.float32, torch.float64):
        basis = torch.eye(H * W, dtype=dtype, device=DEV).reshape(H * W, 1, H, W)
        for mode in ('bilinear', 'nearest'):
            for angle in (0.0, 33.7, 90.0, 137.25, 180.0, 270.0, -267.1875, 3.75):
                M_t = rotate_cached(basis, angle, mode=mode).reshape(H * W, H * W)            # row j: M e_j
                M = rotate_cached_transpose(basis, angle, mode=mode).reshape(H * W, H * W)    # row j: M^T e_j
                error = (M - M_t.T).abs().max().item()
                if mode == 'nearest':
                    assert error == 0, (angle, dtype)
                else:
                    assert error <= (3e-7 if dtype == torch.float32 else 1e-14), (angle, dtype, error)
                if mode == 'bilinear' and angle % 90 != 0:
                    inverse = rotate_cached(basis, -angle, mode=mode).reshape(H * W, H * W)
                    assert (inverse - M).abs().max() > 1e-3, angle                            # not the transpose
    for key, transpose in _ROTATION_TRANSPOSES.items():
        h, w, dtype = key[1], key[2], key[4]
        size_bytes = sum(t.numel() * t.element_size() for t in (transpose.crow_indices(), transpose.col_indices(), transpose.values()))
        assert size_bytes <= rotation_transpose_bytes(h, w, dtype)


def test_transposes_of_object_rotations_and_their_layouts():
    """RotationTransform.backward and rotate_detector_z_transpose are the exact transposes of RotationTransform.forward
    and rotate_detector_z on [Lx, Ly, Lz] objects (dot test, float64 sums of float32 arrays), the same for contiguous
    and permuted (non-contiguous) inputs, and deterministic (bit-identical when repeated)."""
    gen = torch.Generator().manual_seed(4)
    rotation = RotationTransform()
    for shape in ((31, 31, 7), (30, 34, 5)):
        x = torch.randn(*shape, generator=gen).to(DEV)
        y = torch.randn(*shape, generator=gen).to(DEV)
        for angle in (0.0, 90.0, 33.7, -137.25, 266.25):
            for forward, transpose in ((rotation.forward, rotation.backward),
                                       (lambda o, a: rotate_detector_z(o, torch.tensor(a)), lambda o, a: rotate_detector_z_transpose(o, torch.tensor(a)))):
                Rx = forward(x, angle)
                RTy = transpose(y, angle)
                lhs, rhs = float((Rx.double() * y.double()).sum()), float((x.double() * RTy.double()).sum())
                assert abs(lhs - rhs) <= 1e-6 * float(Rx.norm() * y.norm()), (shape, angle)
                y_permuted = y.permute(2, 0, 1).contiguous().permute(1, 2, 0)                     # same values, z not contiguous
                assert torch.equal(transpose(y_permuted, angle), RTy)
                assert torch.equal(transpose(y, angle), RTy)
    with pytest.raises(ValueError):
        rotate_cached_transpose(torch.ones(1, 1, 5, 5, device=DEV), 10.0, mode='bicubic')


def test_spect_back_projection_is_the_exact_transpose_of_forward_projection():
    """SPECTSystemMatrix.backward is the transpose of forward to float32 rounding, with subsets (here attenuation; the
    contract tests cover the PSF too): the rotation by the opposite angle, used before, left ~1e-4 (see
    test_contracts_spect)."""
    sm, _, _ = _system_matrix(False)
    gen = torch.Generator().manual_seed(5)
    f = torch.randn(16, 16, 4, generator=gen).to(DEV)
    g = torch.randn(12, 16, 4, generator=gen).to(DEV)
    sm.set_n_subsets(3)
    for subset_idx in (None, 1):
        gs = g if subset_idx is None else sm.get_projection_subset(g, subset_idx)
        Hf = sm.forward(f, subset_idx)
        lhs, rhs = float((Hf.double() * gs.double()).sum()), float((f.double() * sm.backward(gs, subset_idx).double()).sum())
        assert abs(lhs - rhs) <= 1e-7 * float(Hf.norm() * gs.norm()), subset_idx
