"""Cached rotation grids and the optional attenuation probability cache must reproduce the original
implementation exactly (they only avoid recomputing angle-dependent constants)."""
from __future__ import annotations

import torch
from kornia.geometry.transform import rotate

import pytomography
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform
from pytomography.utils import rotate_detector_z
from pytomography.utils.spatial import _ROTATION_GRIDS, rotate_cached

DEV = pytomography.device


def test_rotate_cached_matches_kornia():
    gen = torch.Generator().manual_seed(0)
    for dtype in (torch.float32, torch.float64):
        x = torch.rand(1, 6, 23, 23, generator=gen, dtype=dtype).to(DEV)
        for mode in ("bilinear", "nearest"):
            for angle in (0.0, 33.7, 90.0, 137.25, 180.0, 270.0, 359.9, -45.0):
                ref = rotate(x, torch.tensor([angle], device=DEV, dtype=dtype), mode=mode)
                new = rotate_cached(x, torch.tensor(angle, device=DEV, dtype=dtype), mode=mode)
                new_again = rotate_cached(x, angle, mode=mode)          # second call: from the cache
                assert torch.equal(ref, new), f"angle {angle} {mode} {dtype}"
                assert torch.equal(new, new_again)
    assert len(_ROTATION_GRIDS) >= 8


def _kornia_rotate_z(x, angle_deg):
    """Reference: kornia rotation of an [Lx, Ly, Lz] object about the z axis by ``angle_deg`` (anti-clockwise)."""
    return rotate(x.permute(2, 0, 1).unsqueeze(0), angle_deg.reshape(1), mode="bilinear").squeeze().permute(1, 2, 0)


def test_rotate_detector_z_negative_semantics_and_round_trip():
    """``rotate_detector_z(x, a)`` rotates by ``-(270 - a)`` and ``negative=True`` by ``+(270 - a)`` (bit-identical to
    kornia's ``rotate``). Rotating there and back at a multiple of 90 degrees is a pure permutation of the odd-sized
    grid (round-trip error ~1e-6 from cos/sin rounding); at a general angle two bilinear passes blur a sharp blob
    (sigma ~2.4 voxels), measured peak error 0.06-0.07, so the round trip is only required to be within 0.1 of the
    object while a single rotation alone moves it by more than 0.3."""
    ii, jj = torch.meshgrid(torch.arange(23.0), torch.arange(23.0), indexing="ij")
    blob = torch.exp(-((ii - 8) ** 2 + (jj - 13) ** 2) / 12.0).unsqueeze(-1).repeat(1, 1, 6).to(DEV)
    for angle in (0.0, 90.0, 180.0, 270.0, 23.5, 45.0, 137.25, 300.0):
        a = torch.tensor(angle, device=DEV)
        phi = 270 - a
        fwd = rotate_detector_z(blob, a)
        back = rotate_detector_z(fwd, a, negative=True)
        assert torch.equal(fwd, _kornia_rotate_z(blob, -phi)), angle
        assert torch.equal(back, _kornia_rotate_z(fwd, phi)), angle
        if angle % 90 == 0:
            assert torch.allclose(back, blob, atol=1e-5), angle
        else:
            assert (fwd - blob).abs().max() > 0.3, angle          # the forward rotation really moved the blob
            assert (back - blob).abs().max() < 0.1, angle         # and negative=True brought it back


def _system_matrix(cache):
    dx = 0.5
    object_meta = SPECTObjectMeta([dx] * 3, (16, 16, 4))
    angles = list(range(0, 360, 30))
    proj_meta = SPECTProjMeta((16, 4), [dx, dx], angles, radii=[10.0] * len(angles))
    gen = torch.Generator().manual_seed(2)
    mu = (0.15 * torch.rand(16, 16, 4, generator=gen)).to(DEV)
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
