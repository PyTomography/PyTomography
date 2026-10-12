"""Filtered back projection of parallel-hole SPECT (SPECTSystemMatrix._fbp, issue #227): a phantom is reconstructed
from projections made by the system matrix itself, from scans over 360 and over 180 degrees; each view is weighted by
the angle it stands for; the transforms of the system matrix are not used."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pytomography
from pytomography.algorithms import FilteredBackProjection
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.projectors.SPECT.dualhead_system_matrix import _view_weights
from pytomography.transforms import Transform

N, NZ, DX = 64, 4, 0.4                                          # voxels across, slices, voxel size (cm)
RODS = [((3.0, 0.0), 1.5, 3.0), ((-3.0, 2.5), 1.5, -1.0)]        # centre (cm), radius (cm), activity added


def _grid():
    x = (np.arange(N) - (N - 1) / 2) * DX
    return np.meshgrid(x, x, indexing='ij')


def _phantom():
    """A cylinder of radius 8 cm with activity 1 per voxel, a hot rod (4) and a cold rod (0)."""
    X, Y = _grid()
    f = 1.0 * (np.hypot(X, Y) < 8.0)
    for (cx, cy), radius, df in RODS:
        f = f + df * (np.hypot(X - cx, Y - cy) < radius)
    return torch.tensor(np.repeat(f[:, :, None], NZ, axis=2), dtype=torch.float32).to(pytomography.device)


def _system_matrix(angles, obj2obj_transforms=()):
    object_meta = SPECTObjectMeta([DX] * 3, (N, N, NZ))
    proj_meta = SPECTProjMeta((N, NZ), [DX, DX], list(angles))
    return SPECTSystemMatrix(list(obj2obj_transforms), [], object_meta, proj_meta)


class _Zero(Transform):
    """An object transform that removes everything."""
    def forward(self, object_i, ang_idx):
        return 0 * object_i

    def backward(self, object_i, ang_idx):
        return 0 * object_i


@pytest.mark.parametrize("arc", [360, 180])
def test_spect_fbp_reconstructs_the_phantom(arc):
    sm = _system_matrix(np.arange(0, arc, 3.0))
    image = FilteredBackProjection(sm.forward(_phantom()), sm, filter='hann')()
    assert image.shape == (N, N, NZ) and image.device.type == torch.device(pytomography.device).type
    image = image.cpu().numpy()
    X, Y = _grid()
    background = np.hypot(X, Y) < 6.5
    for (cx, cy), radius, df in RODS:
        background &= np.hypot(X - cx, Y - cy) > radius + 1.2
    err = image[background] - 1
    assert abs(err.mean()) < 0.005 and np.abs(err).max() < 0.05
    for (cx, cy), radius, df in RODS:                    # each rod's activity, in its core
        core = np.hypot(X - cx, Y - cy) < radius - 0.6
        assert image[core].mean() == pytest.approx(1 + df, abs=0.08)


def test_views_are_weighted_by_the_angle_they_stand_for():
    assert np.allclose(_view_weights(np.arange(120) * 3.0), np.pi / 120)          # 360 degrees
    assert np.allclose(_view_weights(np.arange(60) * 3.0), np.pi / 60)            # 180 degrees
    heads = np.concatenate([np.arange(60) * 3.0, 181.5 + np.arange(60) * 3.0])     # two heads, interleaved
    assert np.allclose(_view_weights(heads), np.pi / 120)
    angles = np.arange(90) * 3.0                                                   # 270 degrees: lines seen twice share
    w = np.array(_view_weights(angles))
    assert np.allclose(w[angles % 180 < 90], np.deg2rad(1.5)) and np.allclose(w[angles % 180 >= 90], np.deg2rad(3.0))
    with pytest.warns(UserWarning, match='less than 180 degrees'):
        w = _view_weights(np.arange(30) * 3.0)                                     # 90 degrees
    assert np.allclose(w, np.deg2rad(3.0))


def test_spect_fbp_does_not_use_the_transforms():
    angles = np.arange(0, 360, 6.0)
    plain = _system_matrix(angles)
    proj = plain.forward(_phantom())
    image = FilteredBackProjection(proj, plain)()
    assert image.abs().max() > 0
    assert torch.equal(FilteredBackProjection(proj, _system_matrix(angles, [_Zero()]))(), image)


def test_spect_fbp_checks_the_shape_of_the_projections():
    with pytest.raises(ValueError, match='do not match'):
        FilteredBackProjection(torch.zeros(10, N, NZ), _system_matrix(np.arange(0, 360, 6.0)))()
