"""The PET sinogram projector generates crystal coordinates on the device per chunk instead of materialising the
coordinates of the whole sinogram on the CPU, and the single scatter simulation keeps its sparse estimate as a table of
sampled bins instead of a dense sinogram. These tests check that both give the same numbers as the dense/host versions.
The projector library (parallelproj) is not needed by any of these tests; it is stubbed if it is not installed."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

try:
    import parallelproj  # noqa: F401
except ImportError:                                   # the tests below never call the projector kernels
    _pp = types.ModuleType("parallelproj")
    _pp.cuda_present = False
    sys.modules["parallelproj"] = _pp

import pytomography
from pytomography.io.PET import shared
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETSinogramPolygonProjMeta, PETTOFMeta
from pytomography.projectors.PET import PETSinogramSystemMatrix
from pytomography.utils import sss

DEV = pytomography.device
INFO = dict(min_rsector_difference=0, crystal_length=20.0, radius=120.0, firstCrystalAxis=0,
            rsectorTransNr=16, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=2, moduleTransSpacing=0.0, moduleAxialSpacing=17.0,
            submoduleTransNr=1, submoduleAxialNr=1, submoduleTransSpacing=0.0, submoduleAxialSpacing=0.0,
            crystalTransNr=4, crystalAxialNr=4, crystalTransSpacing=4.0, crystalAxialSpacing=4.0,
            NrCrystalsPerRing=64, NrRings=8)


def _system_matrix(tof, n_splits=3):
    tof_meta = PETTOFMeta(5, 300.0, 60.0) if tof else None
    proj_meta = PETSinogramPolygonProjMeta(INFO, tof_meta=tof_meta)
    sm = PETSinogramSystemMatrix(ObjectMeta(dr=(4, 4, 4), shape=(32, 32, 16)), proj_meta, N_splits=n_splits, device='cpu')
    sm.set_n_subsets(4)
    return sm


@pytest.mark.parametrize("subset_idx", [None, 1])
def test_device_chunk_coordinates_match_host_coordinates(subset_idx):
    sm = _system_matrix(False)
    xyz1_host, xyz2_host = sm._get_xyz_sinogram_coordinates(subset_idx)   # previous implementation: whole sinogram on the CPU
    N = sm._N_sinogram(subset_idx)
    assert N == xyz1_host.shape[0]
    bounds = sm._chunk_bounds(N)
    assert [b[1] - b[0] for b in bounds] == [len(t) for t in torch.tensor_split(torch.arange(N), sm.N_splits)]
    for start, end in bounds:
        xyz1, xyz2 = sm._xyz_chunk(start, end, subset_idx)
        assert torch.equal(xyz1.cpu(), xyz1_host[start:end]) and torch.equal(xyz2.cpu(), xyz2_host[start:end])


def test_geometry_tables_are_memoised():
    a = shared.sinogram_coordinates(INFO)
    b = shared.sinogram_coordinates(dict(INFO))                      # equal content, different dict object
    assert a[0] is b[0] and a[1] is b[1]
    assert shared.sinogram_to_spatial(INFO)[1] is shared.sinogram_to_spatial(INFO)[1]
    other = dict(INFO, radius=121.0)
    assert shared.sinogram_to_spatial(other)[0] is not shared.sinogram_to_spatial(INFO)[0]


@pytest.mark.parametrize("tof", [False, True])
def test_sparse_sinogram_matches_dense_binning(tof):
    gen = torch.Generator().manual_seed(0)
    tof_meta = PETTOFMeta(5, 300.0, 60.0) if tof else None
    _, _, detector_ids = sss.get_sample_detector_ids(PETSinogramPolygonProjMeta(INFO, tof_meta), 4, 4)
    N = detector_ids.shape[0]
    weights = torch.rand((5, N) if tof else (N,), generator=gen).to(DEV)
    sparse = sss.SparseSinogram(detector_ids, weights, INFO, tof_meta=tof_meta)
    if tof:
        ids = torch.cat([detector_ids.repeat(5, 1), torch.arange(5).repeat_interleave(N).unsqueeze(1)], dim=1)
        dense = shared.listmode_to_sinogram(ids, INFO, tof_meta=tof_meta, weights=weights.flatten().cpu())
    else:
        dense = shared.listmode_to_sinogram(detector_ids, INFO, weights=weights.cpu())
    assert torch.equal(sparse.to_dense(), dense)
    # gather at every bin (sampled or not) reproduces the dense sinogram
    theta, r, plane = torch.meshgrid(torch.arange(dense.shape[0]), torch.arange(dense.shape[1]), torch.arange(dense.shape[2]), indexing="ij")
    for b in (range(5) if tof else [None]):
        expected = dense if b is None else dense[..., b]
        assert torch.equal(sparse.gather(theta, r, plane, tof_bin=b).cpu(), expected)


@pytest.mark.parametrize("tof", [False, True])
def test_interpolation_from_sparse_matches_interpolation_from_dense(tof):
    gen = torch.Generator().manual_seed(1)
    tof_meta = PETTOFMeta(5, 300.0, 60.0) if tof else None
    proj_meta = PETSinogramPolygonProjMeta(INFO, tof_meta)
    idx_intraring, idx_ring, detector_ids = sss.get_sample_detector_ids(proj_meta, 4, 4)
    N = detector_ids.shape[0]
    weights = torch.rand((5, N) if tof else (N,), generator=gen).to(DEV)
    sparse = sss.SparseSinogram(detector_ids, weights, INFO, tof_meta=tof_meta)
    dense = sparse.to_dense()
    if not tof:
        assert torch.equal(sss.interpolate_sparse_sinogram(sparse, proj_meta, idx_intraring, idx_ring),
                           sss.interpolate_sparse_sinogram(dense, proj_meta, idx_intraring, idx_ring))
    else:
        joint = sss.interpolate_sparse_sinogram(sparse, proj_meta, idx_intraring, idx_ring, tof_bins=range(5))
        assert joint.shape == dense.shape
        for b in range(5):     # one solve for all bins instead of one per bin: equal to float32 solver precision (measured 2e-5 of the maximum)
            per_bin = sss.interpolate_sparse_sinogram(dense[..., b], proj_meta, idx_intraring, idx_ring)
            assert (joint[..., b] - per_bin).abs().max() < 1e-4 * per_bin.abs().max()


def test_interpolation_chunk_size_does_not_change_result():
    gen = torch.Generator().manual_seed(2)
    proj_meta = PETSinogramPolygonProjMeta(INFO)
    idx_intraring, idx_ring, detector_ids = sss.get_sample_detector_ids(proj_meta, 4, 4)
    sparse = sss.SparseSinogram(detector_ids, torch.rand(detector_ids.shape[0], generator=gen).to(DEV), INFO)
    a = sss.interpolate_sparse_sinogram(sparse, proj_meta, idx_intraring, idx_ring, eval_chunk_size=100)
    b = sss.interpolate_sparse_sinogram(sparse, proj_meta, idx_intraring, idx_ring, eval_chunk_size=10 ** 9)
    assert (a - b).abs().max() < 1e-4 * b.abs().max()      # float32 kernel-matrix products differ per chunk shape (measured 1e-5 of the maximum)
