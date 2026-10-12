"""parallelproj 2 clips each ray at the faces of the image, while parallelproj 1 integrated out to one voxel beyond the
outermost voxel centres, where the linearly interpolated image falls to zero. The PET system matrices therefore project
the object with one voxel of zeros on every face. These tests check that a ray through an image of ones then integrates
to its chord through the image cube, as with parallelproj 1, and that forward and back projection stay adjoint."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pytomography

parallelproj_core = pytest.importorskip("parallelproj_core", reason="PET projection requires parallelproj 2")
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETSinogramPolygonProjMeta, PETTOFMeta
from pytomography.projectors.PET import PETLMSystemMatrix, PETSinogramSystemMatrix

DEV = pytomography.device
INFO = dict(min_rsector_difference=0, crystal_length=20.0, radius=120.0, firstCrystalAxis=0,
            rsectorTransNr=16, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=2, moduleTransSpacing=0.0,
            moduleAxialSpacing=17.0, submoduleTransNr=1, submoduleAxialNr=1, submoduleTransSpacing=0.0,
            submoduleAxialSpacing=0.0, crystalTransNr=4, crystalAxialNr=4, crystalTransSpacing=4.0,
            crystalAxialSpacing=4.0, NrCrystalsPerRing=64, NrRings=8)
N_DETECTORS = INFO['NrCrystalsPerRing'] * INFO['NrRings']
OBJECT_META = ObjectMeta(dr=(4, 4, 4), shape=(32, 32, 6))     # 128 x 128 x 24 mm: rays leave through every face
# 128 x 128 x 64 mm, axially longer than the scanner (+-14.5 mm): rays leave only through the transaxial faces, which they
# cross at an angle. (A ray that runs almost parallel to a face, just inside it, sees the linearly interpolated image
# fall off towards the face, with or without padding, and is not compared with a chord through a box.)
CHORD_META = ObjectMeta(dr=(4, 4, 4), shape=(32, 32, 16))


def _events(n, tof_meta=None, seed=0):
    gen = torch.Generator().manual_seed(seed)
    d0 = torch.randint(0, N_DETECTORS, (n,), generator=gen)
    d1 = (d0 + torch.randint(1, N_DETECTORS, (n,), generator=gen)) % N_DETECTORS
    columns = [d0, d1] + ([torch.randint(0, tof_meta.num_bins, (n,), generator=gen)] if tof_meta else [])
    return torch.stack(columns, dim=1)


def _chord_lengths(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """Exact length of each segment ([N, 3] float64 endpoints) inside the image cube of ``CHORD_META``."""
    half = torch.tensor(CHORD_META.shape, dtype=torch.float64) * torch.tensor(CHORD_META.dr, dtype=torch.float64) / 2
    direction = end - start
    t0, t1 = torch.zeros(start.shape[0], dtype=torch.float64), torch.ones(start.shape[0], dtype=torch.float64)
    for k in range(3):
        step = torch.where(direction[:, k] == 0, torch.full_like(direction[:, k], 1e-300), direction[:, k])
        a, b = (-half[k] - start[:, k]) / step, (half[k] - start[:, k]) / step
        t0, t1 = torch.maximum(t0, torch.minimum(a, b)), torch.minimum(t1, torch.maximum(a, b))
    return (t1 - t0).clamp_min(0) * direction.norm(dim=1)


def test_list_mode_rays_through_an_image_of_ones_are_chord_lengths():
    """Without the padding: mean error 0.43 mm and 16% of the rays more than 1 mm short; with it: 0.05 mm and none."""
    events = _events(20000)
    system_matrix = PETLMSystemMatrix(CHORD_META, PETLMProjMeta(events, INFO), N_splits=2)
    projection = system_matrix.forward(torch.ones(CHORD_META.shape, device=DEV)).double().cpu()
    lut = system_matrix.proj_meta.scanner_lut.double().cpu()
    error = projection - _chord_lengths(lut[events[:, 0]], lut[events[:, 1]])
    assert error.abs().mean() < 0.15
    assert (error.abs() > 1).double().mean() < 0.01


def test_sinogram_rays_through_an_image_of_ones_are_chord_lengths():
    system_matrix = PETSinogramSystemMatrix(CHORD_META, PETSinogramPolygonProjMeta(INFO), N_splits=2, device='cpu')
    projection = system_matrix.forward(torch.ones(CHORD_META.shape, device=DEV)).double().flatten()
    xyz1, xyz2 = system_matrix._xyz_chunk(0, projection.shape[0])
    error = projection - _chord_lengths(xyz1.double().cpu(), xyz2.double().cpu())
    assert error.abs().mean() < 0.15
    assert (error.abs() > 1).double().mean() < 0.01


@pytest.mark.parametrize("tof", [False, True])
@pytest.mark.parametrize("kind", ["listmode", "sinogram"])
def test_forward_and_back_projection_are_adjoint(kind, tof):
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3) if tof else None
    gen = torch.Generator().manual_seed(1)
    x = torch.rand(OBJECT_META.shape, generator=gen, dtype=torch.float64).float().to(DEV)
    if kind == "listmode":
        system_matrix = PETLMSystemMatrix(OBJECT_META, PETLMProjMeta(_events(5000, tof_meta), INFO, tof_meta=tof_meta), N_splits=2)
    else:
        system_matrix = PETSinogramSystemMatrix(OBJECT_META, PETSinogramPolygonProjMeta(INFO, tof_meta=tof_meta), N_splits=2, device='cpu')
    Hx = system_matrix.forward(x)
    y = torch.rand(Hx.shape, generator=gen, dtype=torch.float64).float().to(Hx.device)
    HTy = system_matrix.backward(y)
    lhs, rhs = (Hx.double() * y.double()).sum(), (x.double() * HTy.double().to(x.device)).sum()
    assert torch.isclose(lhs, rhs.to(lhs.device), rtol=1e-4)
