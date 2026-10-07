"""The single scatter simulation computes its emission and transmission integrals with parallelproj 2. Unlike the
system matrices, it integrates along segments that start, and for time of flight also end, inside the image: from a
scatter point to each detector, and the time of flight estimate cuts that line into pieces and adds up their weighted
integrals. These tests check that a segment is integrated over its own length, that the pieces of a line add up to the
whole line, so the time of flight estimate summed over its bins equals the non time of flight one (parallelproj 1
counted the plane where two pieces meet in both of them, which made the sum 14% too large on the GATE mMR scan), and
that grouping scatter points into one projector call, or the dtype, layout and device of the images, do not change
the estimate."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pytomography

parallelproj_core = pytest.importorskip("parallelproj_core", reason="the scatter simulation requires parallelproj 2")
from pytomography.io.PET import shared
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.projectors.PET.petlm_system_matrix import _float32
from pytomography.utils import sss

DEV = pytomography.device
INFO = dict(min_rsector_difference=0, crystal_length=20.0, radius=120.0, firstCrystalAxis=0,
            rsectorTransNr=16, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=2, moduleTransSpacing=0.0,
            moduleAxialSpacing=17.0, submoduleTransNr=1, submoduleAxialNr=1, submoduleTransSpacing=0.0,
            submoduleAxialSpacing=0.0, crystalTransNr=4, crystalAxialNr=4, crystalTransSpacing=4.0,
            crystalAxialSpacing=4.0, NrCrystalsPerRing=64, NrRings=8)
N_DETECTORS = INFO['NrCrystalsPerRing'] * INFO['NrRings']
OBJECT_META = ObjectMeta(dr=(4, 4, 4), shape=(32, 32, 6))      # within the scanner's axial field of view (+-14.5 mm)
ORIGIN = _float32((- np.array(OBJECT_META.shape) / 2 + 0.5) * np.array(OBJECT_META.dr), DEV)
VOXEL_SIZE = _float32(np.array(OBJECT_META.dr), DEV)
HALF_WIDTH = torch.tensor(OBJECT_META.shape, dtype=torch.float64) * torch.tensor(OBJECT_META.dr, dtype=torch.float64) / 2
STEPS = (2, 0.004, 4, 4)     # image step, attenuation cutoff, inter-ring step, intra-ring step


def _phantom():
    """Water cylinder of radius 40 mm with uniform activity and a hot sphere: (activity, attenuation) on the device."""
    x, y, z = torch.meshgrid(*[(torch.arange(n) - n / 2 + 0.5) * d for n, d in zip(OBJECT_META.shape, OBJECT_META.dr)], indexing="ij")
    cylinder = (x**2 + y**2 < 40**2).float()
    sphere = (((x - 15)**2 + (y + 10)**2 + z**2) < 12**2).float()
    return (cylinder + 3 * sphere).to(DEV), (0.0096 * cylinder).to(DEV)


def _proj_meta(tof_meta=None):
    return PETLMProjMeta(torch.zeros((1, 3 if tof_meta else 2), dtype=torch.long), INFO, tof_meta=tof_meta)


def _detectors(n: int, gen: torch.Generator) -> torch.Tensor:
    return _proj_meta().scanner_lut.to(torch.float64)[torch.randint(0, N_DETECTORS, (n,), generator=gen)]


def _length_inside_image(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """Exact length of the part of each segment ([N, 3] float64 endpoints) that lies inside the image."""
    direction = end - start
    t0, t1 = torch.zeros(start.shape[0], dtype=torch.float64), torch.ones(start.shape[0], dtype=torch.float64)
    for k in range(3):
        step = torch.where(direction[:, k] == 0, torch.full_like(direction[:, k], 1e-300), direction[:, k])
        a, b = (-HALF_WIDTH[k] - start[:, k]) / step, (HALF_WIDTH[k] - start[:, k]) / step
        t0, t1 = torch.maximum(t0, torch.minimum(a, b)), torch.minimum(t1, torch.maximum(a, b))
    return (t1 - t0).clamp_min(0) * direction.norm(dim=1)


def test_segments_are_integrated_over_their_own_length():
    """Through an image of ones a segment's integral is its length, counted in the planes of voxel centres it crosses:
    within a plane spacing of it, and right on average. parallelproj 1 added about a plane at each end, so segments of
    one to five voxels came out 46% too long on average (and segments from inside the image to a detector 2.8%)."""
    gen = torch.Generator().manual_seed(0)
    ones = torch.ones(OBJECT_META.shape, device=DEV)
    inner = HALF_WIDTH - torch.tensor(OBJECT_META.dr, dtype=torch.float64)     # a voxel in: halfway between two planes
    start = (torch.rand(40000, 3, generator=gen, dtype=torch.float64) * 2 - 1) * inner
    direction = torch.randn(40000, 3, generator=gen, dtype=torch.float64)
    end = start + direction / direction.norm(dim=1, keepdim=True) * (4 + 16 * torch.rand(40000, 1, generator=gen, dtype=torch.float64))
    inside = (end.abs() < inner).all(dim=1)                                    # both ends inside
    start, end = start[inside], end[inside]
    integral = sss._line_integrals(start, end, ones, ORIGIN, VOXEL_SIZE).cpu().double()
    length = (end - start).norm(dim=1)
    plane_spacing = max(OBJECT_META.dr) * length / (end - start).abs().max(dim=1).values
    assert start.shape[0] > 5000
    assert abs(integral.sum() / length.sum() - 1) < 0.01
    assert ((integral - length).abs() < 1.001 * plane_spacing).all()
    # From inside the image to a detector the segment also leaves the image, where parallelproj 2 is less exact than
    # parallelproj 1 was (rays that clip the edge of the image), but there is still no bias from the start
    start = (torch.rand(20000, 3, generator=gen, dtype=torch.float64) * 2 - 1) * inner
    end = _detectors(20000, gen)
    integral = sss._line_integrals(start, end, ones, ORIGIN, VOXEL_SIZE).cpu().double()
    assert abs(integral.sum() / _length_inside_image(start, end).sum() - 1) < 0.02


def test_pieces_of_a_line_add_up_to_the_line():
    """The time of flight estimate cuts each scatter point to detector line into pieces, as here."""
    gen = torch.Generator().manual_seed(1)
    activity, _ = _phantom()
    image = activity + torch.rand(OBJECT_META.shape, generator=gen).to(DEV)
    start = (torch.rand(500, 3, generator=gen, dtype=torch.float64) * 2 - 1) * HALF_WIDTH * 0.9
    end = _detectors(500, gen)
    edges = start.unsqueeze(1) + torch.linspace(0, 1, 26, dtype=torch.float64).reshape(1, -1, 1) * (end - start).unsqueeze(1)
    pieces = sss._line_integrals(edges[:, :-1].reshape(-1, 3), edges[:, 1:].reshape(-1, 3), image, ORIGIN, VOXEL_SIZE).reshape(500, 25)
    whole = sss._line_integrals(start, end, image, ORIGIN, VOXEL_SIZE)
    assert whole.min() > 0
    assert torch.allclose(pieces.sum(dim=1), whole, rtol=1e-4, atol=0)


def test_points_per_projection_does_not_change_the_estimate(monkeypatch):
    """Grouping scatter points into one projector call only regroups the same rays."""
    activity, attenuation = _phantom()
    estimates = []
    for points_per_projection in (1, 7, 10 ** 6):    # one point per call, a partial last group, all points in one call
        monkeypatch.setattr(sss, "_POINTS_PER_PROJECTION", points_per_projection)
        torch.manual_seed(0)
        estimates.append(sss.compute_sss_sparse_sinogram(OBJECT_META, _proj_meta(), activity, attenuation, *STEPS).weights)
    assert estimates[0].abs().sum() > 0
    assert all(torch.equal(e, estimates[0]) for e in estimates[1:])


@pytest.mark.parametrize("n_splits", [1, 2])
def test_tof_estimate_summed_over_bins_equals_non_tof_estimate(n_splits):
    """The time of flight probabilities of each piece are normalised over the bins, so summing the time of flight
    estimate over its bins must give the non time of flight estimate for the same scatter points (parallelproj 1 gave
    a sum 76% larger on this phantom; with 2 splits of the TOF bins, the estimate used to be twice as large)."""
    activity, attenuation = _phantom()
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3)
    torch.manual_seed(0)
    non_tof = sss.compute_sss_sparse_sinogram(OBJECT_META, _proj_meta(tof_meta), activity, attenuation, *STEPS)
    torch.manual_seed(0)
    tof = sss.compute_sss_sparse_sinogram_TOF(OBJECT_META, _proj_meta(tof_meta), activity, attenuation, tof_meta, *STEPS, 25, n_splits)
    assert torch.equal(tof.keys, non_tof.keys) and tof.weights.shape == (5, non_tof.weights.shape[0])
    assert torch.allclose(tof.weights.sum(dim=0), non_tof.weights, rtol=1e-3, atol=1e-6 * non_tof.weights.abs().max().item())


def test_tof_estimate_does_not_depend_on_how_the_bins_are_split():
    """N_splits only bounds memory. The TOF kernel used to be normalised over the bins of each split, which multiplied
    the estimate by the number of splits and gave every split the whole emission."""
    activity, attenuation = _phantom()
    tof_meta = PETTOFMeta(6, 300.0, 60.0, n_sigmas=3)
    estimates = []
    for n_splits in (1, 2, 4, 6):            # 4 splits are uneven (2, 2, 1, 1 bins)
        torch.manual_seed(0)
        estimates.append(sss.compute_sss_sparse_sinogram_TOF(OBJECT_META, _proj_meta(tof_meta), activity, attenuation, tof_meta, *STEPS, 25, n_splits).weights)
    for estimate in estimates[1:]:
        assert torch.allclose(estimate, estimates[0], rtol=1e-5, atol=1e-6 * estimates[0].abs().max().item())


def test_tof_weighted_emission_matches_normalising_the_kernel():
    """The emission seen by each TOF bin is computed by normalising the emission of each piece instead of the kernel,
    with a batched product for the sum over pieces; it must equal the direct formula."""
    gen = torch.Generator().manual_seed(5)
    n_bins, n_lors, n_pieces, sigma = 7, 3000, 25, 35.0
    offset = ((torch.rand(n_bins, n_lors, generator=gen) - 0.5) * 300).to(DEV)
    # pieces within 200 mm: farther out every bin's kernel can underflow, where the direct formula gives 0/0
    centers = (torch.rand(n_lors, 1, generator=gen) * 200 * (torch.arange(n_pieces) + 0.5) / n_pieces).to(DEV)
    emission = torch.rand(n_lors, n_pieces, generator=gen).to(DEV)
    direct = (sss._tof_efficiency(offset, centers, sigma) * emission.unsqueeze(0)).sum(dim=-1)
    for splits in ([(0, 7)], [(0, 3), (3, 7)]):
        computed = sss._tof_weighted_emission(offset, centers, emission, sigma, splits)
        assert computed.shape == (n_bins, n_lors)
        assert torch.allclose(computed, direct, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("tof", [False, True])
def test_images_of_any_dtype_layout_and_device(tof):
    """parallelproj 2 takes contiguous float32 arrays on one device; the images are converted, whatever they are given as."""
    activity, attenuation = _phantom()
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3) if tof else None
    def estimate(act, att):
        torch.manual_seed(0)
        if tof:
            return sss.compute_sss_sparse_sinogram_TOF(OBJECT_META, _proj_meta(tof_meta), act, att, tof_meta, *STEPS).weights
        return sss.compute_sss_sparse_sinogram(OBJECT_META, _proj_meta(), act, att, *STEPS).weights
    def float64_strided_cpu(x):
        x = x.double().cpu().permute(2, 1, 0).contiguous().permute(2, 1, 0)
        assert not x.is_contiguous()
        return x
    assert torch.equal(estimate(activity, attenuation), estimate(float64_strided_cpu(activity), float64_strided_cpu(attenuation)))


@pytest.mark.parametrize("tof", [False, True])
def test_list_mode_scatter_estimate(tof):
    """The scatter estimate cell of the list mode tutorials: the estimate (sparse estimate, interpolation, scaling with a
    sinogram system matrix) and its conversion back to the events, whose detector IDs the system matrix keeps on its
    lor_device."""
    gen = torch.Generator().manual_seed(2)
    activity, attenuation = _phantom()
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3) if tof else None
    d0 = torch.randint(0, N_DETECTORS, (20000,), generator=gen)
    d1 = (d0 + N_DETECTORS // 2 + torch.randint(-20, 20, (20000,), generator=gen)) % N_DETECTORS
    columns = [d0, d1] + ([torch.randint(0, tof_meta.num_bins, (20000,), generator=gen)] if tof else [])
    proj_meta = PETLMProjMeta(torch.stack(columns, dim=1), INFO, tof_meta=tof_meta)
    system_matrix = PETLMSystemMatrix(OBJECT_META, proj_meta, attenuation_map=attenuation, N_splits=2)
    torch.manual_seed(0)
    scatter = sss.get_sss_scatter_estimate(OBJECT_META, proj_meta, activity, attenuation, system_matrix,
                                           image_stepsize=2, sinogram_interring_stepsize=4, sinogram_intraring_stepsize=4,
                                           tof_meta=tof_meta)
    n_planes = (INFO['moduleAxialNr'] * INFO['crystalAxialNr'])**2
    assert scatter.shape == (INFO['NrCrystalsPerRing'] // 2, INFO['NrCrystalsPerRing'] + 1, n_planes) + ((tof_meta.num_bins,) if tof else ())
    assert torch.isfinite(scatter).all() and scatter.sum() > 0
    lm_scatter = shared.sinogram_to_listmode(proj_meta.detector_ids, scatter, proj_meta.info)
    assert lm_scatter.shape == (20000,)
    assert torch.equal(lm_scatter, shared.sinogram_to_listmode(proj_meta.detector_ids.cpu(), scatter, proj_meta.info))
