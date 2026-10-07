"""The CT system matrices call parallelproj 2, whose kernels write the forward projection into a buffer they are given
and add the back projection into the image they are given. These tests check, on small synthetic scanners, that the
back projection is the adjoint of the forward projection, that splitting the computation or projecting a subset
does not change what is computed, that projections land on the requested device, that the projection of a Gaussian
matches its analytic line integrals, and that the projection of an object does not depend on how much empty space
surrounds it."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pytomography

parallelproj_core = pytest.importorskip("parallelproj_core", reason="CT projection requires parallelproj 2")
from pytomography.algorithms import SART
from pytomography.metadata import ObjectMeta
from pytomography.metadata.CT import CTConeBeamFlatPanelProjMeta, CTGen3ProjMeta
from pytomography.projectors.CT import CTConeBeamFlatPanelSystemMatrix, CTGen3SystemMatrix

DEV = pytomography.device
OBJECT_META = ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(32, 32, 12))


def _gen3_meta(n_views=48, n_cols=48, n_rows=8):
    """Two helical rotations onto a cylindrical detector, with an alternating in-plane flying focal spot, laid out
    the way ``dicom_ct_pd`` reads a DICOM-CT-PD scan."""
    phis = torch.linspace(0, 4 * np.pi, n_views + 1)[:-1]
    zero = torch.zeros(n_views)
    DSD = 270.0
    return CTGen3ProjMeta(
        source_phis=phis, source_rhos=torch.full((n_views,), 150.0), source_zs=torch.linspace(-10, 10, n_views),
        source_phi_offsets=1e-3 * (-1.0) ** torch.arange(n_views), source_rho_offsets=zero.clone(),
        source_z_offsets=zero.clone(),
        detector_centers_col_idx=torch.full((n_views,), (n_cols + 1) / 2),
        detector_centers_row_idx=torch.full((n_views,), (n_rows + 1) / 2),
        col_det_spacing=float(np.arcsin(5.0 / DSD)), row_det_spacing=2.0, DSD=DSD, shape=(n_cols, n_rows))


def _conebeam_meta(n_views=24):
    """Circular cone beam scan onto a flat panel."""
    angles = torch.linspace(0, 2 * np.pi, n_views + 1)[:-1]
    return CTConeBeamFlatPanelProjMeta(angles, torch.zeros(n_views), detector_radius=100.0, beam_radius=150.0,
                                       shape=(40, 16), dr=(2.5, 2.5))


def _gen3(N_splits=1, device=DEV, object_meta=OBJECT_META):
    return CTGen3SystemMatrix(object_meta, _gen3_meta(), N_splits=N_splits, device=device)


def _conebeam(N_splits=1, device=DEV, object_meta=OBJECT_META):
    return CTConeBeamFlatPanelSystemMatrix(object_meta, _conebeam_meta(), N_splits=N_splits, device=device)


SYSTEMS = pytest.mark.parametrize("make", [_gen3, _conebeam], ids=["gen3", "conebeam"])


def _object(shape=OBJECT_META.shape, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=gen).to(DEV)


def _like(proj, seed=2):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(proj.shape, generator=gen).to(proj.device)


def _rays(sm):
    """Start (focal spot) and end (detector element) of every ray, in the order of the projections."""
    if isinstance(sm, CTGen3SystemMatrix):
        return sm._ray_coordinates(torch.arange(sm.proj_meta.N_angles))
    xend = torch.stack([sm.proj_meta._get_detector_coordinates(i) for i in range(sm.proj_meta.N_angles)])
    return sm.proj_meta.beam_locations[:, None, None].expand(xend.shape), xend


@SYSTEMS
@pytest.mark.parametrize("subset_idx", [None, 1])
def test_back_projection_is_the_adjoint(make, subset_idx):
    """<Hf, g> = <f, H^T g>. The projector accumulates in float32, so the two agree to rounding, not exactly."""
    sm = make()
    sm.set_n_subsets(3)
    f = _object()
    Hf = sm.forward(f, subset_idx)
    assert (Hf > 0).float().mean() > 0.5, "most rays should cross the object, or the test proves little"
    g = _like(Hf)
    HTg = sm.backward(g, subset_idx)
    lhs = (Hf.double() * g.double()).sum()
    rhs = (f.double() * HTg.double().to(f.device)).sum()
    assert abs(lhs - rhs) <= 1e-5 * abs(lhs)


@SYSTEMS
def test_splits_and_subsets_do_not_change_the_projection(make):
    """Splitting is a memory trade, not a different computation: forward projections are identical, and back
    projections differ only by the order of the projector's atomic additions. Projecting a subset gives the views of
    that subset, and the normalization factors of the subsets add up to that of all views."""
    whole, split = make(N_splits=1), make(N_splits=5)
    for sm in (whole, split):
        sm.set_n_subsets(4)
    f = _object()
    full = whole.forward(f)
    assert torch.equal(split.forward(f), full)
    for subset_idx in range(4):
        assert torch.equal(split.forward(f, subset_idx), full[whole.subset_indices_array[subset_idx].to(full.device)])
    g = _like(full)
    a, b = whole.backward(g), split.backward(g)
    assert (a - b).abs().max() <= 1e-5 * a.abs().max()
    norm = whole.compute_normalization_factor().to(DEV)
    norm_subsets = sum(whole.compute_normalization_factor(k).to(DEV) for k in range(4))
    assert (norm - norm_subsets).abs().max() <= 1e-5 * norm.abs().max()


def test_gen3_projection_is_split_to_bound_the_rays_per_call(monkeypatch):
    """The gen3 system matrix builds the ray coordinates on the GPU, so it splits a projection into groups of views of
    at most ``_MAX_RAYS_PER_CALL`` rays whatever ``N_splits`` is; like N_splits, that does not change the result."""
    import pytomography.projectors.CT.ct_gen3_system_matrix as gen3
    sm = _gen3()
    f = _object()
    full = sm.forward(f)
    g = _like(full)
    BP = sm.backward(g)
    rays_per_view = sm.proj_meta.shape[0] * sm.proj_meta.shape[1]
    monkeypatch.setattr(gen3, "_MAX_RAYS_PER_CALL", 5 * rays_per_view)
    splits = sm._splits(torch.arange(sm.proj_meta.N_angles))
    assert len(splits) == -(-sm.proj_meta.N_angles // 5)
    assert all(end - start <= 5 for start, end, _ in splits)
    assert torch.equal(sm.forward(f), full)
    assert (sm.backward(g) - BP).abs().max() <= 1e-5 * BP.abs().max()


@SYSTEMS
@pytest.mark.parametrize("device", ["cpu", DEV], ids=["cpu", "default"])
def test_reconstruction_with_projections_on_either_device(make, device):
    """``device`` says where the projections are kept. Both choices must give the same reconstruction; the default,
    ``pytomography.device``, used to fail for the gen3 scanner by indexing CPU geometry with GPU subset indices."""
    reference = make(device=DEV)
    measured = reference.forward(_object())
    sm = make(device=device)
    proj = sm.forward(_object())
    assert proj.device.type == torch.device(device).type
    assert torch.equal(proj.to(DEV), measured)
    recon = SART(sm, proj)(n_iters=1, n_subsets=3)
    expected = SART(reference, measured)(n_iters=1, n_subsets=3)
    assert torch.isfinite(recon).all()
    assert (recon - expected).abs().max() <= 1e-4 * expected.abs().max()


@SYSTEMS
def test_forward_projection_of_a_gaussian_matches_its_line_integrals(make):
    """Ties the geometry conventions (image origin, voxel size, axis order) to something analytic: the line integral
    of exp(-|r-c|^2 / 2 sigma^2) along a ray is sigma sqrt(2 pi) exp(-d^2 / 2 sigma^2), with d the distance from c to
    the ray. Interpolating the sampled Gaussian blurs it slightly, to 2.7% relative RMS error here; an origin half a
    voxel out gives 13 to 18%, and a flipped or swapped axis over 100%."""
    object_meta = ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(32, 32, 20))
    sm = make(object_meta=object_meta)
    center, sigma = torch.tensor([9.0, -5.0, 2.0], dtype=torch.float64, device=DEV), 4.0
    axes = [(torch.arange(n, device=DEV) - (n - 1) / 2) * d for n, d in zip(object_meta.shape, object_meta.dr)]
    grid = torch.meshgrid(*axes, indexing="ij")
    gaussian = torch.exp(-sum((grid[i] - center[i]) ** 2 for i in range(3)) / (2 * sigma ** 2))
    proj = sm.forward(gaussian).reshape(-1).double().to(DEV)
    xstart, xend = (x.reshape(-1, 3).double().to(DEV) for x in _rays(sm))
    direction = (xend - xstart) / (xend - xstart).norm(dim=1, keepdim=True)
    to_center = center - xstart
    distance_sq = (to_center ** 2).sum(dim=1) - (to_center * direction).sum(dim=1) ** 2
    expected = sigma * np.sqrt(2 * np.pi) * torch.exp(-distance_sq / (2 * sigma ** 2))
    assert (proj - expected).pow(2).mean().sqrt() <= 0.06 * expected.pow(2).mean().sqrt()


@SYSTEMS
def test_projection_does_not_depend_on_the_empty_space_around_the_object(make):
    """parallelproj 2 clips each ray to the faces of the image it is given and drops the interpolation outside the
    outermost voxel centres, which parallelproj 1 included. Unless the system matrix accounts for it (by projecting
    the object with a voxel of zeros on every face), an object that is not zero at the faces, such as a patient who
    extends past the reconstructed slices, projects differently depending on how much empty space surrounds it: the
    rays leaving through such a face come out short by up to several millimetres."""
    shape = OBJECT_META.shape
    larger = ObjectMeta(dr=OBJECT_META.dr, shape=tuple(n + 4 for n in shape))
    small, large = make(), make(object_meta=larger)
    f = _object()
    proj = small.forward(f)
    proj_embedded = large.forward(torch.nn.functional.pad(f, (2, 2, 2, 2, 2, 2)))
    assert (proj_embedded - proj).abs().max() <= 1e-5 * proj.abs().max()
    g = _like(proj)
    BP, BP_embedded = small.backward(g), large.backward(g)
    assert (BP_embedded[2:-2, 2:-2, 2:-2] - BP).abs().max() <= 1e-5 * BP.abs().max()
