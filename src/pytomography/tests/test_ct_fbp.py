"""Filtered back projection in the CT system matrices. Helical WFBP (CTGen3SystemMatrix) reconstructs a phantom from its
exact line integrals (computed analytically along each ray, so independent of the projector), with and without a
flying focal spot, and so does the circular FDK (CTConeBeamFlatPanelSystemMatrix) in the central slices; voxels outside
the field of view are zero; the GPU budget bounds the peak. A data test (marked ``data``) compares TCIA
LDCT-and-Projection-data case C145 with the scanner's own images."""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest
import torch

import pytomography

parallelproj_core = pytest.importorskip("parallelproj_core", reason="CT system matrices require parallelproj 2")
from pytomography.algorithms import FilteredBackProjection
from pytomography.io.CT import dicom_ct_pd, preprocessing
from pytomography.metadata import ObjectMeta
from pytomography.metadata.CT import CTConeBeamFlatPanelProjMeta, CTGen3ProjMeta
from pytomography.projectors.CT import CTConeBeamFlatPanelSystemMatrix, CTGen3SystemMatrix
from pytomography.utils import PeakMemory, TabulatedFilter

MU = 0.02                                                      # per mm, about water
CYLINDER = dict(radius=36.0, z0=-14.0, z1=14.0)
SPHERES = [((10.0, 6.0, 0.0), 7.0, 0.004), ((-12.0, -9.0, 5.0), 6.0, -0.006)]
OBJECT_META = ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(48, 48, 14))


def _helix(ffs=False, views_per_rotation=240, rotations=6, n_cols=72, n_rows=12, feed=12.0):
    """A helical scan (focal spot radius 150 mm, 270 mm to a cylindrical detector, pitch 0.9) laid out as dicom_ct_pd
    reads one. With ``ffs``, every other view's focal spot is 0.5 mm lower and 2 mm further out, and every focal spot
    is turned by 3e-4 rad, like the Siemens z flying focal spot."""
    n = views_per_rotation * rotations
    phis = torch.arange(n, dtype=torch.float64) * (2 * np.pi / views_per_rotation)
    zs = feed / (2 * np.pi) * phis
    odd = (torch.arange(n) % 2).double()
    dphi = torch.full((n,), 3e-4 if ffs else 0.0, dtype=torch.float64)
    return CTGen3ProjMeta(phis.float(), torch.full((n,), 150.0), (zs - zs.mean()).float(), dphi.float(),
                          (2.0 * odd if ffs else 0 * odd).float(), (-0.5 * odd if ffs else 0 * odd).float(),
                          torch.full((n,), (n_cols + 1) / 2), torch.full((n,), (n_rows + 1) / 2),
                          float(2 * np.arcsin(50.0 / 150.0) / (n_cols - 1)), 2.0, 270.0, shape=(n_cols, n_rows))


def _chords_sphere(a, u, L, centre, radius):
    oc = a - torch.tensor(centre, dtype=a.dtype)
    b = (oc * u).sum(-1)
    disc = b * b - ((oc * oc).sum(-1) - radius ** 2)
    s = torch.sqrt(torch.clamp(disc, min=0))
    return torch.clamp(torch.minimum(-b + s, L) - torch.clamp(-b - s, min=0), min=0) * (disc > 0)


def _chords_cylinder(a, u, L, radius, z0, z1):
    A = u[..., 0] ** 2 + u[..., 1] ** 2
    B = 2 * (a[..., 0] * u[..., 0] + a[..., 1] * u[..., 1])
    disc = B * B - 4 * A * (a[..., 0] ** 2 + a[..., 1] ** 2 - radius ** 2)
    s = torch.sqrt(torch.clamp(disc, min=0))
    uz = torch.where(u[..., 2].abs() < 1e-12, torch.full_like(u[..., 2], 1e-12), u[..., 2])
    t0, t1 = (z0 - a[..., 2]) / uz, (z1 - a[..., 2]) / uz
    lo = torch.maximum(torch.maximum((-B - s) / (2 * A), torch.minimum(t0, t1)), torch.zeros_like(L))
    hi = torch.minimum(torch.minimum((-B + s) / (2 * A), torch.maximum(t0, t1)), L)
    return torch.clamp(hi - lo, min=0) * (disc > 0)


def _line_integrals(meta, spheres=SPHERES):
    """Exact line integrals of the phantom along every ray, from each view's focal spot to each detector element."""
    end = meta.get_detector_coordinates(torch.arange(meta.N_angles)).double()
    start = meta.source_focal_spots.double()[:, None, None, :].expand_as(end)
    d = end - start
    L = d.norm(dim=-1)
    u = d / L[..., None]
    p = MU * _chords_cylinder(start, u, L, **CYLINDER)
    for centre, radius, dmu in spheres:
        p = p + dmu * _chords_sphere(start, u, L, centre, radius)
    return p.float()


def _truth(object_meta=OBJECT_META):
    (Nx, Ny, Nz), (dx, dy, dz) = object_meta.shape, object_meta.dr
    x, y, z = [(np.arange(n) - (n - 1) / 2) * d for n, d in ((Nx, dx), (Ny, dy), (Nz, dz))]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    r = np.hypot(X, Y)
    mu = np.where((r <= CYLINDER['radius']) & (Z >= CYLINDER['z0']) & (Z <= CYLINDER['z1']), MU, 0.0)
    safe = (r < CYLINDER['radius'] - 6) & (np.abs(Z) <= 8)
    for (cx, cy, cz), radius, dmu in SPHERES:
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
        mu = mu + np.where(dist <= radius, dmu, 0.0)
        safe &= np.abs(dist - radius) > 4
    return mu, safe


@pytest.mark.parametrize("ffs", [False, True], ids=["one focal spot", "flying focal spot"])
def test_helical_fbp_reconstructs_the_phantom_from_its_exact_line_integrals(ffs):
    meta = _helix(ffs)
    sm = CTGen3SystemMatrix(OBJECT_META, meta)
    image = FilteredBackProjection(_line_integrals(meta), sm, filter='hann')().cpu().numpy()
    truth, safe = _truth()
    err = (image - truth)[safe] / MU
    assert abs(err.mean()) < 0.01                       # within 1% of water on average, away from edges
    assert np.abs(err).max() < 0.05
    for (cx, cy, cz), radius, dmu in SPHERES:            # each sphere's contrast, in its core
        (Nx, Ny, Nz), (dx, dy, dz) = OBJECT_META.shape, OBJECT_META.dr
        x, y, z = [(np.arange(n) - (n - 1) / 2) * d for n, d in ((Nx, dx), (Ny, dy), (Nz, dz))]
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        core = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) < radius - 3
        assert image[core].mean() - MU == pytest.approx(dmu, rel=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel runs on CUDA")
@pytest.mark.parametrize("ffs", [False, True], ids=["one focal spot", "flying focal spot"])
def test_the_fused_kernel_matches_the_pytorch_back_projection(ffs):
    from pytomography.projectors.CT import _wfbp_cuda
    if not _wfbp_cuda.available('cuda'):
        pytest.skip('the fused kernel needs CuPy')
    meta = _helix(ffs)
    proj, sm = _line_integrals(meta), CTGen3SystemMatrix(OBJECT_META, meta)
    stats = {}
    fused = FilteredBackProjection(proj, sm, slice_thickness=1.25, backend='cuda', stats=stats)()
    plain = FilteredBackProjection(proj, sm, slice_thickness=1.25, backend='torch')()
    assert stats['backend'] == 'cuda'
    assert float((fused - plain).abs().max()) < 1e-4 * MU


def test_helical_fbp_is_zero_outside_the_field_of_view_and_on_the_device():
    sm = CTGen3SystemMatrix(ObjectMeta(dr=(3.0, 3.0, 2.0), shape=(48, 48, 6)), _helix())
    image = FilteredBackProjection(_line_integrals(sm.proj_meta), sm)()
    assert image.device.type == torch.device(pytomography.device).type
    x = (torch.arange(48) - 23.5) * 3.0
    outside = (x[:, None] ** 2 + x[None, :] ** 2) > sm.fov_radius ** 2
    assert torch.all(image[outside.to(image.device)] == 0)


def _conebeam(n_views=120, n_cols=64):
    """A circular cone-beam scan: focal spot 200 mm from the axis, flat panel 100 mm beyond it (64 x 24, 2 mm)."""
    angles = torch.linspace(0, 2 * np.pi, n_views + 1)[:-1]
    return CTConeBeamFlatPanelProjMeta(angles, torch.zeros(n_views), detector_radius=100.0, beam_radius=200.0,
                                       shape=(n_cols, 24), dr=(2.0, 2.0))


def _conebeam_line_integrals(meta, sphere):
    """Exact line integrals through a water cylinder (radius 30 mm, longer than the cone) and a sphere of contrast."""
    end = torch.stack([meta._get_detector_coordinates(i) for i in range(meta.N_angles)]).double().cpu()
    start = meta.beam_locations.double().cpu()[:, None, None, :].expand_as(end)
    d = end - start
    L = d.norm(dim=-1)
    u = d / L[..., None]
    centre, radius, dmu = sphere
    return (MU * _chords_cylinder(start, u, L, radius=30.0, z0=-40.0, z1=40.0)
            + dmu * _chords_sphere(start, u, L, centre, radius)).float()


@pytest.mark.parametrize("n_cols", [64, 63], ids=["even columns", "odd columns"])
def test_conebeam_fdk_reconstructs_the_phantom_in_the_central_slices(n_cols):
    meta, sphere = _conebeam(n_cols=n_cols), ((8.0, 5.0, 0.0), 8.0, 0.004)
    object_meta = ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(40, 40, 6))
    sm = CTConeBeamFlatPanelSystemMatrix(object_meta, meta)
    proj = _conebeam_line_integrals(meta, sphere).to(pytomography.device)
    image = FilteredBackProjection(proj, sm, filter='hann')()
    x, z = (np.arange(40) - 19.5) * 2.0, (np.arange(6) - 2.5) * 2.0
    X, Y, Z = np.meshgrid(x, x, z, indexing='ij')
    (cx, cy, cz), radius, dmu = sphere
    to_sphere = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    truth = MU * (np.hypot(X, Y) <= 30.0) + dmu * (to_sphere <= radius)
    safe = (np.hypot(X, Y) < 24.0) & (np.abs(to_sphere - radius) > 4)
    err = (image.cpu().numpy() - truth)[safe] / MU
    assert abs(err.mean()) < 0.01 and np.abs(err).max() < 0.05
    assert image.cpu().numpy()[to_sphere < radius - 3].mean() - MU == pytest.approx(dmu, rel=0.15)
    ram_lak = FilteredBackProjection(proj, sm, filter='ram-lak')()
    assert (image - ram_lak).abs().max() > 0.01 * MU                   # the window is applied


def test_the_column_scale_is_fitted_back_from_a_reference_image():
    # the "scanner" applied a per-column scale before reconstructing; the fit finds it from the two images
    meta = _helix(rotations=3)
    meta.water_attenuation = MU
    proj = _line_integrals(meta)
    sm = CTGen3SystemMatrix(OBJECT_META, meta)
    truth = dict(g0=-0.02, g2=0.05)
    reference = FilteredBackProjection(preprocessing.scale_columns(proj, meta, **truth), sm, filter='hann')()
    fit = preprocessing.fit_column_scale(proj, meta, sm, reference, filter='hann')
    assert fit['g0'] == pytest.approx(truth['g0'], abs=1e-3) and fit['g2'] == pytest.approx(truth['g2'], abs=1e-3)
    meta.water_attenuation = None
    with pytest.raises(ValueError, match='pass a mask'):
        preprocessing.fit_column_scale(proj, meta, sm, reference)


def test_the_window_is_fitted_back_from_a_reference_image():
    # the "scanner" reconstructed the same noisy projections of a water cylinder with its own window, a clinical
    # kernel's shape; the fit finds that window from the noise the two images share
    meta = _helix(rotations=3, n_cols=144)                # projections sampled up to 0.7 cycles per mm
    meta.water_attenuation = MU
    proj = _line_integrals(meta, spheres=[])
    proj = proj + 0.01 * torch.from_numpy(np.random.default_rng(0).standard_normal(proj.shape).astype(np.float32))
    sm = CTGen3SystemMatrix(ObjectMeta(dr=(0.75, 0.75, 2.0), shape=(80, 80, 10)), meta)
    scanner = TabulatedFilter([0.0, 0.25, 0.4, 0.55], [1.0, 1.0, 0.7, 0.35], taper=0.1)
    reference = FilteredBackProjection(proj, sm, filter=scanner)()
    window = preprocessing.fit_window(proj, meta, sm, reference, block=24.0)
    assert isinstance(window, TabulatedFilter) and window.frequencies[0] == 0 and window.values[0] == 1
    f = window.frequencies
    measured = (f > 0) & (f < 0.6)
    expected = scanner(torch.as_tensor(f[measured]), 0.5).numpy()
    assert np.abs(window.values[measured] - expected).max() < 0.02
    with pytest.raises(TypeError):
        preprocessing.fit_window(proj, meta, sm, reference, filter='hann')


def test_the_fbp_streams_its_views_and_weights_columns_as_it_reads_them():
    meta = _helix(rotations=3)
    proj = _line_integrals(meta)
    sm = CTGen3SystemMatrix(OBJECT_META, meta)
    whole = FilteredBackProjection(proj, sm, filter='hann')()
    stats = {}
    try:   # a budget whose eighth holds about 150 of the 70 x 12 rebinned views: several chunks
        pytomography.set_memory_budget(8 * 150 * 70 * 12 * 4 / 1e9)
        streamed = FilteredBackProjection(proj, sm, filter='hann', stats=stats)()
    finally:
        pytomography.set_memory_budget(None)
    assert stats['groups'][0]['chunks'] > 2
    torch.testing.assert_close(streamed, whole, rtol=1e-4, atol=1e-6)
    w = torch.linspace(0.5, 1.5, meta.shape[0])
    weighted = FilteredBackProjection(proj * w[None, :, None], sm, filter='hann')()
    torch.testing.assert_close(FilteredBackProjection(proj, sm, filter='hann', column_weights=w)(), weighted, rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError):
        pytomography.set_memory_budget(0)


@pytest.mark.parametrize("which", ["helical", "cone beam"])
def test_the_projectors_no_longer_accept_a_projection_type(which):
    if which == "helical":
        sm = CTGen3SystemMatrix(OBJECT_META, _helix(rotations=1))
    else:
        sm = CTConeBeamFlatPanelSystemMatrix(ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(8, 8, 4)), _conebeam(12))
    with pytest.raises(TypeError):
        sm.backward(torch.zeros(sm.proj_meta.N_angles, *sm.proj_meta.shape), projection_type='FBP')
    with pytest.raises(TypeError):
        sm.forward(torch.zeros(sm.object_meta.shape), projection_type='FBP')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="measures CUDA memory")
def test_the_gpu_budget_bounds_the_peak():
    object_meta = ObjectMeta(dr=(1.0, 1.0, 2.0), shape=(96, 96, 14))
    meta = _helix()
    proj = _line_integrals(meta)
    budget = 12e6
    with PeakMemory('cuda') as m:
        FilteredBackProjection(proj, CTGen3SystemMatrix(object_meta, meta, device='cpu'), gpu_budget=budget)()
    assert m.peak <= 1.25 * budget


C145_PROJECTIONS, C145_IMAGES = os.environ.get('PYTOMOGRAPHY_C145_PROJECTIONS'), os.environ.get('PYTOMOGRAPHY_C145_IMAGES')


@pytest.mark.data
@pytest.mark.skipif(not (C145_PROJECTIONS and C145_IMAGES), reason="set PYTOMOGRAPHY_C145_PROJECTIONS and PYTOMOGRAPHY_C145_IMAGES")
def test_c145_matches_the_scanner_images():
    """TCIA LDCT-and-Projection-data C145 (GE), 40 slices at the centre of the scan: the GE central column read
    automatically, the per-column scale fitted to the scanner's own (STANDARD) images, and then soft tissue, fat and
    lung within 5 HU of those images, with no radial trend."""
    from scipy import ndimage
    from pytomography.io.shared import open_multifile, align_images_affine
    proj, meta = dicom_ct_pd.get_projections_and_metadata_gen3(C145_PROJECTIONS, table_feed='pitch')
    assert float(meta.detector_centers_col_idx[0]) == pytest.approx(888 - 444.75)   # GE counts it from the other end
    # the scanner's images on our grid; GE's ImagePositionPatient marks the corner of the first pixel, not its centre
    scanner, scanner_meta = open_multifile(sorted(glob.glob(os.path.join(C145_IMAGES, '*.dcm'))), return_object_meta=True)
    ps = float(scanner_meta.dr[0])
    object_meta = ObjectMeta(dr=(ps, ps, 1.0), shape=(512, 512, 40))
    system_matrix = CTGen3SystemMatrix(object_meta, meta)
    affine = scanner_meta.affine_matrix.copy()
    affine[:2, 3] += 0.5 * ps
    theirs = align_images_affine(np.zeros(object_meta.shape, np.float32), scanner.cpu().numpy(),
                                 meta.get_patient_affine(object_meta).numpy(), affine, cval=-3000)
    del scanner
    fit = preprocessing.fit_column_scale(proj, meta, system_matrix, meta.water_attenuation * (1 + theirs / 1000),
                                         filter='hann', slice_thickness=1.25)
    assert fit['g0'] == pytest.approx(-0.012, abs=0.003) and fit['g2'] == pytest.approx(0.0165, abs=0.003)
    image = FilteredBackProjection(preprocessing.scale_columns(proj, meta, **fit), system_matrix, filter='hann', slice_thickness=1.25)()
    ours = (1000 * (image.cpu().numpy() / meta.water_attenuation - 1)).astype(np.float32)
    valid = theirs > -1500
    smooth = ndimage.uniform_filter(np.where(valid, theirs, -1000), size=(7, 7, 3))
    r = np.hypot(*np.meshgrid((np.arange(512) - 255.5) * ps, (np.arange(512) - 255.5) * ps, indexing='ij'))[:, :, None]
    for name, (lo, hi), tol in (('soft tissue', (10, 70), 5), ('fat', (-130, -70), 5), ('lung', (-900, -700), 5)):
        m = ndimage.binary_erosion((smooth >= lo) & (smooth < hi) & valid, np.ones((7, 7, 3)))
        assert ours[m].mean() - theirs[m].mean() == pytest.approx(0, abs=tol), name
    soft = ndimage.binary_erosion((smooth >= 10) & (smooth < 70) & valid, np.ones((9, 9, 3)))
    d = ours - theirs
    assert d[soft & (r < 40)].mean() - d[soft & (r >= 140) & (r < 180)].mean() == pytest.approx(0, abs=12)
