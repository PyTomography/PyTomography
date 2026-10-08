"""Filtered back projection in the CT system matrices. Helical WFBP (CTGen3SystemMatrix) reconstructs a phantom from its
exact line integrals (computed analytically along each ray, so independent of the projector), with and without a
flying focal spot; the circular FDK (CTConeBeamFlatPanelSystemMatrix) gives the same image through the algorithm as
through the deprecated back projection flag; voxels outside the field of view are zero; the GPU budget bounds the peak.
A data test (marked ``data``) compares TCIA LDCT-and-Projection-data case C145 with the scanner's own images."""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest
import torch

import pytomography

parallelproj_core = pytest.importorskip("parallelproj_core", reason="CT system matrices require parallelproj 2")
from pytomography.algorithms import FilteredBackProjection
from pytomography.io.CT import dicom_ct_pd
from pytomography.metadata import ObjectMeta
from pytomography.metadata.CT import CTConeBeamFlatPanelProjMeta, CTGen3ProjMeta
from pytomography.projectors.CT import CTConeBeamFlatPanelSystemMatrix, CTGen3SystemMatrix
from pytomography.utils import PeakMemory

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


def _line_integrals(meta):
    """Exact line integrals of the phantom along every ray, from each view's focal spot to each detector element."""
    end = meta.get_detector_coordinates(torch.arange(meta.N_angles)).double()
    start = meta.source_focal_spots.double()[:, None, None, :].expand_as(end)
    d = end - start
    L = d.norm(dim=-1)
    u = d / L[..., None]
    p = MU * _chords_cylinder(start, u, L, **CYLINDER)
    for centre, radius, dmu in SPHERES:
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


def test_helical_fbp_is_zero_outside_the_field_of_view_and_on_the_device():
    sm = CTGen3SystemMatrix(ObjectMeta(dr=(3.0, 3.0, 2.0), shape=(48, 48, 6)), _helix())
    image = FilteredBackProjection(_line_integrals(sm.proj_meta), sm)()
    assert image.device.type == torch.device(pytomography.device).type
    x = (torch.arange(48) - 23.5) * 3.0
    outside = (x[:, None] ** 2 + x[None, :] ** 2) > sm.fov_radius ** 2
    assert torch.all(image[outside.to(image.device)] == 0)


def test_back_projection_no_longer_accepts_a_projection_type():
    sm = CTGen3SystemMatrix(OBJECT_META, _helix(rotations=1))
    with pytest.raises(TypeError):
        sm.backward(torch.zeros(sm.proj_meta.N_angles, *sm.proj_meta.shape), projection_type='FBP')


def test_conebeam_fdk_through_the_algorithm_matches_the_deprecated_flag():
    angles = torch.linspace(0, 2 * np.pi, 25)[:-1]
    meta = CTConeBeamFlatPanelProjMeta(angles, torch.zeros(24), detector_radius=100.0, beam_radius=150.0, shape=(40, 16), dr=(2.5, 2.5))
    sm = CTConeBeamFlatPanelSystemMatrix(ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(32, 32, 12)), meta)
    proj = torch.rand((24, 40, 16), generator=torch.Generator().manual_seed(0)).to(pytomography.device)
    with pytest.warns(DeprecationWarning):
        old = sm.backward(proj, projection_type='FBP')
    new = FilteredBackProjection(proj, sm, filter='ram-lak')()
    # the same computation; parallelproj accumulates the back projection with atomic adds, so not bit for bit
    torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-6 * float(old.abs().max()))
    hann = FilteredBackProjection(proj, sm, filter='hann')()
    assert (hann - new).abs().max() > 0.01 * new.abs().max()


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
    """TCIA LDCT-and-Projection-data C145 (GE), 40 slices at the centre of the scan, with the conventions found for it,
    against the scanner's own (STANDARD) images: soft tissue, fat and lung within 5 HU, and no radial trend."""
    import pydicom
    from scipy import ndimage
    proj, meta = dicom_ct_pd.get_projections_and_metadata_gen3(
        C145_PROJECTIONS, central_column_offset=-0.42, angle_offset_deg=0.161, table_feed='pitch',
        column_scale=dict(g0=-0.0149, g2=0.0186))
    object_meta = ObjectMeta(dr=(0.662109, 0.662109, 1.0), shape=(512, 512, 40))
    image = FilteredBackProjection(proj, CTGen3SystemMatrix(object_meta, meta), filter='hann', slice_thickness=1.25)()
    ours = (1000 * (image.cpu().numpy() / meta.water_attenuation - 1)).astype(np.float32)
    # the scanner's images, sampled at our voxels (their pixel centres at ImagePositionPatient + half a pixel)
    sl = sorted((pydicom.dcmread(f) for f in glob.glob(os.path.join(C145_IMAGES, '*.dcm'))), key=lambda s: float(s.ImagePositionPatient[2]))
    zs = np.array([float(s.ImagePositionPatient[2]) for s in sl])
    ps = float(sl[0].PixelSpacing[0])
    x0, y0 = (float(v) for v in sl[0].ImagePositionPatient[:2])
    hu_v = np.stack([s.pixel_array * float(s.RescaleSlope) + float(s.RescaleIntercept) for s in sl]).astype(np.float32)
    A = meta.get_patient_affine(object_meta).numpy()
    i, j, k = np.meshgrid(*[np.arange(n) for n in object_meta.shape], indexing='ij')
    xyz = A[:3, :3] @ np.stack([i.ravel(), j.ravel(), k.ravel()]) + A[:3, 3:4]
    coords = np.stack([np.interp(xyz[2], zs, np.arange(len(zs))), (xyz[1] - y0) / ps - 0.5, (xyz[0] - x0) / ps - 0.5])
    theirs = ndimage.map_coordinates(hu_v, coords, order=1, cval=-3000).reshape(object_meta.shape)
    valid = theirs > -1500
    smooth = ndimage.uniform_filter(np.where(valid, theirs, -1000), size=(7, 7, 3))
    r = np.hypot(*np.meshgrid((np.arange(512) - 255.5) * ps, (np.arange(512) - 255.5) * ps, indexing='ij'))[:, :, None]
    for name, (lo, hi), tol in (('soft tissue', (10, 70), 5), ('fat', (-130, -70), 5), ('lung', (-900, -700), 5)):
        m = ndimage.binary_erosion((smooth >= lo) & (smooth < hi) & valid, np.ones((7, 7, 3)))
        assert ours[m].mean() - theirs[m].mean() == pytest.approx(0, abs=tol), name
    soft = ndimage.binary_erosion((smooth >= 10) & (smooth < 70) & valid, np.ones((9, 9, 3)))
    d = ours - theirs
    assert d[soft & (r < 40)].mean() - d[soft & (r >= 140) & (r < 180)].mean() == pytest.approx(0, abs=12)
