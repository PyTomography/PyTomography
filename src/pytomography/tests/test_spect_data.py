"""The SPECT checks of the I/O audit on the tutorial data, kept as data tests (design section 7). Each reduces a
reconstruction to a number with a threshold, and every axis flip must score worse: they show that the readers put
the reconstruction where the patient is."""
from __future__ import annotations

import warnings

import numpy as np
import pydicom
import pytest
import torch

import pytomography
from pytomography import io
from pytomography.algorithms import OSEM
from pytomography.io import SPECT
from pytomography.io._vendors import UnverifiedVendorRule
from pytomography.io.SPECT.shared import subsample_projections
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.projectors.SPECT import SPECTSystemMatrix, StarGuideSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform

pytestmark = pytest.mark.data
FLIPS = {'x': (0,), 'y': (1,), 'z': (2,), 'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2), 'xyz': (0, 1, 2)}


def _folder(data_dir, *path):
    target = data_dir.joinpath('SPECT', *path)
    if not target.exists():
        pytest.skip(f'SPECT/{"/".join(path)} is not in the tutorial data folder')
    return target


def _ncc(a, b, mask=None):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if mask is not None:
        a, b = a[mask], b[mask]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def _osem(object_meta, proj_meta, projections, n_iters, n_subsets, transforms=(), scatter=None):
    H = SPECTSystemMatrix(list(transforms), [], object_meta, proj_meta)
    return OSEM(PoissonLogLikelihood(H, projections, additive_term=scatter))(n_iters=n_iters, n_subsets=n_subsets).cpu().numpy()


def _quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnverifiedVendorRule)
        return f(*args, **kwargs)


def test_siemens_lu177_matches_siemens_reconstruction(data_dir):
    """Siemens Symbia Intevo, Lu-177 NEMA (v1): our OSEM against Siemens' own reconstruction, in patient space."""
    root = _folder(data_dir, 'Lu177-NEMA-SymT2')
    files_CT = sorted(str(f) for f in (root / 'CT').iterdir())
    projections, object_meta, proj_meta = SPECT.read_dicom(str(root / 'projection_data.dcm'))
    window = proj_meta.energy_windows[0]
    mu = io.attenuation_map(files_CT, object_meta, window.photon_keV)
    scatter = SPECT.energy_window_scatter(projections, proj_meta, 0, 1, 2)
    recon = _osem(object_meta, proj_meta, projections[0], 2, 8, [SPECTAttenuationTransform(mu)], scatter)
    vendor = io.load_image(root / 'scanner_recon.dcm').on(object_meta).cpu().numpy()
    score = _ncc(recon, vendor)
    assert score >= 0.93
    for axes in FLIPS.values():
        assert _ncc(np.flip(recon, axes), vendor) < score - 0.05
    for axis in range(3):                                         # the best shift is within one voxel
        scores = {s: _ncc(np.roll(recon, s, axis=axis), vendor) for s in range(-2, 3)}
        assert abs(max(scores, key=scores.get)) <= 1


def test_ge_kidney_masks_land_on_the_kidneys(data_dir):
    """GE Discovery 670, Lu-177 PSMA (v3): the kidneys drawn on the CT (RTSTRUCT) are the hottest organs of our
    reconstruction, best at no shift."""
    root = _folder(data_dir, 'Lu177-PSMA-GEDisc')
    files_CT = sorted(str(f) for f in (root / 'CT').iterdir())
    projections, object_meta, proj_meta = _quiet(SPECT.read_dicom, str(root / 'bed2_projections.dcm'), energy_window=1)
    recon = _osem(object_meta, proj_meta, projections, 2, 6)
    masks = io.load_mask(root / 'segmentations.dcm', reference=files_CT, rois=['kidney_left', 'kidney_right'])
    kidneys = (masks.on(object_meta, label='kidney_left', threshold=0.5)
               | masks.on(object_meta, label='kidney_right', threshold=0.5)).cpu().numpy()
    body = (io.attenuation_map(files_CT, object_meta, 208.4) > 0.005).cpu().numpy()
    ratio = lambda image: image[kidneys].mean() / image[body].mean()
    assert ratio(recon) >= 10
    for axes in FLIPS.values():
        assert ratio(np.flip(recon, axes)) < ratio(recon)
    for axis in range(3):
        for shift in (-1, 1):
            assert np.roll(recon, shift, axis=axis)[kidneys].mean() < recon[kidneys].mean()


def test_multibed_masks_on_the_stitched_grid(data_dir):
    """GE, two beds (v3): on the stitched grid, the lungs drawn on the CT hold lung attenuation and the liver soft
    tissue (with the mask on the wrong bed they were 354 mm off)."""
    root = _folder(data_dir, 'Lu177-PSMA-GEDisc')
    files_CT = sorted(str(f) for f in (root / 'CT').iterdir())
    metas = [_quiet(SPECT.read_dicom, str(root / f'bed{b}_projections.dcm'), energy_window=1)[1] for b in (1, 2)]
    mus = [io.attenuation_map(files_CT, meta, 208.4) for meta in metas]
    mu, meta = SPECT.stitch_beds(mus, metas)
    lungs = ['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right', 'lung_lower_lobe_right']
    masks = io.load_mask(root / 'segmentations.dcm', reference=files_CT, rois=lungs + ['liver'])
    mu = mu.cpu().numpy()
    for name in lungs:
        assert mu[masks.on(meta, label=name, threshold=0.5).cpu().numpy()].mean() < 0.008
    assert mu[masks.on(meta, label='liver', threshold=0.5).cpu().numpy()].mean() > 0.012


def test_cardiac_heart_is_left_and_anterior(data_dir):
    """Siemens Symbia T6, Tc-99m cardiac (v4): two heads at 90 degrees, a 180 degree arc, clockwise: the heart comes
    out to the patient's left of the body's activity, and anterior."""
    root = _folder(data_dir, 'Tc99m-Cardiac')
    projections, object_meta, proj_meta = SPECT.read_dicom(str(root / 'anonymized_file.dcm'), energy_window=0)
    recon = _osem(object_meta, proj_meta, projections, 3, 4)
    def centroid(fraction):
        m = recon > fraction * recon.max()
        ijk = (np.argwhere(m) * recon[m][:, None]).sum(0) / recon[m].sum()
        return (object_meta.affine @ [*ijk, 1])[:3]
    heart_minus_all = centroid(0.6) - centroid(0.05)
    assert heart_minus_all[0] > 30 and heart_minus_all[1] < 0


def test_starguide_matches_ge_reconstruction(data_dir):
    """GE StarGuide, Tc-99m NEMA (v5): our reconstruction against GE's, both on the grid centred on the CT."""
    root = _folder(data_dir, 'Tc99m-NEMA-Starguide')
    files_NM = sorted(str(f) for f in (root / 'NM_files').iterdir()
                      if pydicom.dcmread(str(f), stop_before_pixels=True).SeriesDescription == 'NoFocus Cont')
    files_CT = sorted(str(f) for f in (root / 'CT_files').iterdir())
    projections, object_meta, proj_meta = SPECT.read_starguide(files_NM, files_CT=files_CT, nearest_theta=2.0)
    mu = io.attenuation_map(files_CT, object_meta, proj_meta.energy_windows[0].photon_keV)
    scatter = SPECT.energy_window_scatter(projections, proj_meta, 0, 1)
    H = StarGuideSystemMatrix(object_meta, proj_meta, obj2obj_transforms=[SPECTAttenuationTransform(mu, assume_padded=False)])
    recon = OSEM(PoissonLogLikelihood(H, projections[0], additive_term=scatter))(n_iters=1, n_subsets=10).cpu().numpy()
    vendor = io.load_image(root / 'vendor_recon' / 'i196884.NMDC.1').on(object_meta).cpu().numpy()
    ii, jj = np.meshgrid(np.arange(196), np.arange(196), indexing='ij')
    central = np.broadcast_to(((ii - 97.5) ** 2 + (jj - 97.5) ** 2 < 80 ** 2)[:, :, None], recon.shape)
    score = _ncc(recon * central, vendor * central)
    assert score >= 0.9
    for axes in FLIPS.values():
        assert _ncc(np.flip(recon, axes) * central, vendor * central) < score - 0.2


def test_simind_xcat_lungs_with_the_mu_castor_layout(data_dir):
    """SIMIND XCAT (v6, v6b): reconstructions of the lung sources lie on the lungs of the density map read with
    layout 'mu-castor' (index 22 = 5); with 'mu' (the old default) they do not."""
    root = _folder(data_dir, 'SIMIND-MultiOrgan', 'multi_projections')
    for organ in ('r_lung', 'l_lung'):
        projections, object_meta, proj_meta = SPECT.read_simind(str(root / organ / 'primary.h00'), energy_window=0)
        projections = subsample_projections(projections, 2, 2)
        object_meta, proj_meta = object_meta.subsample(2), proj_meta.subsample(2, 2)
        recon = _osem(object_meta, proj_meta, projections * 1e4, 2, 6)
        for layout, expected in (('mu-castor', True), ('mu', False)):
            mu = io.load_image(root / 'mu208.hct', layout=layout).on(object_meta).cpu().numpy() / 10   # 1/mm
            lung_like = (mu > 0.001) & (mu < 0.007)
            fraction = recon[lung_like].sum() / recon.sum()
            assert (fraction >= 0.7) == expected, (layout, organ, fraction)
