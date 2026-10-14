"""Patient positions on open data (decision 6): the readers place images and projections by their orientation and
position tags (ImageOrientationPatient, ImagePositionPatient, the NM detector vectors) alone, never by PatientPosition,
and that is right in every position: HFS, HFP, FFP, FFS, decubitus, on GE, Siemens, Philips and Toshiba.

The data are the open sets listed in ``io_tests`` of the tutorial data folder (TCIA CT COLONOGRAPHY, LDCT C095,
COVID-19-NY-SBU, ACRIN-NSCLC-FDG-PET, VAREPOP-APOLLO; Zenodo 17897044 and 18890542; figshare 14785056); each test is
skipped when its folder is absent. A few slices of each CT are read."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pydicom
import pytest
from scipy import ndimage

from pytomography import io
from pytomography.algorithms import OSEM
from pytomography.io import SPECT
from pytomography.io._vendors import UnverifiedVendorRule
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.projectors.SPECT import SPECTSystemMatrix

pytestmark = pytest.mark.data
FLIPS = {'x': (0,), 'y': (1,), 'z': (2,), 'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2), 'xyz': (0, 1, 2)}
BELGIAN = 'zenodo17897044_belgian_lu177_multicentre'

# (folder pattern under io_tests, PatientPosition): CT series in every position, from four vendors
CT_SERIES = [
    ('tcia_ldct_c095_siemens_hfs/C095/Full_Dose_Images_*', 'HFS'),
    ('tcia_ctcolonography_prone_supine/*0261/Colo_prone_*', 'FFP'),
    ('tcia_ctcolonography_prone_supine/*0258/COLONOGRAPHY_*', 'FFP'),
    ('tcia_ctcolonography_prone_supine/*0334/Colo_supine_*', 'HFS'),
    ('tcia_ctcolonography_positions_subsets/*0482/Colon_Prone_*', 'HFP'),
    ('tcia_ctcolonography_positions_subsets/*0653/Prone_Colon_flipped_*', 'HFP'),
    ('tcia_ctcolonography_positions_subsets/*0282/PRONE_4.314078_every4', 'FFP'),
    ('tcia_ctcolonography_positions_subsets/*0534/prone_C_MINUS_*every4', 'FFP'),
    ('tcia_ctcolonography_positions_subsets/*0169/PRONE_4.190292_every4', 'FFP'),
    ('tcia_ctcolonography_positions_subsets/*0448/PRONE_C-_LOW_DOSE_COLON_4.503271_every4', 'FFP'),
    ('tcia_ctcolonography_positions_subsets/*0448/SUPINE_C-_LOW_DOSE_COLON_4.502735_every4', 'FFS'),
    ('tcia_ctcolonography_positions_subsets/*0608/Decub_Colon_*', 'HFDR'),
    ('tcia_ctcolonography_positions_subsets/*0193/RT_DECUB_*every4', 'FFDR'),
]


def _root(data_dir):
    root = data_dir / 'io_tests'
    if not root.exists():
        pytest.skip('io_tests is not in the tutorial data folder')
    return root


def _one(root, pattern):
    found = sorted(p for p in root.glob(pattern) if not p.name.endswith('_offgrid'))
    if not found:
        pytest.skip(f'{pattern} is not in io_tests')
    return found[0]


def _top_run(folder, n=24):
    """Up to ``n`` evenly spaced slices of the most superior evenly spaced run of a series (the downloaded subsets
    have gaps, and one series has two slabs 330 mm apart), chosen from the headers."""
    heads = []
    for name in os.listdir(folder):
        try:
            ds = pydicom.dcmread(os.path.join(folder, name), stop_before_pixels=True,
                                 specific_tags=['ImagePositionPatient', 'ImageOrientationPatient', 'PatientPosition'])
        except Exception:
            continue
        if 'ImagePositionPatient' in ds:
            heads.append((os.path.join(folder, name), ds))
    iop = np.array(heads[0][1].ImageOrientationPatient, float)
    normal = np.cross(iop[:3], iop[3:])
    positions = np.array([np.array(ds.ImagePositionPatient, float) for _, ds in heads])
    order = np.argsort(positions @ normal)
    along = (positions @ normal)[order]
    gaps = np.diff(along)
    runs, start = [], 0
    for i in range(1, len(gaps) + 1):
        if i == len(gaps) or abs(gaps[i] - gaps[start]) > 0.01 * gaps[start]:
            runs.append(order[start:i + 1])
            start = i
    runs = [r for r in runs if len(r) >= 8] or runs
    run = max(runs, key=lambda r: positions[r, 2].mean())             # the most superior (patient z)
    return [heads[i][0] for i in run[::max(1, len(run) // n)]], heads[0][1].get('PatientPosition')


@pytest.mark.parametrize('pattern, position', CT_SERIES, ids=[f'{p}-{s.split("/")[1][:14]}' for s, p in CT_SERIES])
def test_ct_spine_is_posterior_in_every_position(data_dir, pattern, position):
    """load_image places a CT by IOP/IPP only: in every position the midline bone (the spine) lies posterior (+y)
    of the body's centre, above the pelvis."""
    files, stated = _top_run(_one(_root(data_dir), pattern))
    assert stated == position
    ct = io.load_image(files)
    body = np.zeros(ct.shape, bool)
    for k in range(ct.shape[2]):                     # the patient: the largest connected region of each slice
        labels, n = ndimage.label(ct.data[:, :, k] > -300)
        if n:
            body[:, :, k] = labels == 1 + int(np.argmax(ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))))
    ijk = np.argwhere(body)
    xyz = ijk @ ct.affine[:3, :3].T + ct.affine[:3, 3]
    centre = xyz.mean(axis=0)
    bone = (ct.data[body] > 400) & (np.abs(xyz[:, 0] - centre[0]) < 25)
    assert bone.sum() > 100
    assert np.median(xyz[bone, 1]) - centre[1] > 15, position


def test_hfs_nm_reconstructions_land_on_their_ct(data_dir):
    """Belgian sites D (GE Discovery 670 Pro: no PatientPosition, orientation code head first) and B2 (GE StarGuide,
    HFS): the vendor's reconstructions, placed by load_image, lie on their CTs, their hot regions on water. Flips of x
    and y score worse for site D's uniform cylinder, and flips of y for B2's NEMA phantom; the NEMA phantom at site D
    lies symmetric on the grid, and along z every phantom is symmetric, so those flips cannot be told apart."""
    root = _root(data_dir)
    for nm_pattern, ct_pattern, minimum, axes in [
            (f'{BELGIAN}/D/*Cylinder.-.C-Recon*', f'{BELGIAN}/D/*Cylinder.-.ACCT*', 0.99, 'xy'),
            (f'{BELGIAN}/D/*NEMA.-.C-Recon*', f'{BELGIAN}/D/*NEMA.-.ACCT*', 0.99, ''),
            (f'{BELGIAN}/B2/*C-Recon*', f'{BELGIAN}/B2/*ACCT*', 0.98, 'y')]:
        nm = io.load_image(_one(root, nm_pattern))
        ct = io.load_image(_one(root, ct_pattern))
        assert ct.meta['PatientPosition'] == 'HFS'
        hu = ct.on(nm.object_meta, fill=-1000).cpu().numpy()
        activity = nm.data
        body = hu > -500
        inside = activity[body].sum() / activity.sum()
        assert inside >= minimum, nm_pattern
        hot = activity > 0.5 * activity.max()
        assert -100 < hu[hot].mean() < 100                # the hot regions hold water
        for name, flip in FLIPS.items():                  # the flips the phantom's shape can tell apart
            if set(name) & set(axes):
                assert np.flip(activity, flip)[body].sum() / activity.sum() < inside - 0.03, (nm_pattern, name)


# (folder pattern, detector-1 start angle, direction): FFS TOMO from systems the tutorials do not have
TOMO = [
    (f'{BELGIAN}/B1/*Tomo_n120*', 180.03, 'CW'),                      # GE Discovery 870 DR, CW from 180
    (f'{BELGIAN}/B1/*Tomo_n360*', 180.03, 'CW'),
    ('tcia_covid19nysbu_siemens_spect/A130302/Tc99m_PYP_SPECT_1*', None, 'CC'),   # Siemens Symbia T6
    ('tcia_covid19nysbu_siemens_spect/A201130/LUNG_TOMO_4*', None, 'CC'),
    ('tcia_acrin_nsclc_siemens_spect/ACRIN-NSCLC-FDG-PET-134/Lung_Thorax_4*', None, 'CC'),
    ('tcia_acrin_nsclc_siemens_spect/ACRIN-NSCLC-FDG-PET-164/Lung_Thorax_0*', None, 'CC'),
    ('zenodo18890542_mrtdosimetry_ge/*/Raw SPECT/EM208.dcm', 179.58, 'CC'),       # GE D670 Pro, one window a file
    ('figshare14785056_mezzenga_lu177_cylinder/TW.dcm', 180.23, 'CC'),            # GE D670
    ('tcia_varepop_apollo_siemens_iqspect/AP-2BGM/Rest_IQ_3*', 239.0, 'CW'),      # Siemens IQ-SPECT, heads at 90
]


@pytest.mark.parametrize('pattern, start, direction', TOMO, ids=[p.split('/')[-1][:20] for p, _, _ in TOMO])
def test_ffs_tomo_files_read_with_their_angles(data_dir, pattern, start, direction):
    """read_dicom reads every new TOMO file. The first two frames of the file (window 1, detector 1, views 1 and 2)
    are the projections the reader puts at β = start + 180 and start + 180 ± step (clockwise: minus), with the rows
    and columns flipped as the detector's ImageOrientationPatient says; on Siemens files with a registration to their
    CT (rule 'siemens-nm-registration'), every β turned by its angle about z."""
    path = _one(_root(data_dir), pattern)
    path = next(path.glob('*.dcm')) if path.is_dir() else path
    ds = pydicom.dcmread(str(path))
    rotation = ds.RotationInformationSequence[0]
    assert str(rotation.RotationDirection) == direction
    detector = ds.DetectorInformationSequence[0]
    first = float(detector.StartAngle) if 'StartAngle' in detector else float(rotation.StartAngle)
    if start is not None:
        assert first == pytest.approx(start)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnverifiedVendorRule)
        projections, object_meta, proj_meta = SPECT.read_dicom(str(path))
    assert projections.shape[0] == len(proj_meta.energy_windows) and object_meta.frame == 'patient'
    angles = proj_meta.angles.cpu().numpy()
    assert np.all(np.diff(angles) > 0)
    step = float(rotation.AngularStep) * (1 if direction == 'CC' else -1)
    iop = np.array(detector.ImageOrientationPatient, float)
    frames = ds.pixel_array.astype(np.float32) * float(ds.get('RescaleSlope', 1) or 1)
    M = SPECT.dicom.siemens_registration(ds)
    turn = 0.0 if M is None else np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    for frame, beta in ((0, first + 180), (1, first + 180 + step)):
        turned = (beta - turn) % 360
        view = int(np.argmin(np.abs((angles - turned + 180) % 360 - 180)))
        assert abs((angles[view] - turned + 180) % 360 - 180) < 1e-3
        raw = frames[frame][::-1] if iop[5] < 0 else frames[frame]          # rows superior first: flipped
        b = np.radians(beta % 360)                                          # the IOP is in the acquisition's frame
        if iop[:3] @ [np.cos(b), -np.sin(b), 0] < 0:
            raw = raw[:, ::-1]
        assert np.array_equal(projections[0, view].cpu().numpy(), raw.T)


def test_siemens_tomo_matches_its_vendor_reconstruction(data_dir):
    """COVID-19-NY-SBU Tc-99m PYP on a Siemens Symbia T6: our OSEM, without attenuation, against Siemens' own
    reconstruction without attenuation, in patient space; every flip scores worse. Its mu map (a Siemens NM RECON
    without PatientPosition) reads too, on the same patient frame."""
    root = _root(data_dir)
    folder = 'tcia_covid19nysbu_siemens_spect/A130302/'
    tomo = next(_one(root, folder + 'Tc99m_PYP_SPECT_1*').glob('*.dcm'))
    projections, object_meta, proj_meta = SPECT.read_dicom(str(tomo), energy_window=0)
    H = SPECTSystemMatrix([], [], object_meta, proj_meta)
    recon = OSEM(PoissonLogLikelihood(H, projections))(n_iters=2, n_subsets=8).cpu().numpy()
    vendor = io.load_image(_one(root, folder + 'Tc99m_PYP_SPECT_Recon_-_NoAC*')).on(object_meta).cpu().numpy()

    def ncc(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    score = ncc(recon, vendor)
    assert score >= 0.93
    for flip in FLIPS.values():
        assert ncc(np.flip(recon, flip), vendor) < score - 0.05
    mu = io.load_image(_one(root, folder + 'AC_Chest*'))
    assert mu.modality == 'NM' and 'PatientPosition' not in mu.meta
    assert np.linalg.norm(mu.centre[:2] - np.array(object_meta.centre[:2])) < 10


def test_siemens_reconstructions_lie_where_the_registration_puts_the_grid(data_dir):
    """Siemens' registration (rule 'siemens-nm-registration'): every reconstruction (AC and NoAC) and CT mu map of the
    COVID-19-NY-SBU Symbia T6 studies lies on the axis of read_dicom's grid, which the registration has moved into the
    CT's frame: its axes are the registration's rotation, and its centre is on the (tilted) rotation axis through the
    grid's centre, to 0.05 mm. The reconstructions are cropped along that axis, so their centres differ along it."""
    root = _root(data_dir)
    studies = sorted(p for p in root.glob('tcia_covid19nysbu_siemens_spect/*') if p.is_dir())
    checked = 0
    for study in studies:
        tomos = [p for p in study.iterdir() if p.is_dir() and ('TOMO_' in p.name or p.name.startswith('Tc99m_PYP_SPECT_'))
                 and 'Recon' not in p.name and 'MIP' not in p.name]
        for tomo in tomos:
            first = next(tomo.glob('*.dcm'))
            ds = pydicom.dcmread(str(first), stop_before_pixels=True)
            M = SPECT.dicom.siemens_registration(ds)
            assert M is not None
            _, object_meta, _ = SPECT.read_dicom(str(first), energy_window=0)
            axis = M[:3, :3] @ [0.0, 0.0, 1.0]
            for image_dir in [p for p in study.iterdir() if p.is_dir() and ('Recon' in p.name and 'MIP' not in p.name)]:
                files = sorted(str(f) for f in image_dir.glob('*.dcm'))
                image = io.load_image(files if len(files) > 1 else files[0])
                d = image.centre - np.array(object_meta.centre)
                off_axis = d - (d @ axis) * axis
                assert np.linalg.norm(off_axis) < 0.05, (image_dir.name, off_axis)
                axes = image.affine[:3, :3] / np.linalg.norm(image.affine[:3, :3], axis=0)
                assert np.allclose(axes, M[:3, :3], atol=2e-3), image_dir.name
                checked += 1
    if not checked:
        pytest.skip('no COVID-19-NY-SBU reconstructions in io_tests')
