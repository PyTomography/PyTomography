"""The SPECT readers (io.SPECT): synthetic NM files written in each vendor's conventions (design section 7), SIMIND
headers of each version, energy windows, collimator PSFs, multi-bed stitching.

The NM files hold projections made by PyTomography's own projector from three point markers, written as a Siemens or
GE camera would: DICOM angles (β - 180), rows and columns in the order the heads' ImageOrientationPatient say, the
vendor's ImagePositionPatient. Reading a file must give back the projections, the angles and radii, and the grid's
place in the patient; a reconstruction must put the markers back within one voxel."""
from __future__ import annotations

import datetime
import warnings

import numpy as np
import pydicom
import pytest
import torch
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import pytomography
from pytomography.algorithms import OSEM
from pytomography.io import SPECT
from pytomography.io._vendors import UnverifiedVendorRule
from pytomography.io.SPECT._energy import nuclide_key, photon_energy
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.metadata.SPECT import EnergyWindow, SPECTObjectMeta, SPECTProjMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix

LR, LZ, DR, DZ = 24, 20, 4.0, 6.0                  # columns, rows, column and row spacing (mm): non-square
CENTRE = (0.0, -150.0, 400.0)
MARKERS = {(17, 8, 5): 100.0, (6, 18, 9): 200.0, (12, 5, 15): 300.0}      # voxel index: value, off-centre on each axis
WINDOWS = [('Lu177 208', (187.2, 228.8)), ('lower', (166.4, 187.2)), ('upper', (228.8, 249.6))]


def _u(beta_deg):
    b = np.radians(beta_deg)
    return np.array([np.cos(b), -np.sin(b), 0.0])


def _acquisition(n_heads, views_per_head, step, direction, starts):
    """The DICOM angle of every (head, view) and β, sorted."""
    sign = 1 if direction == 'CC' else -1
    dicom = {(h + 1, v + 1): starts[h] + sign * step * v for h in range(n_heads) for v in range(views_per_head)}
    beta = {key: (angle + 180) % 360 for key, angle in dicom.items()}
    return dicom, beta, np.array(sorted(beta.values()))


def _projections(betas, radius=200.0):
    """Expected counts of the markers (and a weak uniform cylinder) at the views ``betas``, every window."""
    object_meta = SPECTObjectMeta((DR, DR, DZ), (LR, LR, LZ))
    proj_meta = SPECTProjMeta((LR, LZ), (DR, DZ), betas, np.full(len(betas), radius))
    obj = torch.zeros(object_meta.shape)
    for ijk, value in MARKERS.items():
        obj[ijk] = value
    H = SPECTSystemMatrix([], [], object_meta, proj_meta)
    peak = H.forward(obj.to(pytomography.device)).cpu().numpy()
    return np.round(np.stack([peak, 0.3 * peak, 0.1 * peak])).astype(np.float32)        # [W, θ, r, z]


def _write_nm(path, *, vendor='SIEMENS', n_heads=2, views_per_head=12, step=15.0, direction='CC', starts=(180.0, 0.0),
              per_head_start=True, position='FFS', flip_r=False, flip_z=True, centre=CENTRE, rotations=1, slots=1,
              drop_frames=0, registration=None):
    """A NM TOMO file of the marker projections, as ``vendor`` writes it (with Siemens' registration matrix
    ``registration``, 4 x 4, in the private tag (0043,1002) when given); returns (β sorted, projections [W, θ, r, z],
    radii sorted)."""
    dicom_angle, beta, betas = _acquisition(n_heads, views_per_head, step, direction, starts)
    radius = {key: 200.0 + 3 * k for k, key in enumerate(sorted(beta))}          # a non-circular orbit
    order = sorted(beta, key=lambda key: beta[key])
    projections = _projections(betas)
    index = {key: i for i, key in enumerate(order)}
    frames, vectors = [], {'EnergyWindowVector': [], 'DetectorVector': [], 'RotationVector': [], 'AngularViewVector': []}
    if slots > 1:
        vectors['TimeSlotVector'] = []
    for w in range(len(WINDOWS)):
        for s in range(slots):
            for r in range(rotations):
                for h in range(1, n_heads + 1):
                    for v in range(1, views_per_head + 1):
                        img = projections[w, index[(h, v)]] * (1 + s)            # (r, z)
                        if flip_r:
                            img = img[::-1, :]
                        img = img.T                                               # (z ascending, r)
                        if flip_z:
                            img = img[::-1, :]                                    # row 0 superior
                        frames.append(img)
                        vectors['EnergyWindowVector'].append(w + 1)
                        vectors['DetectorVector'].append(h)
                        vectors['RotationVector'].append(r + 1)
                        vectors['AngularViewVector'].append(v)
                        if slots > 1:
                            vectors['TimeSlotVector'].append(s + 1)
    frames = frames[:len(frames) - drop_frames]
    ds = Dataset()
    ds.Modality, ds.ImageType = 'NM', ['ORIGINAL', 'PRIMARY', 'TOMO', 'EMISSION']
    ds.Manufacturer = {'SIEMENS': 'SIEMENS NM', 'GE': 'GE MEDICAL SYSTEMS', 'ACME': 'Acme'}[vendor]
    ds.PatientPosition = position
    ds.AcquisitionDate, ds.AcquisitionTime = '20260101', '101500.25'
    ds.Rows, ds.Columns = LZ, LR
    ds.PixelSpacing = [DZ, DR]                                   # between rows (z), between columns (r)
    ds.NumberOfFrames = len(frames)
    for key, value in vectors.items():
        setattr(ds, key, value)
    ds.NumberOfDetectors, ds.NumberOfRotations, ds.NumberOfEnergyWindows = n_heads, rotations, len(WINDOWS)
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = np.ascontiguousarray(np.stack(frames)).astype(np.uint16).tobytes()
    windows = []
    for name, (lo, hi) in WINDOWS:
        rng = Dataset()
        rng.EnergyWindowLowerLimit, rng.EnergyWindowUpperLimit = lo, hi
        item = Dataset()
        item.EnergyWindowName, item.EnergyWindowRangeSequence = name, [rng]
        windows.append(item)
    ds.EnergyWindowInformationSequence = windows
    code = Dataset()
    code.CodeMeaning = 'Lutetium^177^'
    pharma = Dataset()
    pharma.RadionuclideCodeSequence = [code]
    ds.RadiopharmaceuticalInformationSequence = [pharma]
    z_row0 = centre[2] + (LZ - 1) / 2 * DZ * (1 if flip_z else -1)
    if vendor == 'GE':
        ipp = [centre[0] - (LR - 1) / 2 * DR, centre[1] - (LR - 1) / 2 * DR, z_row0]
    else:
        ipp = [0.0, 0.0, z_row0]
    heads = []
    for h in range(1, n_heads + 1):
        head = Dataset()
        head.ImagePositionPatient = ipp
        row = _u(beta[(h, 1)]) * (-1 if flip_r else 1)
        head.ImageOrientationPatient = [*row.round(8).tolist(), 0.0, 0.0, -1.0 if flip_z else 1.0]
        if per_head_start:
            head.StartAngle = starts[h - 1]
        if n_heads > 1:                                         # per head (Siemens); GE: one detector, below
            head.RadialPosition = [radius[(h, v)] for v in range(1, views_per_head + 1)]
        heads.append(head)
    ds.DetectorInformationSequence = heads
    rotation_items = []
    for r in range(rotations):
        rot = Dataset()
        rot.StartAngle, rot.AngularStep, rot.RotationDirection = starts[0], step, direction
        rot.NumberOfFramesInRotation = views_per_head
        rot.ActualFrameDuration = 15000
        rot.TableHeight = -centre[1]
        if n_heads == 1:
            rot.RadialPosition = [radius[(h, v)] for h in range(1, n_heads + 1) for v in range(1, views_per_head + 1)]
        rotation_items.append(rot)
    ds.RotationInformationSequence = rotation_items
    if registration is not None:
        ds.private_block(0x0043, 'SIEMENS MED NM', create=True).add_new(0x02, 'FD', np.asarray(registration, float).ravel().tolist())
    ds.SOPClassUID, ds.SOPInstanceUID = '1.2.840.10008.5.1.4.1.1.20', generate_uid()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.save_as(str(path), enforce_file_format=True)
    radii = np.array([radius[key] for key in order])
    return betas, projections, radii


def _read(path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnverifiedVendorRule)
        return SPECT.read_dicom(str(path), **kwargs)


CASES = {
    'siemens-FFS-CC': dict(),
    'siemens-FFS-CW': dict(direction='CW', starts=(180.0, 0.0)),
    'siemens-per-head-starts': dict(starts=(225.0, 135.0), step=5.625, views_per_head=16),
    'ge-one-detector-CW': dict(vendor='GE', n_heads=1, views_per_head=24, starts=(0.16,), direction='CW',
                               per_head_start=False, centre=(0.3, -6.9, -177.0)),
    'HFS-rows-up-columns-mirrored': dict(position='HFS', flip_z=False, flip_r=True),
    'FFP-columns-mirrored': dict(position='FFP', flip_r=True),
    'HFP-rows-up': dict(position='HFP', flip_z=False),
}


@pytest.mark.parametrize('case', CASES)
def test_nm_files_read_back_in_the_standard_form(tmp_path, case):
    kwargs = CASES[case]
    betas, projections, radii = _write_nm(tmp_path / 'nm.dcm', **kwargs)
    position = kwargs.get('position', 'FFS')
    if position != 'FFS':
        with pytest.warns(UnverifiedVendorRule, match=position):        # Decision 6: unverified positions warn
            SPECT.read_dicom(str(tmp_path / 'nm.dcm'))
    read, object_meta, proj_meta = _read(tmp_path / 'nm.dcm')
    assert read.shape == (3, len(betas), LR, LZ)
    assert torch.equal(read.cpu(), torch.from_numpy(projections))
    assert np.allclose(proj_meta.angles.cpu().numpy(), betas, atol=1e-4) and np.allclose(proj_meta.radii, radii)
    assert proj_meta.dr == (DR, DZ) and object_meta.dr == (DR, DR, DZ) and object_meta.shape == (LR, LR, LZ)
    assert np.allclose(object_meta.centre, kwargs.get('centre', CENTRE)) and object_meta.frame == 'patient'
    assert [w.ranges_keV for w in proj_meta.energy_windows] == [(r,) for _, r in WINDOWS]
    assert proj_meta.energy_windows[0].photon_keV == pytest.approx(208.4)              # Lu-177's line in the window
    assert np.allclose(proj_meta.frame_duration_s, 15.0) and proj_meta.nuclide == 'Lutetium^177^'
    assert proj_meta.acquisition_start == datetime.datetime(2026, 1, 1, 10, 15, 0, 250000)
    one, _, _ = _read(tmp_path / 'nm.dcm', energy_window=1)                           # the shape depends on the arguments
    assert one.shape == (len(betas), LR, LZ) and torch.equal(one, read[1])


def test_reconstruction_puts_the_markers_back(tmp_path):
    """The whole chain: projections written as a Siemens camera would, read, reconstructed (CPU): every marker within
    one voxel of where it was, in LPS mm."""
    _write_nm(tmp_path / 'nm.dcm', views_per_head=16, step=11.25)
    projections, object_meta, proj_meta = _read(tmp_path / 'nm.dcm', energy_window=0)
    H = SPECTSystemMatrix([], [], object_meta, proj_meta)
    recon = OSEM(PoissonLogLikelihood(H, projections))(n_iters=6, n_subsets=4).cpu().numpy()
    truth = SPECTObjectMeta((DR, DR, DZ), (LR, LR, LZ), centre=CENTRE).affine
    positions = {v: (truth @ [*ijk, 1])[:3] for ijk, v in MARKERS.items()}
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in recon.shape], indexing='ij'), -1)
    world = grid @ object_meta.affine[:3, :3].T + object_meta.affine[:3, 3]
    nearest = np.argmin(np.stack([np.linalg.norm(world - p, axis=-1) for p in positions.values()]), axis=0)
    for n, (value, position) in enumerate(positions.items()):
        region = np.where(nearest == n, recon, -np.inf)
        found = object_meta.affine @ [*np.unravel_index(np.argmax(region), recon.shape), 1]
        assert np.all(np.abs(found[:3] - position) <= np.array([DR, DR, DZ]) + 1e-6), (value, found[:3], position)


def _rigid(theta_deg, translation):
    t = np.radians(theta_deg)
    M = np.eye(4)
    M[:2, :2] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]       # from +x towards +y
    M[:3, 3] = translation
    return M


def test_siemens_registration_moves_the_grid_into_the_ct_frame(tmp_path):
    """Siemens' registration (0043,1002), from the acquisition's frame to the CT's: the reconstruction puts every
    marker where M takes it (within one voxel), so the turn about z has the right sense; the grid's centre is M's image
    of the acquisition's, and the views turn by M's angle. A matrix that is not rigid is left out with a warning."""
    M = _rigid(15.0, (10.0, -20.0, 5.0))       # 15 degrees: the markers 20 to 30 mm off the axis move 5 to 8 mm
    _write_nm(tmp_path / 'nm.dcm', views_per_head=16, step=11.25, registration=M)
    _write_nm(tmp_path / 'plain.dcm', views_per_head=16, step=11.25)
    projections, object_meta, proj_meta = _read(tmp_path / 'nm.dcm', energy_window=0)
    _, plain_meta, plain_proj = _read(tmp_path / 'plain.dcm', energy_window=0)
    assert np.allclose(object_meta.centre, (M @ [*CENTRE, 1])[:3])
    turned = np.sort((plain_proj.angles.cpu().numpy() - 15.0) % 360)
    assert np.allclose(proj_meta.angles.cpu().numpy(), turned, atol=1e-4)
    H = SPECTSystemMatrix([], [], object_meta, proj_meta)
    recon = OSEM(PoissonLogLikelihood(H, projections))(n_iters=6, n_subsets=4).cpu().numpy()
    acquisition = SPECTObjectMeta((DR, DR, DZ), (LR, LR, LZ), centre=CENTRE).affine
    positions = {v: (M @ acquisition @ [*ijk, 1])[:3] for ijk, v in MARKERS.items()}
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in recon.shape], indexing='ij'), -1)
    world = grid @ object_meta.affine[:3, :3].T + object_meta.affine[:3, 3]
    nearest = np.argmin(np.stack([np.linalg.norm(world - p, axis=-1) for p in positions.values()]), axis=0)
    for n, (value, position) in enumerate(positions.items()):
        region = np.where(nearest == n, recon, -np.inf)
        found = object_meta.affine @ [*np.unravel_index(np.argmax(region), recon.shape), 1]
        assert np.all(np.abs(found[:3] - position) <= np.array([DR, DR, DZ]) + 1e-6), (value, found[:3], position)
    sheared = M.copy()
    sheared[0, 1] += 0.1
    _write_nm(tmp_path / 'bad.dcm', registration=sheared)
    with pytest.warns(UserWarning, match='not a rigid'):
        _, bad_meta, _ = SPECT.read_dicom(str(tmp_path / 'bad.dcm'))
    assert np.allclose(bad_meta.centre, CENTRE)
    _write_nm(tmp_path / 'ge.dcm', vendor='GE', n_heads=1, views_per_head=24, starts=(0.0,), per_head_start=False,
              registration=M)                                              # other vendors: not read
    assert np.allclose(_read(tmp_path / 'ge.dcm')[1].centre, CENTRE)


def test_frames_are_counted_and_rotations_refused(tmp_path):
    """A missing frame, a second rotation or a gated file without a time slot is refused, not truncated."""
    _write_nm(tmp_path / 'short.dcm', drop_frames=1)
    with pytest.raises(ValueError, match='frames'):
        _read(tmp_path / 'short.dcm')
    _write_nm(tmp_path / 'two.dcm', rotations=2)
    with pytest.raises(NotImplementedError, match='rotations'):
        _read(tmp_path / 'two.dcm')
    _write_nm(tmp_path / 'gated.dcm', slots=2)
    with pytest.raises(ValueError, match='time_slot'):
        _read(tmp_path / 'gated.dcm')
    first, _, _ = _read(tmp_path / 'gated.dcm', time_slot=0)
    second, _, _ = _read(tmp_path / 'gated.dcm', time_slot=1)
    assert torch.allclose(second, 2 * first)
    _write_nm(tmp_path / 'nostart.dcm', per_head_start=False)
    with pytest.raises(ValueError, match='StartAngle'):
        _read(tmp_path / 'nostart.dcm')


def test_unknown_vendors_warn(tmp_path):
    _write_nm(tmp_path / 'acme.dcm', vendor='ACME')
    with pytest.warns(UnverifiedVendorRule, match='Acme'):
        _, object_meta, _ = SPECT.read_dicom(str(tmp_path / 'acme.dcm'))
    assert np.allclose(object_meta.centre, CENTRE)                # IPP (0, 0): placed as Siemens'


def test_projection_files_are_not_images(tmp_path):
    from pytomography.io import load_image
    _write_nm(tmp_path / 'nm.dcm')
    with pytest.raises(ValueError, match='read_dicom'):
        load_image(tmp_path / 'nm.dcm')


# ---------- several beds ----------

def test_beds_are_read_inferior_first_and_stitched_in_place(tmp_path):
    shift = 12                                                   # slices between the beds' centres; LZ - shift overlap
    lower = (0.3, -6.9, -100.0)
    upper = (0.3, -6.9, -100.0 + shift * DZ)
    ge = dict(vendor='GE', n_heads=1, views_per_head=24, starts=(0.0,), per_head_start=False)
    _write_nm(tmp_path / 'upper.dcm', centre=upper, **ge)
    _write_nm(tmp_path / 'lower.dcm', centre=lower, **ge)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnverifiedVendorRule)
        beds = SPECT.read_dicom_beds([str(tmp_path / 'upper.dcm'), str(tmp_path / 'lower.dcm')])
        alone = SPECT.read_dicom(str(tmp_path / 'lower.dcm'))[0]
    assert [b[1].centre[2] for b in beds] == [lower[2], upper[2]]
    middle = (LZ - shift) // 2                                   # beyond the middle of the overlap: the neighbour's rows
    assert torch.equal(beds[0][0][..., :shift + middle], alone[..., :shift + middle])
    assert torch.equal(beds[0][0][..., shift + middle:], beds[1][0][..., middle:LZ - shift])
    recons = [torch.rand(LR, LR, LZ), torch.rand(LR, LR, LZ)]
    stitched, meta = SPECT.stitch_beds(recons[::-1], [beds[1][1], beds[0][1]])      # any order
    assert meta.shape == (LR, LR, LZ + shift) and stitched.shape == meta.shape
    assert np.allclose(meta.affine @ [0, 0, 0, 1], beds[0][1].affine @ [0, 0, 0, 1])   # starts at the lowest bed
    assert np.allclose(meta.affine @ [0, 0, LZ + shift - 1, 1], beds[1][1].affine @ [0, 0, LZ - 1, 1])
    assert torch.equal(stitched[..., 1:shift], recons[0][..., 1:shift])
    assert torch.equal(stitched[..., LZ:LZ + shift - 1], recons[1][..., LZ - shift:LZ - 1])


# ---------- SIMIND ----------

def _simind(tmp_path, name, version='V8.0', radius='150.0', direction='CCW', start=0.0, views=12, window=(187.2, 228.8),
            nuclide='lu177'):
    """A SIMIND .h00/.a00 pair whose view n (acquisition order) holds the value n + 1 in pixel (r = n, z = 2), stored
    with row 0 at +z."""
    data = np.zeros((views, LZ, LR), np.float32)                 # [view, row (z from the top), column (r)]
    for n in range(views):
        data[n, LZ - 1 - 2, n] = n + 1
    data.tofile(tmp_path / f'{name}.a00')
    radius_key = f';#Radius := {radius}' if version.startswith('V7') else f'Radius := {radius}'
    lines = ['!INTERFILE :=', '!originating system := simind', f'program version := {version}',
             f'!name of data file := {name}.a00', 'imagedata byte order := LITTLEENDIAN',
             f';energy window lower level := {window[0]}', f';energy window upper level := {window[1]}',
             f'!total number of images := {views}', '!number format := short float', '!number of bytes per pixel := 4',
             f'!matrix size [1] := {LR}', f'!matrix size [2] := {LZ}', f'scaling factor (mm/pixel) [1] := {DR}',
             f'scaling factor (mm/pixel) [2] := {DZ}', f';# Radionuclide := {nuclide}', '!extent of rotation := 360',
             f'!number of projections := {views}', '!time per projection (sec) := 1.000', radius_key,
             f'!direction of rotation := {direction}', f'start angle := {start}',
             ';# Collimator hole diameter := 0.294' if version.startswith('V7') else ';# Collimator hole diameter := 2.94',
             ';# Collimator thickness := 4.064' if version.startswith('V7') else ';# Collimator thickness := 40.64',
             ';# Intrinsic FWHM for the camera := 0.36' if version.startswith('V7') else ';# Intrinsic FWHM for the camera := 3.6',
             ';# SIMIND: Photon Energy := 208.000', '!END OF INTERFILE :=']
    (tmp_path / f'{name}.h00').write_text('\n'.join(lines))
    return str(tmp_path / f'{name}.h00')


@pytest.mark.parametrize('version, radius', [('V7.0.3', '15.0'), ('V8.0', '150.0'), ('V10.1', '150.0')])
def test_simind_versions_and_units(tmp_path, version, radius):
    """The version is an integer, so V10 reads its mm keys as mm."""
    header = _simind(tmp_path, 'tot_w1', version=version, radius=radius)
    projections, object_meta, proj_meta = SPECT.read_simind(header)
    assert projections.shape == (1, 12, LR, LZ) and np.allclose(proj_meta.radii, 150.0)
    assert object_meta.dr == (DR, DR, DZ) and object_meta.frame == 'scanner' and proj_meta.units == 'counts/s/MBq'
    assert np.allclose(proj_meta.angles.cpu().numpy(), np.arange(12) * 30.0)
    for n in range(12):                                          # rows flipped: z index 2 from the bottom
        assert projections[0, n, n, 2] == n + 1
    psf = SPECT.simind_psf(header)
    assert psf.sigma_fit_params[1] == pytest.approx(2.94 / (2 * np.sqrt(2 * np.log(2))))   # mm


def test_simind_clockwise_views_are_sorted_and_warn_with_a_start_angle(tmp_path):
    header = _simind(tmp_path, 'cw', direction='CW', start=30.0)
    with pytest.warns(UnverifiedVendorRule, match='clockwise'):
        projections, _, proj_meta = SPECT.read_simind(header, energy_window=0)
    angles = proj_meta.angles.cpu().numpy()
    assert np.all(np.diff(angles) > 0)
    acquired = (30.0 - np.arange(12) * 30.0) % 360                 # view n's angle, acquisition order
    for n, angle in enumerate(acquired):
        assert projections[np.argmin(np.abs(angles - angle)), n, 2] == n + 1


def test_simind_windows_and_organs(tmp_path):
    """Every window's bounds and photon energy (the nuclide's line, not SIMIND's photon energy); organs added up
    with their weights."""
    w1 = _simind(tmp_path, 'tot_w1')
    w4 = _simind(tmp_path, 'tot_w4', window=(101.7, 124.3))
    projections, _, proj_meta = SPECT.read_simind([w1, w4])
    assert [w.photon_keV for w in proj_meta.energy_windows] == [208.4, 112.9]
    assert projections.shape[0] == 2
    organs, _, _ = SPECT.read_simind([[w1, w4]], weights=[2.0, 3.0])
    assert torch.allclose(organs[0], 5 * projections[0])
    assert SPECT.simind_psf(w4).sigma_fit_params[2] > SPECT.simind_psf(w1).sigma_fit_params[2]    # at 112.9 keV


# ---------- energy windows, scatter, collimators ----------

def test_photon_energy_of_a_window():
    assert nuclide_key('99m Technetium') == 'Tc-99m' and nuclide_key('Lutetium^177^') == 'Lu-177'
    assert nuclide_key('Lu177_EM') == 'Lu-177' and nuclide_key('In-111') == 'In-111' and nuclide_key('lu177') == 'Lu-177'
    assert nuclide_key('Window Group A') is None
    assert photon_energy([(187.2, 228.8)], 'Lu-177') == 208.4
    assert photon_energy([(80.0, 104.0), (172.0, 196.0), (280.0, 320.0)], 'Ga-67') == 93.3   # #33: every range
    assert photon_energy([(187.2, 228.8)], None) == pytest.approx(208.0)                      # the centre
    assert photon_energy([(101.7, 124.3)], None, default=208.0) == pytest.approx(113.0)       # SIMIND's 208: outside


def test_energy_window_scatter_takes_the_widths_from_proj_meta():
    windows = [EnergyWindow('peak', [(126.0, 154.0)], 140.5), EnergyWindow('lower', [(114.0, 126.0)], 120.0),
               EnergyWindow('upper', [(154.0, 160.0)], 157.0)]
    proj_meta = SPECTProjMeta((4, 3), (4.0, 4.0), [0.0, 90.0], energy_windows=windows)
    projections = torch.stack([torch.full((2, 4, 3), v) for v in (10.0, 6.0, 3.0)]).to(pytomography.device)
    scatter = SPECT.energy_window_scatter(projections, proj_meta, 0, 1, 2)
    assert torch.allclose(scatter, torch.full_like(scatter, (6 / 12 * 0.5 + 3 / 6 * 0.5) * 28))
    dew = SPECT.energy_window_scatter(projections, proj_meta, 0, 1)
    assert torch.allclose(dew, torch.full_like(dew, 6 / 12 * 0.5 * 28))
    with pytest.raises(ValueError, match='every one'):
        SPECT.energy_window_scatter(projections[0], proj_meta, 0, 1)


def test_square_hole_psf_matches_the_collimator():
    """The 'square' PSF of a square-hole collimator is, along r, the response averaged over the hole lattice: a
    triangle reaching zero at the geometric resolution D (L + d) / L, which is also its FWHM."""
    from pytomography.io.SPECT._energy import _psf_meta
    from pytomography.transforms.SPECT.psf import get_1D_PSF_layer_square
    D, L, d, dx = 2.46, 38.0, 100.0, 0.05
    psf = _psf_meta(D, L, 1e9, 0.0, 'square', 3)                 # no penetration
    s = psf.sigma_fit(np.array([d]), *psf.sigma_fit_params)[0]
    assert s == pytest.approx(D * (L + d) / L)
    # the collimator: a point source at distance d from the front, holes of width D and length L at every position a
    x = np.arange(-400, 401) * dx                                 # detector positions (back face) from the source
    a = np.linspace(-20, 20, 40001)                               # hole centres
    front = x[:, None] * d / (d + L)                              # where the ray to x crosses the front face
    through = (np.abs(x[:, None] - a) <= D / 2) & (np.abs(front - a) <= D / 2)
    response = through.mean(axis=1)
    response /= response.sum()
    layer = get_1D_PSF_layer_square(np.array([s / dx]), int(2 * np.ceil(s / dx) + 1))
    kernel = layer.weight.data.flatten().cpu().numpy()
    kernel = np.pad(kernel, (len(x) - len(kernel)) // 2)
    assert np.abs(kernel - response).max() < 0.02 * response.max()
    half = x[response >= response.max() / 2]
    assert half.max() - half.min() == pytest.approx(s, abs=2 * dx)


def test_collimator_psf_is_in_mm():
    gaussian = SPECT.collimator_psf('SY-ME', 208.4, intrinsic_fwhm_mm=3.8)
    slope, intercept, intrinsic = gaussian.sigma_fit_params
    assert intercept == pytest.approx(2.94 / (2 * np.sqrt(2 * np.log(2))))
    assert intrinsic == pytest.approx(3.8 / (2 * np.sqrt(2 * np.log(2))))
    at_140 = SPECT.collimator_psf('SY-ME', 208.4, intrinsic_fwhm_140keV_mm=3.8).sigma_fit_params[2]
    assert at_140 == pytest.approx(intrinsic * (208.4 / 140) ** -0.5)
    with pytest.raises(ValueError, match='no collimator'):
        SPECT.collimator_psf('XX-NONE', 140.5)
    with pytest.raises(ValueError, match='square'):
        SPECT.collimator_psf('SY-ME', 140.5, shape='box')


def test_subsampled_projection_space():
    proj_meta = SPECTProjMeta((8, 6), (4.0, 4.0), np.arange(12) * 30.0, np.arange(12) + 100.0, frame_duration_s=15.0)
    sub = proj_meta.subsample(2, angle_factor=3, angle_start=1)
    assert sub.shape == (4, 4, 3) and sub.dr == (8.0, 8.0) and np.allclose(sub.radii, [101, 104, 107, 110])
    assert np.allclose(sub.angles.cpu().numpy(), [30, 120, 210, 300]) and len(sub.frame_duration_s) == 4
    assert proj_meta.shape == (12, 8, 6)
    from pytomography.io.SPECT.shared import subsample_projections
    assert subsample_projections(torch.ones(12, 8, 6), 2, 3, 1).shape == sub.shape


def test_patient_position_from_the_codes_only_warns(tmp_path):
    """GE NM files often have no PatientPosition, only the orientation codes (head first, supine): the position is
    read from them for a warning, and nothing is placed by it (the IOPs place the views)."""
    from pytomography.io.SPECT.dicom import patient_position
    _write_nm(tmp_path / 'ffs.dcm')
    ds = pydicom.dcmread(str(tmp_path / 'ffs.dcm'))
    del ds.PatientPosition
    modifier, orientation, gantry = Dataset(), Dataset(), Dataset()
    modifier.CodeMeaning, orientation.CodeMeaning, gantry.CodeMeaning = 'supine', 'recumbent', 'headfirst'
    orientation.PatientOrientationModifierCodeSequence = [modifier]
    ds.PatientOrientationCodeSequence, ds.PatientGantryRelationshipCodeSequence = [orientation], [gantry]
    assert patient_position(ds) == 'HFS'
    ds.save_as(str(tmp_path / 'codes.dcm'))
    with pytest.warns(UnverifiedVendorRule, match='HFS'):
        from_codes = SPECT.read_dicom(str(tmp_path / 'codes.dcm'))
    plain = _read(tmp_path / 'ffs.dcm')
    assert torch.equal(from_codes[0], plain[0]) and np.allclose(from_codes[1].affine, plain[1].affine)
    gantry.CodeMeaning = 'feet-first'
    assert patient_position(ds) == 'FFS'
    ds.PatientPosition = ''
    assert patient_position(ds) == 'FFS'
    del ds.PatientOrientationCodeSequence, ds.PatientGantryRelationshipCodeSequence
    assert patient_position(ds) is None
