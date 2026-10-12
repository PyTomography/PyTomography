"""io.load_image: every format, read back with known geometry (design section 7). Each test writes a marker phantom
(three point markers at known LPS positions, one off-centre along each axis, with distinct values, so a flip or a swap
shows), reads it back and checks every marker within 0.01 mm. Then Image.on: averaging, interpolation, labels."""
from __future__ import annotations

import gzip
import warnings

import numpy as np
import pydicom
import pytest
import torch

import pytomography
from pytomography.io import Image, load_image, save_dicom, save_nifti
from pytomography.io._vendors import UnverifiedVendorRule
from pytomography.metadata import ObjectMeta

SHAPE = (12, 10, 8)
MARKERS = {(9, 4, 3): 100.0, (5, 8, 2): 200.0, (3, 2, 6): 300.0}     # off-centre along x, y and z


def _phantom(shape=SHAPE, dtype=np.float32):
    x = np.zeros(shape, dtype=dtype)
    for ijk, v in MARKERS.items():
        x[ijk] = v
    return x


def _truth(A) -> dict:
    return {v: (A @ [*ijk, 1.0])[:3] for ijk, v in MARKERS.items()}


def _check_markers(image: Image, A_true, atol=0.01):
    """Each marker of the image read back is where it was written (LPS mm), and the axes run along +x, +y, +z."""
    for v, position in _truth(A_true).items():
        ijk = np.unravel_index(np.argmin(np.abs(image.data.astype(np.float64) - v)), image.shape)
        assert abs(float(image.data[ijk]) - v) < 0.01 * v, (v, float(image.data[ijk]))
        found = (image.affine @ [*ijk, 1.0])[:3]
        assert np.allclose(found, position, atol=atol), (v, found, position)


def _rotation(axis, degrees):
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    i, j = [k for k in range(3) if k != axis]
    R = np.eye(4)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def _affine(columns, origin):
    A = np.eye(4)
    A[:3, :3] = np.array(columns, float).T
    A[:3, 3] = origin
    return A


AXIAL = _affine([[0.8, 0, 0], [0, 0.8, 0], [0, 0, 2.5]], [-40, -30, -100])
AXIAL_FLIPPED_X = _affine([[-0.8, 0, 0], [0, 0.8, 0], [0, 0, -2.5]], [40, -30, 100])     # normal -z: slices head first
CORONAL = _affine([[0.9, 0, 0], [0, 0, -1.1], [0, 2.0, 0]], [-50, -20, 60])               # rows run inferior
SAGITTAL = _affine([[0, 1.0, 0], [0, 0, -1.2], [3.0, 0, 0]], [-15, -60, 40])
NON_SQUARE = _affine([[0.5, 0, 0], [0, 0.8, 0], [0, 0, 2.0]], [-3, -4, -5])


# ---------- DICOM series ----------

def _series(tmp_path, A, name='s', modality='CT', x=None, **edits):
    x = _phantom() if x is None else x
    files = save_dicom(Image(x, A, modality=modality, units='HU' if modality == 'CT' else 'Bq/mL'), tmp_path / name)
    for f in files:
        ds = pydicom.dcmread(str(f))
        for key, value in edits.items():
            if value is None:
                delattr(ds, key)
            else:
                setattr(ds, key, value)
        ds.save_as(str(f))
    return files


@pytest.mark.parametrize('A', [AXIAL, AXIAL_FLIPPED_X, CORONAL, SAGITTAL, NON_SQUARE],
                         ids=['axial', 'axial-x-flipped-normal-down', 'coronal', 'sagittal', 'non-square-pixels'])
def test_dicom_series_any_orientation(tmp_path, A):
    """PixelSpacing is (dy, dx); slices sort along the normal whatever its direction; coronal and sagittal
    series are read (all were wrong or singular before)."""
    files = _series(tmp_path, A)
    image = load_image(tmp_path / 's')
    _check_markers(image, A)
    assert image.modality == 'CT' and image.units == 'HU' and image.frame == 'patient'
    assert np.all(np.diag(image.affine)[:3] > 0) and np.allclose(image.affine[:3, :3], np.diag(np.diag(image.affine)[:3]))
    assert sorted(image.voxel_size) == pytest.approx(sorted(np.linalg.norm(A[:3, :3], axis=0)))
    # the same from a shuffled list of files
    shuffled = [str(f) for f in files][::-1]
    _check_markers(load_image(shuffled), A)
    # without reorientation the affine keeps the file's directions
    raw = load_image(shuffled, reorient=False)
    _check_markers(raw, A)


def test_dicom_series_ge_ipp_is_the_corner(tmp_path):
    """GE CT writes the corner of the first pixel; the vendor table moves it to the centre (half a pixel)."""
    A = NON_SQUARE
    files = _series(tmp_path, A, Manufacturer='GE MEDICAL SYSTEMS')
    for f in files:                                    # what GE writes: the corner
        ds = pydicom.dcmread(str(f))
        ds.ImagePositionPatient = [float(v) for v in np.array(ds.ImagePositionPatient, float) - [0.25, 0.4, 0]]
        ds.save_as(str(f))
    _check_markers(load_image(tmp_path / 's'), A)
    _series(tmp_path, A, name='siemens', Manufacturer='SIEMENS')      # standard: as written
    _check_markers(load_image(tmp_path / 'siemens'), A)


def test_dicom_series_gantry_tilt_is_sheared(tmp_path):
    """A 5 degree gantry tilt: the slices step along z and y, so the affine is sheared, and every voxel is placed."""
    A = AXIAL.copy()
    files = _series(tmp_path, A)
    shift = np.tan(np.radians(5)) * 2.5
    for k, f in enumerate(files):
        ds = pydicom.dcmread(str(f))
        ds.ImagePositionPatient = [float(v) for v in np.array(ds.ImagePositionPatient, float) + [0, k * shift, 0]]
        ds.save_as(str(f))
    A_tilted = A.copy()
    A_tilted[1, 2] = shift
    image = load_image(tmp_path / 's')
    _check_markers(image, A_tilted)
    assert image.affine[1, 2] == pytest.approx(shift)


def test_dicom_series_refuses_uneven_spacing_and_mixed_series(tmp_path):
    files = _series(tmp_path, AXIAL)
    ds = pydicom.dcmread(str(files[3]))
    ds.ImagePositionPatient = [float(v) for v in np.array(ds.ImagePositionPatient, float) + [0, 0, 0.7]]
    ds.save_as(str(files[3]))
    with pytest.raises(ValueError, match='unevenly'):
        load_image(tmp_path / 's')
    a = _series(tmp_path, AXIAL, name='two')
    b = _series(tmp_path, SAGITTAL, name='other')
    with pytest.raises(ValueError, match='2 image series'):
        load_image([str(f) for f in a + b])


def test_dicom_series_mr_without_rescale(tmp_path):
    """An MR series without rescale tags reads its stored values."""
    x = _phantom()
    _series(tmp_path, AXIAL, x=x, Modality='MR', RescaleSlope=None, RescaleIntercept=None, RescaleType=None,
            SOPClassUID='1.2.840.10008.5.1.4.1.1.4')
    image = load_image(tmp_path / 's')
    _check_markers(image, AXIAL)
    assert image.modality == 'MR' and image.units is None and np.array_equal(image.data, x)


def test_dicom_series_ignores_non_images(tmp_path):
    """A structure set or a report next to the slices is skipped."""
    _series(tmp_path, AXIAL)
    rt = pydicom.Dataset()
    rt.Modality, rt.SeriesInstanceUID, rt.SOPClassUID = 'RTSTRUCT', pydicom.uid.generate_uid(), '1.2.840.10008.5.1.4.1.1.481.3'
    rt.SOPInstanceUID = pydicom.uid.generate_uid()
    rt.file_meta = pydicom.dataset.FileMetaDataset()
    rt.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    rt.save_as(str(tmp_path / 's' / 'rt.dcm'), enforce_file_format=True)
    _check_markers(load_image(tmp_path / 's'), AXIAL)


# ---------- DICOM single file ----------

def test_dicom_single_file_reads_pytomography_nm(tmp_path):
    """PyTomography's own NM output (geometry in the Detector Information Sequence) reads back, units included."""
    for A in (AXIAL, AXIAL_FLIPPED_X):
        [path] = save_dicom(Image(_phantom(), A, modality='NM', units='counts'), tmp_path / str(A[0, 0]))
        image = load_image(path)
        _check_markers(image, A, atol=0.01)
        assert image.modality == 'NM' and image.units == 'counts'


def test_dicom_single_file_negative_spacing(tmp_path):
    """NM with negative SpacingBetweenSlices (GE StarGuide): the frames run against the normal."""
    A = AXIAL
    [path] = save_dicom(Image(_phantom(), A, modality='NM'), tmp_path / 'nm')
    ds = pydicom.dcmread(str(path))
    frames = ds.pixel_array[::-1]                      # top slice first
    ds.PixelData = np.ascontiguousarray(frames).tobytes()
    ds.SpacingBetweenSlices = str(-A[2, 2])
    top = A @ [0, 0, SHAPE[2] - 1, 1]
    ds.DetectorInformationSequence[0].ImagePositionPatient = [float(v) for v in top[:3]]
    ds.save_as(str(path))
    _check_markers(load_image(path), A)


def test_dicom_enhanced_multiframe(tmp_path):
    """Enhanced multi-frame: positions per frame (here in no particular order), orientation, spacing and rescale in
    the shared functional groups."""
    A = NON_SQUARE
    x = _phantom()
    order = np.random.default_rng(1).permutation(SHAPE[2])
    ds = pydicom.Dataset()
    ds.Modality, ds.Rows, ds.Columns, ds.NumberOfFrames = 'CT', SHAPE[1], SHAPE[0], SHAPE[2]
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    shared = pydicom.Dataset()
    po, pm, pvt = pydicom.Dataset(), pydicom.Dataset(), pydicom.Dataset()
    po.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    pm.PixelSpacing = [0.8, 0.5]
    pvt.RescaleSlope, pvt.RescaleIntercept, pvt.RescaleType = 2, -1000, 'HU'
    shared.PlaneOrientationSequence, shared.PixelMeasuresSequence = [po], [pm]
    shared.PixelValueTransformationSequence = [pvt]
    ds.SharedFunctionalGroupsSequence = [shared]
    ds.PerFrameFunctionalGroupsSequence = []
    for k in order:
        position, group = pydicom.Dataset(), pydicom.Dataset()
        position.ImagePositionPatient = [float(v) for v in (A @ [0, 0, k, 1])[:3]]
        group.PlanePositionSequence = [position]
        ds.PerFrameFunctionalGroupsSequence.append(group)
    stored = ((x + 1000) / 2).astype(np.int16)
    ds.PixelData = np.stack([stored[:, :, k].T for k in order]).tobytes()
    ds.SOPClassUID, ds.SOPInstanceUID = '1.2.840.10008.5.1.4.1.1.2.1', pydicom.uid.generate_uid()
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.save_as(str(tmp_path / 'enhanced.dcm'), enforce_file_format=True)
    image = load_image(tmp_path / 'enhanced.dcm')
    _check_markers(image, A)
    assert np.allclose(image.data, x)


def test_dicom_projections_are_refused(tmp_path):
    ds = pydicom.Dataset()
    ds.Modality, ds.ImageType = 'NM', ['ORIGINAL', 'PRIMARY', 'TOMO', 'EMISSION']
    ds.Rows = ds.Columns = 4
    ds.NumberOfFrames = 2
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, 'MONOCHROME2'
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelData = np.zeros((2, 4, 4), np.uint16).tobytes()
    ds.SOPClassUID, ds.SOPInstanceUID = '1.2.840.10008.5.1.4.1.1.20', pydicom.uid.generate_uid()
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.save_as(str(tmp_path / 'p.dcm'), enforce_file_format=True)
    with pytest.raises(ValueError, match='read_dicom'):
        load_image(tmp_path / 'p.dcm')


# ---------- NIfTI ----------

@pytest.mark.parametrize('case', ['oblique-sform', 'qfac-minus-1', 'off-centre', 'qform-only'])
def test_nifti_uses_its_affine(tmp_path, case):
    """The sform (else the qform) places the image, from RAS to LPS."""
    nib = pytest.importorskip('nibabel')
    A_lps = AXIAL.copy()
    A_lps[:3, 3] += [17, -23, 300] if case == 'off-centre' else 0
    if case == 'oblique-sform':
        A_lps = _rotation(2, 20) @ _rotation(0, 7) @ A_lps
    if case == 'qfac-minus-1':
        A_lps = AXIAL_FLIPPED_X
    A_ras = np.diag([-1.0, -1, 1, 1]) @ A_lps
    img = nib.Nifti1Image(_phantom(), A_ras)
    if case == 'qform-only':
        img.set_sform(None, code=0)
        img.set_qform(A_ras, code=1)
    else:
        img.set_sform(A_ras, code=1)
        img.set_qform(A_ras, code=1)
    nib.save(img, str(tmp_path / 'a.nii.gz'))
    image = load_image(tmp_path / 'a.nii.gz')
    _check_markers(image, A_lps)
    assert image.frame == 'patient'


def test_nifti_round_trip_through_save_nifti(tmp_path):
    meta = ObjectMeta(dr=(2, 3, 4), shape=SHAPE, centre=(5, -6, 70), frame='scanner', units='Bq/mL', modality='PT')
    path = save_nifti(_phantom(), tmp_path / 'r', meta)
    image = load_image(path)
    _check_markers(image, meta.affine)
    assert image.units == 'Bq/mL' and image.frame == 'scanner'
    assert np.allclose(image.object_meta.affine, meta.affine)


@pytest.mark.parametrize('frame', ['patient', 'scanner'])
def test_nifti_frame_from_its_xform_code(tmp_path, frame):
    """save_nifti writes sform code 1 for frame 'patient' and 2 for 'scanner', and load_image reads them back as such;
    frame= overrides what the file says."""
    nib = pytest.importorskip('nibabel')
    meta = ObjectMeta(dr=2.0, shape=SHAPE, centre=(1, 2, 3), frame=frame, units='Bq/mL', modality='PT')
    path = save_nifti(_phantom(), tmp_path / 'r', meta)
    assert int(nib.load(str(path)).get_sform(coded=True)[1]) == (1 if frame == 'patient' else 2)
    assert load_image(path).frame == frame
    other = 'scanner' if frame == 'patient' else 'patient'
    assert load_image(path, frame=other).frame == other
    with pytest.raises(ValueError, match='frame'):
        load_image(path, frame='world')


# ---------- Interfile ----------

def _interfile(tmp_path, name, data_lps, header_lines, dtype='<f4', offset=0, layout=None):
    """An Interfile image whose stored array (x fastest) is ``data_lps`` after the SIMIND layout's flips."""
    stored = data_lps
    if layout == 'mu':
        stored = stored[:, :, ::-1]
    elif layout == 'mu-castor':
        stored = stored[:, ::-1, ::-1]
    raw = np.ascontiguousarray(stored.transpose(2, 1, 0)).astype(dtype).tobytes()
    (tmp_path / f'{name}.raw').write_bytes(b'\0' * offset + raw)
    lines = ['!INTERFILE :='] + header_lines + [f'!name of data file := {name}.raw', f'data offset in bytes := {offset}',
                                                '!END OF INTERFILE :=']
    (tmp_path / f'{name}.hdr').write_text('\n'.join(lines))
    return tmp_path / f'{name}.hdr'


def _centred(dr, shape=SHAPE):
    return ObjectMeta(dr=dr, shape=shape).affine


SIMIND = ['!originating system := simind', 'program version := V8.0', '!matrix size [1] := 12', '!matrix size [2] := 10',
          '!matrix size [3] := 8', 'scaling factor (mm/pixel) [1] := 2.0', 'scaling factor (mm/pixel) [2] := 2.0',
          'scaling factor (mm/pixel) [3] := 3.0', '!number format := short float', '!number of bytes per pixel := 4',
          'imagedata byte order := LITTLEENDIAN', ';# Units of data (ECT) := mu',
          '; matrix size [1] := 999']                                  # a plain comment is not a key
SIEMENS = ['%SMS-MI header name space:=image subheader', 'image data byte order:=BIGENDIAN', 'number format:=signed integer',
           '!number of bytes per pixel:=2', 'matrix size[1]:=12', 'matrix size[2]:=10', 'matrix size[3]:=8',
           'scale factor (mm/pixel) [1]:=2.0', 'scale factor (mm/pixel) [2]:=2.0', 'scale factor (mm/pixel) [3]:=3.0',
           'quantification units:=1/cm']
STIR = ['!number format := float', '!number of bytes per pixel := 4', 'imagedata byte order := LITTLEENDIAN',
        '!matrix size [1] := 12', '!matrix size [2] := 10', '!matrix size [3] := 8', 'scaling factor (mm/pixel) [1] := 2.0',
        'scaling factor (mm/pixel) [2] := 2.0', 'scaling factor (mm/pixel) [3] := 3.0',
        'first pixel offset (mm) [1] := -12', 'first pixel offset (mm) [2] := -10', 'first pixel offset (mm) [3] := 0']


@pytest.mark.parametrize('layout', ['mu', 'mu-castor'])
def test_interfile_simind_layouts(tmp_path, layout):
    """Both SIMIND layouts of an asymmetric map read back the same LPS map; the layout is required."""
    path = _interfile(tmp_path, 'amap', _phantom(), SIMIND, layout=layout)
    image = load_image(path, layout=layout)
    _check_markers(image, _centred((2, 2, 3)))
    assert image.units == '1/cm' and image.frame == 'scanner'
    with pytest.raises(ValueError, match='layout'):
        load_image(path)


def test_interfile_siemens_big_endian_int16_with_offset(tmp_path):
    path = _interfile(tmp_path, 'mu', _phantom(), SIEMENS, dtype='>i2', offset=256)
    image = load_image(path)
    _check_markers(image, _centred((2, 2, 3)))
    assert image.units == '1/cm'


def test_interfile_gate_stir_is_centred_and_warns(tmp_path):
    """GATE/STIR .hv: no SIMIND flips; the offsets are kept, not used; the rule is unverified, so it warns."""
    path = _interfile(tmp_path, 'umap', _phantom(), STIR)
    with pytest.warns(UnverifiedVendorRule):
        image = load_image(path, units='1/cm')
    _check_markers(image, _centred((2, 2, 3)))
    assert image.units == '1/cm' and image.meta['first_pixel_offset_mm'] == [-12, -10, 0]


def test_interfile_missing_key_raises(tmp_path):
    """A missing key raises, instead of giving False."""
    path = _interfile(tmp_path, 'bad', _phantom(), [l for l in STIR if 'scaling factor (mm/pixel) [3]' not in l])
    with pytest.raises(KeyError, match='scaling factor'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UnverifiedVendorRule)
            load_image(path)


# ---------- NRRD ----------

@pytest.mark.parametrize('space', ['left-posterior-superior', 'right-anterior-superior'])
@pytest.mark.parametrize('encoding', ['raw', 'gzip'])
def test_nrrd(tmp_path, space, encoding):
    A_lps = _rotation(2, 10) @ CORONAL
    to_space = np.diag([1.0, 1, 1]) if space.startswith('left') else np.diag([-1.0, -1, 1])
    directions = to_space @ A_lps[:3, :3]
    origin = to_space @ A_lps[:3, 3]
    x = _phantom(dtype=np.int16)
    raw = np.ascontiguousarray(x.transpose(2, 1, 0)).astype('<i2').tobytes()
    if encoding == 'gzip':
        raw = gzip.compress(raw)
    vec = lambda v: '(' + ','.join(f'{c:.10g}' for c in v) + ')'
    header = ['NRRD0004', 'type: short', 'dimension: 3', f'space: {space}', 'sizes: 12 10 8',
              'space directions: ' + ' '.join(vec(directions[:, a]) for a in range(3)), 'kinds: domain domain domain',
              'endian: little', f'encoding: {encoding}', f'space origin: {vec(origin)}',
              'Segment0_Name:=liver', 'Segment0_LabelValue:=100', 'Segment1_Name:=spleen', 'Segment1_LabelValue:=200']
    (tmp_path / 'seg.seg.nrrd').write_bytes(('\n'.join(header) + '\n\n').encode() + raw)
    image = load_image(tmp_path / 'seg.seg.nrrd')
    _check_markers(image, A_lps)
    assert image.labels == {'liver': 100, 'spleen': 200} and image.units == 'label'
    assert image.frame == 'patient'                       # no pytomography_frame field


def _nrrd_with_frame(path, frame):
    header = ['NRRD0004', 'type: short', 'dimension: 3', 'space: left-posterior-superior', 'sizes: 12 10 8',
              'space directions: (2,0,0) (0,2,0) (0,0,3)', 'kinds: domain domain domain', 'endian: little',
              'encoding: raw', 'space origin: (-11,-9,-10.5)', 'Segment0_Name:=sphere', 'Segment0_LabelValue:=100']
    if frame is not None:
        header.append(f'pytomography_frame:={frame}')
    raw = np.ascontiguousarray(_phantom(dtype=np.int16).transpose(2, 1, 0)).astype('<i2').tobytes()
    path.write_bytes(('\n'.join(header) + '\n\n').encode() + raw)
    return path


@pytest.mark.parametrize('frame', ['scanner', 'patient', None])
def test_nrrd_frame_field(tmp_path, frame):
    """A NRRD says its frame with the key-value field pytomography_frame:= (the GE sphere mask, drawn on a grid with no
    patient frame, says 'scanner'); without it the image is in the patient's frame. frame= overrides either."""
    path = _nrrd_with_frame(tmp_path / 'mask.seg.nrrd', frame)
    image = load_image(path)
    _check_markers(image, _centred((2, 2, 3)))
    assert image.frame == (frame or 'patient')
    assert load_image(path, frame='scanner').frame == 'scanner'
    with warnings.catch_warnings():
        warnings.simplefilter('error')                     # same frame: no warning
        image.on(ObjectMeta(dr=2.0, shape=SHAPE, frame=frame or 'patient'))


def test_nrrd_frame_field_must_be_a_frame(tmp_path):
    with pytest.raises(ValueError, match='pytomography_frame'):
        load_image(_nrrd_with_frame(tmp_path / 'mask.nrrd', 'gantry'))


# ---------- Image.on ----------

def test_on_averages_when_downsampling():
    """A 1 mm image on a 4 mm grid is the mean over each 4 mm voxel, not a point sample."""
    rng = np.random.default_rng(0)
    data = rng.random((16, 16, 16)).astype(np.float32)
    source = Image(data, ObjectMeta(dr=1.0, shape=(16, 16, 16)).affine, frame="scanner")
    out = source.on(ObjectMeta(dr=4.0, shape=(4, 4, 4))).cpu().numpy()
    expected = data.reshape(4, 4, 4, 4, 4, 4).mean(axis=(1, 3, 5))
    assert np.allclose(out, expected, atol=1e-6)


def test_on_interpolates_linearly_and_fills_outside():
    data = np.fromfunction(lambda i, j, k: 2 * i + 3 * j - k, (10, 10, 10)).astype(np.float32)
    source = Image(data, ObjectMeta(dr=2.0, shape=(10, 10, 10)).affine, frame="scanner")
    target = ObjectMeta(dr=(1.0, 2.0, 2.0), shape=(16, 10, 10), centre=(0.3, 0, 0))
    out = source.on(target, fill=-7).cpu().numpy()
    i = (np.arange(16) - 7.5) * 1.0 + 0.3                     # target x in mm
    u = i / 2.0 + 4.5                                          # source index
    inside = (u >= 0) & (u <= 9)
    assert np.allclose(out[inside, 3, 4], 2 * u[inside] + 3 * 3 - 4, atol=1e-4)
    far = Image(data, ObjectMeta(dr=2.0, shape=(10, 10, 10), centre=(500, 0, 0)).affine, frame="scanner")
    assert torch.all(far.on(target, fill=-7) == -7)


def test_on_nearest_labels_and_threshold():
    labels = np.zeros((8, 8, 8), np.uint8)
    labels[2:6, 2:6, 2:6] = 3
    source = Image(labels, ObjectMeta(dr=1.0, shape=(8, 8, 8)).affine, labels={'sphere': 3}, frame='scanner')
    coarse = ObjectMeta(dr=2.0, shape=(4, 4, 4))
    coverage = source.on(coarse, label='sphere').cpu().numpy()
    assert np.allclose(coverage[1:3, 1:3, 1:3], 1) and coverage.sum() == pytest.approx(64 / 8)
    shifted = ObjectMeta(dr=2.0, shape=(4, 4, 4), centre=(1, 0, 0))     # half a coarse voxel: half covered
    assert source.on(shifted, label=3).cpu().numpy()[2, 1, 1] == pytest.approx(0.5)
    mask = source.on(shifted, label='sphere', threshold=0.5)
    assert mask.dtype == torch.bool and int(mask.sum()) == 3 * 2 * 2      # x: half, whole, half
    nearest = source.on(ObjectMeta(dr=1.0, shape=(8, 8, 8)), mode='nearest', label=3).cpu().numpy()
    assert np.array_equal(nearest, (labels == 3).astype(np.float32))
    with pytest.raises(KeyError):
        source.on(coarse, label='liver')


def test_on_oblique_image_matches_the_aligned_one():
    """An image rotated by a small angle (a vendor's 0.5 degree registration) goes through the general path; on a
    smooth image it agrees with the aligned one."""
    data = np.fromfunction(lambda i, j, k: np.sin(i / 5) + np.cos(j / 7) + k / 10, (40, 40, 20)).astype(np.float32)
    A = ObjectMeta(dr=(1.0, 1.0, 2.0), shape=(40, 40, 20)).affine
    target = ObjectMeta(dr=4.0, shape=(8, 8, 8))
    aligned = Image(data, A, frame='scanner').on(target).cpu().numpy()
    oblique = Image(data, _rotation(2, 0.5) @ A, frame='scanner').on(target).cpu().numpy()
    assert np.abs(oblique - aligned)[2:6, 2:6, 2:6].max() < 0.05


def test_on_works_in_chunks_whatever_the_budget():
    rng = np.random.default_rng(3)
    data = rng.random((30, 30, 40)).astype(np.float32)
    source = Image(data, ObjectMeta(dr=(1.0, 1.0, 1.0), shape=data.shape).affine, frame="scanner")
    target = ObjectMeta(dr=(2.5, 2.5, 3.0), shape=(12, 12, 13), centre=(0.4, -0.3, 0.2))
    whole = source.on(target)
    old = pytomography.memory_budget
    try:
        pytomography.set_memory_budget(1e-5)                   # 10 kB: one target slice at a time
        chunked = source.on(target)
    finally:
        pytomography.memory_budget = old
    assert torch.allclose(whole, chunked, atol=1e-6)


def test_on_warns_across_frames():
    source = Image(_phantom(), _centred(2.0), frame='scanner')
    with pytest.warns(UserWarning, match='frame'):
        source.on(ObjectMeta(dr=2.0, shape=SHAPE, frame='patient'))


def test_reorientation_only_permutes_and_flips():
    A = _affine([[0, 0, -2.0], [1.5, 0, 0], [0, -3.0, 0]], [10, 20, 30])
    raw = Image(_phantom(), A)
    from pytomography.io._image import reorient_to_lps
    image = reorient_to_lps(raw)
    assert np.allclose(image.affine[:3, :3], np.diag([1.5, 3.0, 2.0]))
    _check_markers(image, A)
    assert np.array_equal(np.sort(image.data, axis=None), np.sort(raw.data, axis=None))


# ---------- the tutorial data ----------

def _data(data_dir, *path):
    target = data_dir.joinpath(*path)
    if not target.exists():
        pytest.skip(f'{"/".join(path)} is not in the tutorial data folder')
    return target


@pytest.mark.data
def test_ge_ct_centre_is_round_with_the_corner_rule(data_dir):
    """GE's Optima CT540 (Lu177-PSMA-GEDisc) is centred on round numbers only with the corner rule."""
    ct = load_image(_data(data_dir, 'SPECT', 'Lu177-PSMA-GEDisc', 'CT'))
    assert ct.modality == 'CT' and ct.units == 'HU' and ct.shape[:2] == (512, 512)
    assert np.allclose(ct.centre[:2], 0, atol=0.01)


@pytest.mark.data
def test_vendor_nm_reconstructions(data_dir):
    """Single-file NM reconstructions: GE StarGuide (frames head to foot, values times the private Pixel Scale) and
    Siemens (an oblique 0.5 degree registration, kept in the affine)."""
    path = _data(data_dir, 'SPECT', 'Tc99m-NEMA-Starguide', 'vendor_recon', 'i196884.NMDC.1')
    ds = pydicom.dcmread(str(path))
    ge = load_image(path)
    expected = np.transpose(ds.pixel_array * float(ds[0x0011, 0x103B].value), (2, 1, 0))[:, :, ::-1]
    assert np.allclose(ge.data, expected, rtol=1e-6)
    assert np.allclose(ge.affine[:3, :3], np.diag([2.46] * 3)) and np.allclose(ge.centre, (-0.49, -0.49, -0.5), atol=1e-3)
    siemens = load_image(_data(data_dir, 'SPECT', 'Lu177-NEMA-SymT2', 'scanner_recon.dcm'))
    assert siemens.shape == (128, 128, 128) and siemens.modality == 'NM'
    angle = np.degrees(np.arccos(siemens.affine[0, 0] / np.linalg.norm(siemens.affine[:3, 0])))
    assert 0.3 < angle < 0.8                                   # the registration rotation is kept


@pytest.mark.data
def test_interfile_and_nrrd_tutorial_files(data_dir):
    mu = load_image(_data(data_dir, 'PET', 'PETSIRD-mMR-NEMA', '20170809_NEMA_MUMAP_UCL.v.hdr'))
    assert mu.units == '1/cm' and mu.shape == (344, 344, 127) and 0.09 < float(np.percentile(mu.data, 99.9)) < 0.11
    seg = load_image(_data(data_dir, 'PET', 'GE-DMI-NEMA', 'fdg_spheres.seg.nrrd'))
    assert seg.labels == {f'Segment_{n}': n for n in range(1, 9)} and set(np.unique(seg.data)) <= set(range(9))
    # drawn on the GE grid (192 x 192 x 71, 2.78 mm, centred on the scanner's axis, no patient frame)
    assert seg.frame == 'scanner' and np.allclose(seg.centre, 0.0, atol=1e-6) and np.allclose(seg.voxel_size, 2.78)
    xcat = load_image(_data(data_dir, 'SPECT', 'SIMIND-MultiOrgan', 'multi_projections', 'mu208.hct'), layout='mu-castor')
    assert xcat.units == '1/cm' and xcat.voxel_size == (3.0, 3.0, 3.0)


def test_vendor_table():
    from pytomography.io import _vendors
    for rule_id in ('ge-dmi-listmode', 'gate-world-frame', 'stir2petsird', 'ge-ct-pd-central-column',
                    'siemens-ct-pd-y-origin', 'nikon-xtek', 'ge-ct-image-corner', 'simind-density-layout'):
        assert rule_id in _vendors.RULES
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _vendors.use('ge-ct-image-corner').status == 'verified'          # verified: silent
    with pytest.warns(UnverifiedVendorRule, match='file.dcm'):
        _vendors.use('mediso-ct-image', 'file.dcm')
    ds = pydicom.Dataset()
    for name, vendor in (('GE MEDICAL SYSTEMS', 'GE'), ('GE Healthcare', 'GE'), ('SIEMENS', 'SIEMENS'),
                         ('Siemens Healthineers', 'SIEMENS'), ('MEDISO', 'MEDISO'), ('Acme', 'ACME')):
        ds.Manufacturer = name
        assert _vendors.manufacturer(ds) == vendor
    assert _vendors.manufacturer(pydicom.Dataset()) == ''
    assert {row['status'] for row in _vendors.table()} <= set(_vendors.STATUSES)


def test_on_labels_that_overlap():
    """A label made of several values (each overlap of labels has a value of its own): its coverage counts every
    one of them, by value or by name."""
    data = np.zeros((8, 8, 8), np.uint8)
    data[1:5, 2:6, 2:6] = 1                      # only 'liver'
    data[5:7, 2:6, 2:6] = 3                      # 'liver' and 'tumour'
    data[7, 2:6, 2:6] = 2                        # only 'tumour'
    source = Image(data, ObjectMeta(dr=1.0, shape=(8, 8, 8)).affine, labels={'liver': (1, 3), 'tumour': [2, 3]},
                   frame='scanner')
    grid = ObjectMeta(dr=1.0, shape=(8, 8, 8))
    liver = source.on(grid, label='liver').cpu().numpy()
    assert np.array_equal(liver, np.isin(data, (1, 3)).astype(np.float32))
    assert np.array_equal(source.on(grid, label=(2, 3), mode='nearest').cpu().numpy(), np.isin(data, (2, 3)).astype(np.float32))
    assert source.on(ObjectMeta(dr=2.0, shape=(4, 4, 4)), label='tumour').cpu().numpy().sum() == pytest.approx(48 / 8)
    assert int(source.on(grid, label=[1, 3], threshold=0.5).sum()) == 6 * 4 * 4
