"""The DICOM helpers place slices by the distance between them, not by SliceThickness (#272): the two differ when slices
overlap or have gaps, as in TCIA LDCT-and-Projection-data C145 (1.25 mm thick, 1 mm apart)."""
from __future__ import annotations

import numpy as np
import pydicom
import pytest

from pytomography.io.shared import _get_affine_single_file
from pytomography.io.shared.dicom import compute_slice_thickness_multifile
from pytomography.io.SPECT.dicom import get_starguide_affine_CT


def _save(ds, path):
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    try:
        ds.save_as(path, enforce_file_format=True)
    except TypeError:                                                     # pydicom < 3
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(path, write_like_original=False)


def _pixels(ds, rows=6, cols=5, frames=None):
    ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.PhotometricInterpretation = rows, cols, 1, 'MONOCHROME2'
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    if frames is not None:
        ds.NumberOfFrames = frames
    ds.PixelData = np.zeros((rows, cols) if frames is None else (frames, rows, cols), np.uint16).tobytes()


def _single_file(path, thickness, spacing=None, frame_z=None, frames=4):
    """One file of ``frames`` frames of 2 mm pixels, with SliceThickness, SpacingBetweenSlices if ``spacing`` is given,
    and the frames' positions (Per-frame Functional Groups) if ``frame_z`` is given."""
    ds = pydicom.Dataset()
    ds.Modality = 'NM'
    ds.PixelSpacing = [2.0, 2.0]
    ds.SliceThickness = thickness
    if spacing is not None:
        ds.SpacingBetweenSlices = spacing
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [-4.0, -5.0, -100.0]
    if frame_z is not None:
        ds.PerFrameFunctionalGroupsSequence = []
        for z in frame_z:
            position = pydicom.Dataset()
            position.ImagePositionPatient = [-4.0, -5.0, z]
            group = pydicom.Dataset()
            group.PlanePositionSequence = [position]
            ds.PerFrameFunctionalGroupsSequence.append(group)
    _pixels(ds, frames=frames)
    _save(ds, str(path))
    return str(path)


def test_the_single_file_affine_uses_the_spacing_between_slices(tmp_path):
    # slices 1.25 mm thick, 1 mm apart, like C145's
    M = _get_affine_single_file(_single_file(tmp_path / 'overlap.dcm', thickness=1.25, spacing=1.0))
    assert M[2, 2] == pytest.approx(1.0) and np.allclose(M[:3, 3], [-4.0, -5.0, -100.0])
    # PyTomography's own exports write the two tags equal: the affine is what it was
    M = _get_affine_single_file(_single_file(tmp_path / 'equal.dcm', thickness=4.4196, spacing=4.4196))
    assert M[2, 2] == pytest.approx(4.4196) and M[0, 0] == M[1, 1] == 2.0


def test_the_frame_positions_give_the_spacing_and_its_direction(tmp_path):
    path = _single_file(tmp_path / 'frames.dcm', thickness=1.25, frame_z=[-100.0, -101.0, -102.0, -103.0])
    M = _get_affine_single_file(path)
    assert M[2, 2] == pytest.approx(-1.0) and M[2, 3] == pytest.approx(-100.0)


def test_the_slice_thickness_is_the_last_resort_with_a_warning(tmp_path):
    path = _single_file(tmp_path / 'thickness.dcm', thickness=2.5)
    with pytest.warns(UserWarning, match='SliceThickness'):
        M = _get_affine_single_file(path)
    assert M[2, 2] == pytest.approx(2.5)


def test_the_starguide_ct_affine_uses_the_slice_positions(tmp_path):
    files = []
    for k in range(4):                                                    # 1 mm apart, 1.25 mm thick
        ds = pydicom.Dataset()
        ds.Modality = 'CT'
        ds.PixelSpacing = [0.9765625, 0.9765625]
        ds.SliceThickness = 1.25
        ds.ImagePositionPatient = [-250.0, -250.0, -20.0 + k]
        _pixels(ds)
        files.append(str(tmp_path / f'{k}.dcm'))
        _save(ds, files[-1])
    assert compute_slice_thickness_multifile(files) == pytest.approx(1.0)
    affine = get_starguide_affine_CT(files)
    assert affine[2, 2] == pytest.approx(0.1)                             # cm
    assert affine[2, 3] == pytest.approx(-1.5 * 0.1)                      # centred: -(N - 1) dz / 2
