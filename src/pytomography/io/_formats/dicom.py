"""DICOM images: series of one slice per file (CT, MR, PT...) and single multi-frame files (NM reconstructions,
PyTomography's own NM output, enhanced multi-frame), each with its full geometry.

The fixes over the readers this replaces: PixelSpacing is (between rows, between columns) = (dy, dx); slices are
sorted along the slice normal, whatever its direction, and non-axial series work; GE CT's corner
ImagePositionPatient is a vendor rule; NM files keep their geometry in the Detector Information Sequence.
"""
from __future__ import annotations

import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from .. import _vendors

# PET units, (0054,1001)
_PET_UNITS = {'BQML': 'Bq/mL', 'CNTS': 'counts', 'GML': 'g/mL', 'PROPCPS': 'propcps', 'NONE': None}
# UCUM codes of a Real World Value Mapping
_UCUM = {'{counts}': 'counts', '{counts}/s': 'counts/s', 'Bq/mL': 'Bq/mL', 'kBq/mL': 'kBq/mL', 'MBq/mL': 'MBq/mL',
         'MBq.s': 'MBq.s', '1': None, '/cm': '1/cm', '1/cm': '1/cm', 'cm-1': '1/cm', '/mm': '1/mm', '1/mm': '1/mm',
         '[hnsf\'U]': 'HU', 'g/ml{SUVbw}': 'g/mL', 'g/mL': 'g/mL'}
# kept in Image.meta when the source has them
_META = ['Modality', 'Manufacturer', 'ManufacturerModelName', 'KVP', 'SeriesDescription', 'SeriesInstanceUID',
         'StudyInstanceUID', 'FrameOfReferenceUID', 'PatientPosition', 'SliceThickness', 'TableHeight',
         'ConvolutionKernel', 'ImageType', 'Units']


def _files(source) -> list[str]:
    """The files of a DICOM source: a folder (searched recursively), a file, or a list of files."""
    if isinstance(source, (str, os.PathLike)):
        source = Path(source)
        if source.is_dir():
            return sorted(str(p) for p in source.rglob('*') if p.is_file())
        return [str(source)]
    return [str(f) for f in source]


def _plain(value):
    """A DICOM value as plain Python (numbers, strings, lists of them)."""
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
        return [_plain(v) for v in value]
    return str(value)


def _meta(ds, files=None) -> dict:
    meta = {key: _plain(ds.get(key)) for key in _META if key in ds}
    if files is not None:
        meta['files'] = list(files)
    return meta


def _float(ds, key, default=None):
    value = ds.get(key)
    return default if value is None or value == '' else float(value)


def _unit_vectors(iop) -> tuple[np.ndarray, np.ndarray]:
    iop = np.asarray([float(v) for v in iop], dtype=np.float64)
    row, col = iop[:3], iop[3:]
    return row / np.linalg.norm(row), col / np.linalg.norm(col)


def _units(ds) -> str | None:
    """The units of a DICOM image's values (after rescaling)."""
    modality = str(ds.get('Modality', '')).upper()
    rwvm = ds.get('RealWorldValueMappingSequence')
    if rwvm and 'MeasurementUnitsCodeSequence' in rwvm[0]:
        code = str(rwvm[0].MeasurementUnitsCodeSequence[0].CodeValue)
        return _UCUM.get(code, code)
    if modality == 'CT':
        rescale_type = str(ds.get('RescaleType', '') or 'HU').upper()
        return 'HU' if rescale_type == 'HU' else None
    if modality == 'PT':
        return _PET_UNITS.get(str(ds.get('Units', '')).upper(), ds.get('Units'))
    return None


def read_series(source) -> dict:
    """A DICOM image as a dict of the :class:`~pytomography.io.Image` fields: a series of one slice per file (a folder
    or a list of files, of one series), or a single multi-frame file (alone, or alone in its folder)."""
    files = _files(source)
    headers = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except (InvalidDicomError, IsADirectoryError, PermissionError):
            continue
        if 'Rows' in ds:                   # an image (not an RT structure set, a report...)
            headers.append((f, ds))
    if not headers:
        raise ValueError(f'no DICOM images in {source}')
    series = defaultdict(list)
    for f, ds in headers:
        series[str(ds.get('SeriesInstanceUID', ''))].append((f, ds))
    if len(series) > 1:
        names = '; '.join(f"{len(v)} files of '{v[0][1].get('SeriesDescription', '')}' ({v[0][1].get('Modality', '')}, "
                          f'SeriesInstanceUID {uid})' for uid, v in series.items())
        raise ValueError(f'{source} holds {len(series)} image series: {names}. Pass the files of one.')
    [members] = series.values()
    if len(members) == 1 and (int(members[0][1].get('NumberOfFrames', 1) or 1) > 1
                              or 'PerFrameFunctionalGroupsSequence' in members[0][1]):
        return read_file(members[0][0])
    return _read_slices(members)


def _read_slices(members: list) -> dict:
    """One slice per file: sorted along the slice normal, the slice step from their positions."""
    first = members[0][1]
    row, col = _unit_vectors(first.ImageOrientationPatient)
    for f, ds in members[1:]:
        r, c = _unit_vectors(ds.ImageOrientationPatient)
        if not (np.allclose(r, row, atol=1e-4) and np.allclose(c, col, atol=1e-4)):
            raise ValueError(f'{f}: its ImageOrientationPatient differs from the rest of the series')
    normal = np.cross(row, col)
    positions = np.array([[float(v) for v in ds.ImagePositionPatient] for _, ds in members])
    along = positions @ normal
    order = np.argsort(along, kind='stable')
    along, positions = along[order], positions[order]
    members = [members[i] for i in order]
    if len(members) > 1:
        gaps = np.diff(along)
        if np.any(gaps < 1e-3):
            raise ValueError('several slices of the series are at the same position (a dynamic, gated or multi-echo '
                             'series?): pass the files of one time point')
        if np.max(np.abs(gaps - gaps.mean())) > max(0.02, 0.01 * gaps.mean()):
            raise ValueError(f'the slices are unevenly spaced ({gaps.min():.4g} to {gaps.max():.4g} mm apart); '
                             'PyTomography reads regular grids only')
        step = (positions[-1] - positions[0]) / (len(members) - 1)      # sheared when the gantry is tilted
    else:
        thickness = _float(first, 'SpacingBetweenSlices', _float(first, 'SliceThickness', 1.0))
        step = normal * thickness
    dy, dx = (float(v) for v in first.PixelSpacing)                    # between rows, between columns
    A = np.eye(4)
    A[:3, 0], A[:3, 1], A[:3, 2], A[:3, 3] = row * dx, col * dy, step, positions[0]
    vendor, modality = _vendors.manufacturer(first), str(first.get('Modality', '')).upper()
    if vendor == 'GE' and modality == 'CT':
        _vendors.use('ge-ct-image-corner')
        A[:3, 3] += 0.5 * dx * row + 0.5 * dy * col
    elif vendor == 'MEDISO' and modality == 'CT' and 'TableHeight' in first:
        _vendors.use('mediso-ct-image', members[0][0])
        A[1, 3] -= float(first.TableHeight)
    slices = []
    for f, _ in members:
        ds = pydicom.dcmread(f)
        values = ds.pixel_array.astype(np.float32)
        slope, intercept = _float(ds, 'RescaleSlope', 1.0), _float(ds, 'RescaleIntercept', 0.0)
        if slope != 1.0 or intercept != 0.0:
            values = values * np.float32(slope) + np.float32(intercept)
        slices.append(values.T)                                         # (rows, columns) = (y, x) -> (x, y)
    data = np.stack(slices, axis=2)
    return dict(data=data, affine=A, units=_units(first), modality=modality or None,
                meta=_meta(first, [f for f, _ in members]), frame='patient')


def read_file(path) -> dict:
    """A 3D image in the frames of one DICOM file: NM (geometry in the Detector Information Sequence), enhanced
    multi-frame (functional groups), or classic multi-frame (top-level geometry)."""
    ds = pydicom.dcmread(str(path))
    modality = str(ds.get('Modality', '')).upper()
    image_type = [str(t).upper() for t in (ds.get('ImageType') or [])]
    if modality == 'NM' and 'RECON' not in ' '.join(image_type) and ('TOMO' in image_type or 'DetectorVector' in ds):
        raise ValueError(f'{path} holds SPECT projections, not an image: read them with io.SPECT.read_dicom')
    frames = ds.pixel_array
    if frames.ndim == 2:
        frames = frames[None]
    n = frames.shape[0]
    shared = (ds.get('SharedFunctionalGroupsSequence') or [None])[0]
    per_frame = ds.get('PerFrameFunctionalGroupsSequence') or []

    def group(item, name):
        return getattr(item, name)[0] if item is not None and name in item else None

    # orientation and pixel spacing
    if 'DetectorInformationSequence' in ds and 'ImageOrientationPatient' in ds.DetectorInformationSequence[0]:
        geometry = ds.DetectorInformationSequence[0]
    else:
        geometry = ds
    iop = geometry.get('ImageOrientationPatient')
    if iop is None and group(shared, 'PlaneOrientationSequence') is not None:
        iop = group(shared, 'PlaneOrientationSequence').ImageOrientationPatient
    if iop is None and per_frame and group(per_frame[0], 'PlaneOrientationSequence') is not None:
        iop = group(per_frame[0], 'PlaneOrientationSequence').ImageOrientationPatient
    if iop is None:
        warnings.warn(f'{path}: no ImageOrientationPatient; the rows are taken to run along +x and the columns along +y')
        iop = [1, 0, 0, 0, 1, 0]
    row, col = _unit_vectors(iop)
    normal = np.cross(row, col)
    spacing = ds.get('PixelSpacing')
    if spacing is None and group(shared, 'PixelMeasuresSequence') is not None:
        spacing = group(shared, 'PixelMeasuresSequence').PixelSpacing
    dy, dx = (float(v) for v in spacing)
    # where the frames are
    positions = [group(f, 'PlanePositionSequence') for f in per_frame]
    order = np.arange(n)
    if positions and all(p is not None for p in positions):
        positions = np.array([[float(v) for v in p.ImagePositionPatient] for p in positions])
        order = np.argsort(positions @ normal, kind='stable')
        positions = positions[order]
        origin = positions[0]
        step = (positions[-1] - positions[0]) / (n - 1) if n > 1 else normal * _float(ds, 'SliceThickness', 1.0)
    else:
        ipp = geometry.get('ImagePositionPatient')
        if ipp is None:
            raise ValueError(f'{path}: no ImagePositionPatient, so the image cannot be placed')
        origin = np.array([float(v) for v in ipp])
        between = _float(ds, 'SpacingBetweenSlices')                    # signed: may run against the normal
        if between is None:
            between = _float(ds, 'SliceThickness', 1.0)
            if n > 1:
                warnings.warn(f'{path}: no frame positions and no SpacingBetweenSlices, so the distance between '
                              f'slices is taken to be SliceThickness ({between} mm)')
        step = normal * between
    A = np.eye(4)
    A[:3, 0], A[:3, 1], A[:3, 2], A[:3, 3] = row * dx, col * dy, step, origin
    # values
    data = frames[order].astype(np.float32)
    units = _units(ds)
    rwvm = ds.get('RealWorldValueMappingSequence')
    transform = group(shared, 'PixelValueTransformationSequence')
    vendor = _vendors.manufacturer(ds)
    if rwvm:
        slope, intercept = float(rwvm[0].RealWorldValueSlope), float(rwvm[0].RealWorldValueIntercept)
    elif 'RescaleSlope' in ds or 'RescaleIntercept' in ds:
        slope, intercept = _float(ds, 'RescaleSlope', 1.0), _float(ds, 'RescaleIntercept', 0.0)
    elif transform is not None:
        slope, intercept = _float(transform, 'RescaleSlope', 1.0), _float(transform, 'RescaleIntercept', 0.0)
    elif vendor == 'GE' and (0x0011, 0x103B) in ds:
        _vendors.use('ge-starguide')
        slope, intercept = float(ds[0x0011, 0x103B].value), 0.0
    elif vendor == 'GE' and (0x0033, 0x1038) in ds:
        _vendors.use('ge-nm-mu-scale', str(path))
        slope, intercept = 1.0 / float(ds[0x0033, 0x1038].value), 0.0
    else:
        slope, intercept = 1.0, 0.0
    if slope != 1.0 or intercept != 0.0:
        data = data * np.float32(slope) + np.float32(intercept)
    return dict(data=np.ascontiguousarray(data.transpose(2, 1, 0)), affine=A, units=units, modality=modality or None,
                meta=_meta(ds, [str(path)]), frame='patient')
