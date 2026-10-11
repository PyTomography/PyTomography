"""The GE StarGuide: 12 swivelling CZT detector heads, each 16 pixels wide, one DICOM NM file per head, with the
head geometry in private tags (see the vendor table, rule 'ge-starguide')."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pydicom
import torch

import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.SPECT import SPECTObjectMeta, StarGuideProjMeta
from .. import _vendors
from .dicom import _acquisition_start, _check_position, _energy_windows, _nuclide

#: The grid of GE's own StarGuide reconstructions: 196 x 196 x 112 voxels of the detector pixel size.
DEFAULT_SHAPE = (196, 196, 112)
_THETA, _X, _Y, _TIME, _SCALE = (0x0099, 0x1053), (0x0099, 0x1051), (0x0099, 0x1052), (0x0009, 0x1003), (0x0011, 0x103B)


def _ct_centre_as_written(files_CT: Sequence[str]) -> np.ndarray:
    """The centre of a CT volume (LPS mm) from its headers, with ImagePositionPatient taken as the first pixel's centre
    as written: GE centres its StarGuide reconstructions on the CT this way (to < 0.01 mm on the tutorial data)."""
    headers = [pydicom.dcmread(str(f), stop_before_pixels=True) for f in files_CT]
    headers = [ds for ds in headers if 'ImagePositionPatient' in ds]
    if not headers:
        raise ValueError('no CT images among files_CT')
    iop = np.array(headers[0].ImageOrientationPatient, float)
    row, col = iop[:3], iop[3:]
    normal = np.cross(row, col)
    positions = sorted((np.array(ds.ImagePositionPatient, float) for ds in headers), key=lambda p: p @ normal)
    dy, dx = (float(v) for v in headers[0].PixelSpacing)
    columns, rows = int(headers[0].Columns), int(headers[0].Rows)
    return (positions[0] + positions[-1]) / 2 + (columns - 1) / 2 * dx * row + (rows - 1) / 2 * dy * col


def _heads(files: Sequence[str]) -> list:
    """The head files of one acquisition, checked."""
    datasets = [pydicom.dcmread(str(f)) for f in files]
    # every head is a series of its own: one acquisition is one study and one series description
    series = {(str(ds.get('StudyInstanceUID', '')), str(ds.get('SeriesDescription', ''))) for ds in datasets}
    if len(series) > 1:
        raise ValueError(f"the files belong to {len(series)} acquisitions ({', '.join(sorted(d for _, d in series))}): "
                         'pass the head files of one')
    for f, ds in zip(files, datasets):
        missing = [hex(t[0]) + ',' + hex(t[1]) for t in (_THETA, _X, _Y, _TIME, _SCALE) if t not in ds]
        if _vendors.manufacturer(ds) != 'GE' or missing:
            raise ValueError(f'{f} is not a StarGuide projection file (missing private tags {missing})')
        n = int(ds.NumberOfFrames)
        for tag in (_THETA, _X, _Y, _TIME):
            if len(ds[tag].value) != n:
                raise ValueError(f'{f}: private tag {tag} has {len(ds[tag].value)} values for {n} frames')
        if len(ds.EnergyWindowVector) != n:
            raise ValueError(f'{f}: EnergyWindowVector has {len(ds.EnergyWindowVector)} values for {n} frames')
    return datasets


def read_starguide(files: Sequence[str], energy_window: int | None = None, *, files_CT: Sequence[str] | None = None,
                   object_meta: ObjectMeta | None = None, nearest_theta: float | None = 0.1):
    """StarGuide projections in PyTomography's standard form.

    * The head geometry comes from private tags (vendor rule 'ge-starguide'): each view's angle θ (used as β), and its
      radius and transaxial offset from the head's x, y (cm, converted to mm); times in ms become seconds.
    * Values are the stored values times the private Pixel Scale.
    * Rows are flipped as the heads' ImageOrientationPatient says (row 0 superior: flipped), so that z runs foot to
      head as for every other reader (the projector is symmetric in z). The patient position is not used; a position
      other than feet-first supine warns, as the private head geometry has been checked on FFS data only.
    * The grid is GE's (196 x 196 x 112 voxels of the detector pixel size) unless ``object_meta`` gives another. It is
      centred on the CT volume when ``files_CT`` are given (frame ``'patient'``, where GE's own reconstruction is),
      else on the scanner (frame ``'scanner'``).

    Args:
        files (Sequence[str]): the head files of one acquisition (12 of them).
        energy_window (int, optional): the energy window to return. Defaults to None: all of them.
        files_CT (Sequence[str], optional): the CT of the acquisition, which places the grid in the patient.
        object_meta (ObjectMeta, optional): the grid to reconstruct on (its voxel size and shape); the reader sets its
            centre and frame.
        nearest_theta (float, optional): the view angles are rounded to this many degrees (views of one angle are
            projected together). Defaults to 0.1; None keeps them as stored.

    Returns:
        (torch.Tensor, SPECTObjectMeta, StarGuideProjMeta): projections [W, view, r, z] (or [view, r, z] for one
        ``energy_window``); the grid; the projection metadata (angles, radii and offsets in mm, frame durations in s,
        every energy window).
    """
    _vendors.use('ge-starguide')
    datasets = _heads(files)
    _check_position(datasets[0], str(files[0]), "PyTomography's reading of StarGuide's private head geometry")
    columns_z = set()
    for f, ds in zip(files, datasets):
        iop = [float(v) for v in ds.DetectorInformationSequence[0].ImageOrientationPatient]
        if abs(iop[5]) < 0.99:
            raise ValueError(f'{f}: the projection rows do not run along z (IOP {iop})')
        columns_z.add(iop[5] > 0)
    if len(columns_z) > 1:
        raise ValueError('the heads disagree on the direction of the projection rows')
    rows_up = columns_z.pop()                                # row 0 inferior: no flip
    window, theta, x, y, times, frames = [], [], [], [], [], []
    for ds in datasets:
        window += list(np.asarray(ds.EnergyWindowVector, dtype=int))
        theta += list(np.asarray(ds[_THETA].value, dtype=np.float64))
        x += list(np.asarray(ds[_X].value, dtype=np.float64) * 10)          # cm to mm
        y += list(np.asarray(ds[_Y].value, dtype=np.float64) * 10)
        times += list(np.asarray(ds[_TIME].value, dtype=np.float64) / 1000)  # ms to s
        frames.append(ds.pixel_array.astype(np.float32) * float(ds[_SCALE].value))
    window, theta, x, y, times = (np.asarray(v) for v in (window, theta, x, y, times))
    frames = np.concatenate(frames)                          # [frame, row = z (superior first), column = r]
    radians = np.radians(theta)
    radii = x * np.sin(radians) + y * np.cos(radians)
    offsets = x * np.cos(radians) - y * np.sin(radians)
    angles = np.round(theta / nearest_theta) * nearest_theta if nearest_theta else theta
    windows = np.unique(window)
    counts = {int(w): int(np.sum(window == w)) for w in windows}
    if len(set(counts.values())) != 1:
        raise ValueError(f'the energy windows have different numbers of views: {counts}')
    first = window == windows[0]
    projections = np.stack([frames[window == w] for w in windows])            # [W, view, z, r]
    if not rows_up:
        projections = projections[:, :, ::-1, :]
    projections = np.ascontiguousarray(np.swapaxes(projections, 2, 3))   # z foot to head: [W, view, r, z]
    projections = torch.tensor(projections).to(pytomography.dtype).to(pytomography.device)
    for w in windows[1:]:      # every window must have the same views
        if not (np.allclose(theta[window == w], theta[first]) and np.allclose(x[window == w], x[first])):
            raise ValueError('the energy windows do not have the same views')
    device = pytomography.device
    ds = datasets[0]
    nuclide = _nuclide(ds)
    proj_meta = StarGuideProjMeta(
        tuple(projections.shape[1:]),
        torch.tensor(angles[first]).to(pytomography.dtype).to(device),
        frame_duration_s=torch.tensor(times[first]).to(pytomography.dtype).to(device),
        offsets=torch.tensor(offsets[first]).to(pytomography.dtype).to(device),
        radii=radii[first],
        energy_windows=_energy_windows(ds, nuclide), acquisition_start=_acquisition_start(ds), nuclide=nuclide,
        units='counts')
    pixel = float(ds.PixelSpacing[0])
    dr, shape = ((pixel,) * 3, DEFAULT_SHAPE) if object_meta is None else (object_meta.dr, object_meta.shape)
    if files_CT is not None:
        centre, frame = _ct_centre_as_written(files_CT), 'patient'
    else:
        centre, frame = (0.0, 0.0, 0.0), 'scanner'
    meta = SPECTObjectMeta(dr, shape, centre=centre, frame=frame, units='counts/s')
    return (projections if energy_window is None else projections[energy_window]), meta, proj_meta
