"""SPECT projections stored as DICOM NM (dual-head cameras: Siemens, GE, Mediso...), one or several bed positions.

:func:`read_dicom` turns a file into PyTomography's standard form: projections [W, θ, r, z] in counts, z rows foot to
head, with β = (DICOM angle + 180) mod 360; an object grid (``SPECTObjectMeta``, mm) placed in the patient by the
vendor rules of :mod:`pytomography.io._vendors`; and the acquisition facts in ``proj_meta`` (every energy window,
the frame duration, the start time, the nuclide).
"""
from __future__ import annotations

import collections.abc
import datetime
import warnings
from typing import Sequence

import numpy as np
import pydicom
import torch
from pydicom.dataset import Dataset

import pytomography
from pytomography.metadata.SPECT import EnergyWindow, SPECTObjectMeta, SPECTProjMeta
from pytomography.utils import compute_EW_scatter
from .. import _vendors
from ._energy import fov_mask, photon_energy


# ---------- the geometry and the acquisition facts, from the header ----------

def _vector(ds: Dataset, key: str, n: int) -> np.ndarray:
    return np.asarray(ds[key].value, dtype=int).ravel() if key in ds else np.ones(n, dtype=int)


def _is_starguide(ds: Dataset) -> bool:
    return _vendors.manufacturer(ds) == 'GE' and (0x0099, 0x1053) in ds


def _energy_windows(ds: Dataset, nuclide: str | None) -> list[EnergyWindow]:
    """Every energy window of a NM file, with all of its ranges (#33) and its photon energy."""
    windows = []
    for i, window in enumerate(ds.get('EnergyWindowInformationSequence') or []):
        ranges = [(float(r.EnergyWindowLowerLimit), float(r.EnergyWindowUpperLimit))
                  for r in window.get('EnergyWindowRangeSequence') or []]
        name = str(window.get('EnergyWindowName', '') or f'window {i}')
        windows.append(EnergyWindow(name, ranges, photon_energy(ranges, nuclide or name)))
    return windows


def _nuclide(ds: Dataset) -> str | None:
    for item in ds.get('RadiopharmaceuticalInformationSequence') or []:
        for code in item.get('RadionuclideCodeSequence') or []:
            if code.get('CodeMeaning'):
                return str(code.CodeMeaning)
    return None


def _acquisition_start(ds: Dataset) -> datetime.datetime | None:
    text = str(ds.get('AcquisitionDateTime', '') or '')
    if not text and ds.get('AcquisitionDate'):
        text = str(ds.AcquisitionDate) + str(ds.get('AcquisitionTime', '') or '')
    text = text.strip()
    for fmt in ('%Y%m%d%H%M%S.%f', '%Y%m%d%H%M%S', '%Y%m%d%H%M', '%Y%m%d'):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _projection_centre(ds: Dataset, Lr: int, Lz: int, dr: float, dz: float, col: np.ndarray, source: str) -> tuple:
    """Where the rotation axis and the middle projection row are in the patient (LPS mm), by the vendor table."""
    ipp = np.asarray(ds.DetectorInformationSequence[0].ImagePositionPatient, dtype=np.float64)
    z = ipp[2] + (Lz - 1) / 2 * dz * np.sign(col[2])          # the middle row, from the first row's position
    vendor = _vendors.manufacturer(ds)
    table_height = float(ds.RotationInformationSequence[0].get('TableHeight', 0) or 0)
    if vendor == 'SIEMENS':
        _vendors.use('siemens-nm-projection', source)
        return (ipp[0], -table_height, z)
    if vendor == 'GE':
        _vendors.use('ge-nm-projection-ipp', source)
        return (ipp[0] + (Lr - 1) / 2 * dr, ipp[1] + (Lr - 1) / 2 * dr, z)
    if vendor == 'MEDISO':
        _vendors.use('mediso-nm-projection', source)
        return (ipp[0] + (Lr - 1) / 2 * dr, 0.0, z)
    siemens_like = ipp[0] == 0 and ipp[1] == 0
    warnings.warn(f"{source}: no vendor rule for '{ds.get('Manufacturer', '')}' NM projections; the rotation axis is "
                  'placed as for ' + ('Siemens (at y = -TableHeight)' if siemens_like else
                                      'GE (ImagePositionPatient is the centre of voxel (0, 0))'),
                  _vendors.UnverifiedVendorRule)
    if siemens_like:
        return (0.0, -table_height, z)
    return (ipp[0] + (Lr - 1) / 2 * dr, ipp[1] + (Lr - 1) / 2 * dr, z)


def _geometry(ds: Dataset, source: str = '') -> dict:
    """Everything about a NM TOMO file but its pixels: which frame is which (window, time slot, view), the angle,
    radius and orientation of every view, the grid and the acquisition facts."""
    image_type = ' '.join(str(t).upper() for t in (ds.get('ImageType') or []))
    if str(ds.get('Modality', '')).upper() != 'NM' or 'RECON' in image_type:
        raise ValueError(f'{source} is not a file of SPECT projections (Modality NM, ImageType TOMO); an image is '
                         'read with io.load_image')
    if _is_starguide(ds):
        raise ValueError(f'{source} is a GE StarGuide head file: read the 12 files of the acquisition with '
                         'io.SPECT.read_starguide')
    n = int(ds.get('NumberOfFrames', 1) or 1)
    window, detector = _vector(ds, 'EnergyWindowVector', n), _vector(ds, 'DetectorVector', n)
    rotation, view, slot = _vector(ds, 'RotationVector', n), _vector(ds, 'AngularViewVector', n), _vector(ds, 'TimeSlotVector', n)
    if any(len(v) != n for v in (window, detector, rotation, view, slot)):
        raise ValueError(f'{source}: the frame vectors do not have NumberOfFrames ({n}) entries')
    if len(np.unique(rotation)) > 1:
        raise NotImplementedError(f'{source} holds {len(np.unique(rotation))} rotations (a dynamic acquisition, or a '
                                  'StarGuide head file); PyTomography reads one rotation per file (#26)')
    rotations = ds.RotationInformationSequence
    rot = rotations[0]
    views_per_detector = int(rot.NumberOfFramesInRotation)
    detectors = np.unique(detector)
    windows, slots = np.unique(window), np.unique(slot)
    expected = len(windows) * len(slots) * len(detectors) * views_per_detector
    if expected != n:
        raise ValueError(f'{source}: {n} frames, but {len(windows)} windows x {len(slots)} time slots x '
                         f'{len(detectors)} detectors x {views_per_detector} views = {expected}')
    step = float(rot.AngularStep)
    direction = 1.0 if str(rot.RotationDirection).upper() in ('CC', 'CCW') else -1.0
    info = ds.DetectorInformationSequence
    vendor = _vendors.manufacturer(ds)
    beta, radius, head, number = [], [], [], []
    for d in detectors:
        item = info[d - 1] if len(info) >= d else info[0]
        if 'StartAngle' in item:
            start = float(item.StartAngle)
        elif len(detectors) == 1:
            start = float(rot.StartAngle)
        else:
            raise ValueError(f'{source}: {len(detectors)} detectors but no StartAngle for detector {d}: the views '
                             'cannot be placed')
        angles = start + direction * step * np.arange(views_per_detector)
        if vendor == 'MEDISO':
            radial = rotations[min(d, len(rotations)) - 1].RadialPosition
        elif 'RadialPosition' in item:
            radial = item.RadialPosition
        else:
            radial = rotations[min(d, len(rotations)) - 1].RadialPosition
        if not isinstance(radial, collections.abc.Sequence) or isinstance(radial, str):
            radial = [radial] * views_per_detector
        radial = np.asarray([float(r) for r in radial])
        if len(radial) != views_per_detector:
            raise ValueError(f'{source}: detector {d} has {len(radial)} radial positions for {views_per_detector} views')
        beta.append((angles + 180) % 360)
        radius.append(radial)
        head += [d] * views_per_detector
        number += list(range(1, views_per_detector + 1))
    beta, radius = np.concatenate(beta), np.concatenate(radius)
    try:      # GE Xeleris: the radius of each view is offset by a private value
        offsets = np.array(ds[0x0055, 0x1022][0][0x0013, 0x101E].value, dtype=np.float64).reshape(-1, 3)[:, -1]
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        offsets = None
    if offsets is not None:
        if len(offsets) == len(radius):
            radius = radius + offsets
        else:
            warnings.warn(f'{source}: the private radius offsets (0055,1022) have {len(offsets)} values for '
                          f'{len(radius)} views; they are not used')
    # orientation of each head (#25): the IOP is that of the head's first view
    dz, dr = (float(v) for v in ds.PixelSpacing)           # between rows (z), between columns (r) (#24)
    Lz, Lr = int(ds.Rows), int(ds.Columns)
    flip_r = {}
    col = None
    for k, d in enumerate(detectors):
        item = info[d - 1] if len(info) >= d else info[0]
        if 'ImageOrientationPatient' not in item:
            raise ValueError(f'{source}: detector {d} has no ImageOrientationPatient')
        iop = np.asarray([float(v) for v in item.ImageOrientationPatient])
        row_dir, col_dir = iop[:3] / np.linalg.norm(iop[:3]), iop[3:] / np.linalg.norm(iop[3:])
        if abs(col_dir[2]) < 0.99:
            raise ValueError(f'{source}: the projection rows of detector {d} do not run along z (IOP {iop.tolist()})')
        if col is not None and np.sign(col_dir[2]) != np.sign(col[2]):
            raise ValueError(f'{source}: the detectors disagree on the direction of the rows')
        col = col_dir
        b = np.radians(beta[views_per_detector * k])
        along = float(row_dir @ np.array([np.cos(b), -np.sin(b), 0.0]))
        if abs(along) < 0.99:
            raise ValueError(f'{source}: the columns of detector {d} do not run along the detector at its first view '
                             f'(IOP {iop.tolist()}, beta {np.degrees(b):.2f})')
        flip_r[d] = along < 0
    position = str(ds.get('PatientPosition', '') or '').upper()
    if position and position != 'FFS':
        warnings.warn(f"{source}: PatientPosition '{position}': only feet-first supine data have been checked; the "
                      "orientation is taken from the detectors' ImageOrientationPatient", _vendors.UnverifiedVendorRule)
    order = np.argsort(beta, kind='stable')
    centre = _projection_centre(ds, Lr, Lz, dr, dz, col, source)
    nuclide = _nuclide(ds)
    duration = rot.get('ActualFrameDuration')
    # the frame of every (window, time slot, view in β order)
    frame = {(w, s, d, v): f for f, (w, s, d, v) in enumerate(zip(window, slot, detector, view))}
    head, number = np.asarray(head)[order], np.asarray(number)[order]
    frames = np.array([[[frame[(w, s, d, v)] for d, v in zip(head, number)] for s in slots] for w in windows])
    return dict(frames=frames, slots=slots, beta=beta[order], radius=radius[order],
                flip_r=np.array([flip_r[d] for d in head]), flip_z=bool(col[2] < 0), Lr=Lr, Lz=Lz, dr=dr, dz=dz,
                centre=centre, energy_windows=_energy_windows(ds, nuclide), nuclide=nuclide,
                frame_duration_s=None if duration is None else float(duration) / 1000,
                acquisition_start=_acquisition_start(ds))


def _metadata(geometry: dict) -> tuple[SPECTObjectMeta, SPECTProjMeta]:
    g = geometry
    object_meta = SPECTObjectMeta((g['dr'], g['dr'], g['dz']), (g['Lr'], g['Lr'], g['Lz']), centre=g['centre'],
                                  frame='patient', units='counts')
    proj_meta = SPECTProjMeta((g['Lr'], g['Lz']), (g['dr'], g['dz']), g['beta'], g['radius'],
                              energy_windows=g['energy_windows'], frame_duration_s=g['frame_duration_s'],
                              acquisition_start=g['acquisition_start'], nuclide=g['nuclide'], units='counts')
    return object_meta, proj_meta


def read_metadata(file: str) -> tuple[SPECTObjectMeta, SPECTProjMeta]:
    """The object grid and the projection metadata of a NM file, without its projections (see :func:`read_dicom`)."""
    return _metadata(_geometry(pydicom.dcmread(file, stop_before_pixels=True), str(file)))


def _select(projections: torch.Tensor, geometry: dict, energy_window: int | None, time_slot: int | None, source: str):
    """[W, S, θ, r, z] -> [W, θ, r, z] or [θ, r, z], by the arguments only."""
    n_slots = len(geometry['slots'])
    if time_slot is None:
        if n_slots > 1:
            raise ValueError(f'{source} is gated ({n_slots} time slots): pass time_slot=')
        time_slot = 0
    projections = projections[:, time_slot]
    return projections if energy_window is None else projections[energy_window]


def read_dicom(file: str, energy_window: int | None = None, *, time_slot: int | None = None):
    """SPECT projections from a DICOM NM file (one bed, one rotation), in PyTomography's standard form.

    * Frames are found by their vectors (energy window, detector, rotation, view, time slot), and their number is
      checked (#26).
    * β = (DICOM angle + 180) mod 360, views sorted by β; per-head start angles; radii in mm.
    * Rows and columns are flipped as the heads' ImageOrientationPatient say, so that z runs foot to head and r along
      (cos β, -sin β, 0) (#25). PixelSpacing is (between rows, between columns) and the grid is Columns wide (#24).
    * The grid is centred on the rotation axis and the middle row, placed in the patient by the vendor table.

    Args:
        file (str): the NM file.
        energy_window (int, optional): index of the energy window to return. Defaults to None: all of them.
        time_slot (int, optional): for gated files, the time slot to return (required for them).

    Returns:
        (torch.Tensor, SPECTObjectMeta, SPECTProjMeta): projections [W, θ, r, z] (or [θ, r, z] for one
        ``energy_window``) on ``pytomography.device``; the object grid (mm, frame ``'patient'``); the projection
        metadata, with every energy window, the frame duration (s), the acquisition start and the nuclide.
    """
    ds = pydicom.dcmread(file)
    geometry = _geometry(ds, str(file))
    pixels = ds.pixel_array.astype(np.float32)
    if pixels.ndim == 2:
        pixels = pixels[None]
    slope = float(getattr(ds, 'RescaleSlope', 1) or 1)
    intercept = float(getattr(ds, 'RescaleIntercept', 0) or 0)
    if slope != 1 or intercept != 0:                       # stored values to real ones (#232)
        pixels = pixels * slope + intercept
    views = pixels[geometry['frames']]                      # [W, S, θ, rows (z), columns (r)]
    if geometry['flip_z']:
        views = views[..., ::-1, :]
    views = np.swapaxes(views, -1, -2)                      # [W, S, θ, r, z]
    flip = geometry['flip_r']
    if flip.any():
        views = views.copy()
        views[:, :, flip] = views[:, :, flip, ::-1, :]
    projections = torch.tensor(np.ascontiguousarray(views)).to(pytomography.dtype).to(pytomography.device)
    object_meta, proj_meta = _metadata(geometry)
    return _select(projections, geometry, energy_window, time_slot, str(file)), object_meta, proj_meta


# ---------- several bed positions ----------

def read_dicom_beds(files: Sequence[str]) -> list:
    """The projections of several bed positions of one study (one NM file per bed), each in the standard form, sorted
    from the most inferior bed to the most superior. Where neighbouring beds overlap, each bed's projections beyond
    the middle of the overlap are replaced by its neighbour's, as the stitched image takes each half of the overlap
    from one bed.

    Args:
        files (Sequence[str]): one NM file per bed, in any order.

    Returns:
        list: one (projections [W, θ, r, z], object_meta, proj_meta) per bed, the inferior bed first.
    """
    beds = sorted((read_dicom(f) for f in files), key=lambda bed: bed[1].centre[2])
    projections = [bed[0] for bed in beds]
    combined = [p.clone() for p in projections]
    dz = beds[0][1].dz
    zs = [int(round((bed[1].centre[2] - beds[0][1].centre[2]) / dz)) for bed in beds]
    Lz = projections[0].shape[-1]
    for i in range(len(beds)):
        if i > 0:                                           # the lower part, from the bed below
            shift = zs[i] - zs[i - 1]
            middle = int((Lz - shift) / 2)
            combined[i][..., :middle] = projections[i - 1][..., shift:shift + middle]
        if i < len(beds) - 1:                               # the upper part, from the bed above
            shift = zs[i + 1] - zs[i]
            middle = int((Lz - shift) / 2)
            combined[i][..., shift + middle:] = projections[i + 1][..., middle:Lz - shift]
    return [(combined[i], beds[i][1], beds[i][2]) for i in range(len(beds))]


def stitch_beds(recons, object_metas: Sequence[SPECTObjectMeta], return_stitching_weights: bool = False):
    """Stitch the reconstructions of several bed positions into one image on one grid.

    Each bed gives the stitched image its own slices, apart from its first and last slice; where two beds overlap,
    each gives half of the overlap (the lower half from the lower bed). Beds are placed by their grids' centres,
    rounded to whole slices.

    Args:
        recons (torch.Tensor | Sequence[torch.Tensor]): [n_beds, Lx, Ly, Lz], in the order of ``object_metas``.
        object_metas (Sequence[SPECTObjectMeta]): each bed's grid (same voxel size and shape).
        return_stitching_weights (bool): instead, return the weights [n_beds, Lx, Ly, Lz] the stitched image gives each
            bed's voxels (1 where it takes them, 0 elsewhere) and the index of each bed's first slice in it, both in
            the order of ``object_metas``: the stitched image is the sum over beds of ``weights[k] * recons[k]``
            placed at ``zs[k]`` (used for uncertainty estimates). Defaults to False.

    Returns:
        (torch.Tensor, SPECTObjectMeta): the stitched image and its grid (its first slice is the first slice of the
        lowest bed), or the weights and slice offsets.
    """
    recons = torch.stack(list(recons)) if not torch.is_tensor(recons) else recons
    centres_z = np.array([meta.centre[2] for meta in object_metas])
    order = np.argsort(centres_z, kind='stable')
    lowest = object_metas[order[0]]
    dz = lowest.dz
    zs_sorted = np.round((centres_z[order] - centres_z[order[0]]) / dz).astype(int)
    recons_sorted = recons[torch.as_tensor(order, device=recons.device)]
    Lx, Ly, Lz = recons.shape[1:]
    new_Lz = int(zs_sorted[-1]) + Lz
    blank_below, blank_above = 1, Lz - 1
    weights = []
    for _ in range(len(recons)):
        w = torch.zeros((Lx, Ly, Lz), device=recons.device)
        w[:, :, blank_below:blank_above] = 1
        weights.append(w)
    for i in range(len(recons) - 1):                        # each overlap: half from each bed
        overlap_lower = zs_sorted[i + 1] - zs_sorted[i] + blank_below
        delta = blank_above - overlap_lower
        half = round(delta / 2)
        weights[i][:, :, overlap_lower + half:overlap_lower + delta] = 0
        weights[i + 1][:, :, blank_below:blank_below + half] = 0
    inverse = np.argsort(order)
    if return_stitching_weights:
        return torch.stack(weights)[torch.as_tensor(inverse, device=recons.device)], zs_sorted[inverse]
    stitched = torch.zeros((Lx, Ly, new_Lz), device=recons.device)
    for i in range(len(recons)):
        stitched[:, :, zs_sorted[i]:zs_sorted[i] + Lz] += recons_sorted[i] * weights[i]
    first_slice = lowest.centre[2] - (Lz - 1) / 2 * dz
    meta = lowest.with_grid(shape=(Lx, Ly, new_Lz))
    meta.centre = (lowest.centre[0], lowest.centre[1], first_slice + (new_Lz - 1) / 2 * dz)
    return stitched, meta


# ---------- stray radiation ----------

def get_mean_stray_radiation_counts(file_blank: str, file_NM: str, index_peak: int | None = None):
    """The mean counts per pixel and view of a blank scan (no source) in each energy window, scaled to the frame
    duration of ``file_NM``: the stray radiation to add to the expected counts.

    Args:
        file_blank (str): the NM file of the blank scan.
        file_NM (str): the NM file of the acquisition.
        index_peak (int, optional): return this window's value only. Defaults to None: every window's.
    """
    projections_blank, _, meta_blank = read_dicom(file_blank)
    _, meta_NM = read_metadata(file_NM)
    mask = fov_mask(projections_blank)
    n_views = projections_blank.shape[-3]
    mean = (projections_blank * mask).sum(dim=(-3, -2, -1)) / (n_views * mask.sum())
    mean = mean * float(meta_NM.frame_duration_s[0]) / float(meta_blank.frame_duration_s[0])
    return mean if index_peak is None else mean[index_peak].item()


def get_mean_stray_radiation_counts_MEW_scatter(file_blank, file_NM, index_peak, index_lower, index_upper=None,
                                                weighting_lower=0.5, weighting_upper=0.5):
    """The stray radiation of the scatter estimate (:func:`pytomography.io.SPECT.energy_window_scatter`) made from the
    windows ``index_lower`` and ``index_upper`` (see :func:`get_mean_stray_radiation_counts`)."""
    mean_stray = get_mean_stray_radiation_counts(file_blank, file_NM)
    windows = read_metadata(file_NM)[1].energy_windows
    stray_upper = mean_stray[index_upper] if index_upper is not None else 0
    width_upper = windows[index_upper].width_keV if index_upper is not None else None
    return compute_EW_scatter(mean_stray[index_lower], stray_upper, windows[index_lower].width_keV, width_upper,
                              windows[index_peak].width_keV, weighting_lower, weighting_upper).item()


def print_energy_window_info(file_NM: str):
    """Print the energy windows of a NM file (as ``proj_meta.energy_windows`` gives them)."""
    for i, window in enumerate(read_metadata(file_NM)[1].energy_windows):
        ranges = ', '.join(f'[{lo:g} keV, {hi:g} keV]' for lo, hi in window.ranges_keV)
        print(f'Index {i}:   Name: "{window.name}", Energies: {ranges}, photon {window.photon_keV:g} keV')
