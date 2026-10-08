from __future__ import annotations
import glob
import os
import warnings
import pydicom
import struct
import torch
import numpy as np
import pytomography
from pytomography.metadata.CT import CTGen3ProjMeta
from . import preprocessing

#: DICOM-CT-PD preprocessing flags (group 7039): what was applied to the projections before export.
CORRECTION_FLAGS = {0x1003: 'beam hardening', 0x1004: 'gain', 0x1005: 'dark field', 0x1006: 'flat field', 0x1007: 'bad pixel',
                    0x1008: 'scatter', 0x1009: 'log'}


def sorted_paths(paths) -> list:
    """The DICOM-CT-PD files of a folder (``*.dcm``), or the given files, in acquisition order (``InstanceNumber``).
    Files without an InstanceNumber keep the order given.

    Args:
        paths (str | list): A folder, or a list of files.

    Returns:
        list: The files, sorted.
    """
    if isinstance(paths, (str, os.PathLike)) and os.path.isdir(paths):
        paths = sorted(glob.glob(os.path.join(paths, '*.dcm')))
    paths = list(paths)
    numbers = [pydicom.dcmread(p, stop_before_pixels=True, specific_tags=['InstanceNumber']).get('InstanceNumber') for p in paths]
    if any(n is None for n in numbers):
        return paths
    return [p for _, p in sorted(zip((int(n) for n in numbers), paths), key=lambda x: x[0])]


def _read_gen3(paths) -> dict:
    """Projections, per-view geometry, InstanceNumber and PhotonStatistics of every file, in the order given, in one
    pass written straight into preallocated arrays (Python parses the files, so threads do not help; preallocation
    avoids the copies of stacking)."""
    first = pydicom.dcmread(paths[0])
    shape = first.pixel_array.shape
    N = len(paths)
    projections = torch.empty((N, *shape), dtype=pytomography.dtype)
    geometry = np.empty((N, 8), dtype=np.float32)
    has_photons = (0x7033, 0x1065) in first
    photons = np.empty((N, len(first[0x7033, 0x1065].value) // 4), dtype=np.float32) if has_photons else None
    instances = [None] * N
    for j, path in enumerate(paths):
        ds = first if j == 0 else pydicom.dcmread(path)
        geometry[j] = (struct.unpack('<f', ds[0x7031,0x1001].value)[0], struct.unpack('<f', ds[0x7031,0x1002].value)[0],
                       struct.unpack('<f', ds[0x7031,0x1003].value)[0], struct.unpack('<f', ds[0x7033,0x100B].value)[0],
                       struct.unpack('<f', ds[0x7033,0x100C].value)[0], struct.unpack('<f', ds[0x7033,0x100D].value)[0],
                       *struct.unpack('<2f', ds[0x7031,0x1033].value))
        if has_photons:
            if (0x7033, 0x1065) in ds:
                photons[j] = np.frombuffer(ds[0x7033, 0x1065].value, dtype='<f4')
            else:
                has_photons, photons = False, None
        instances[j] = ds.get('InstanceNumber')
        projections[j] = torch.from_numpy(ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept)
    geometry = torch.from_numpy(geometry)
    out = {k: geometry[:, i].contiguous() for i, k in enumerate(('phi', 'z', 'rho', 'dphi', 'dz', 'drho', 'col', 'row'))}
    out['projections'] = projections
    out['photons'] = torch.from_numpy(photons) if has_photons else None
    out['instances'] = None if any(n is None for n in instances) else [int(n) for n in instances]
    return out


def get_geometry_info_from_datasets_gen3(paths):
    """Projections and per-view geometry of the given DICOM-CT-PD files, in the order given."""
    d = _read_gen3(paths)
    return d['projections'], d['phi'], d['rho'], d['z'], d['dphi'], d['drho'], d['dz'], d['col'], d['row']


def get_photon_statistics(paths) -> torch.Tensor | None:
    """Incident photons per detector column of every view, (7033,1065) PhotonStatistics, as (views, columns), in the
    order of ``paths``; None if the files do not carry it."""
    photons = []
    for path in paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=[(0x7033, 0x1065)])
        if (0x7033, 0x1065) not in ds:
            return None
        photons.append(np.frombuffer(ds[0x7033, 0x1065].value, dtype='<f4'))
    return torch.from_numpy(np.stack(photons).astype(np.float32))


def get_water_attenuation(path) -> float | None:
    """Attenuation coefficient of water (per mm) the projections are calibrated to, (7041,1001); None if absent."""
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    if (0x7041, 0x1001) not in ds:
        return None
    value = ds[0x7041, 0x1001].value
    return float(value.decode().strip('\x00 ') if isinstance(value, bytes) else value)


def get_correction_flags(path) -> dict:
    """The DICOM-CT-PD preprocessing flags (7039,1003-1009) of a projection file, e.g. ``{'beam hardening': True, ...,
    'scatter': True}``: the corrections applied to the projections before they were exported."""
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    flags = {}
    for element, name in CORRECTION_FLAGS.items():
        if (0x7039, element) in ds:
            value = ds[0x7039, element].value
            value = value.decode() if isinstance(value, bytes) else str(value)
            flags[name] = value.strip('\x00 ').upper() == 'YES'
    return flags


def _table_feed(source_phis: torch.Tensor, source_zs: torch.Tensor) -> float:
    """Advance of the focal spot (mm) per rotation, from a straight line fit of z against the unwrapped angle."""
    beta = np.unwrap(source_phis.double().numpy())
    if len(beta) < 2 or abs(beta[-1] - beta[0]) < 1e-9:
        return 0.0
    slope = np.polyfit(beta - beta[0], source_zs.double().numpy(), 1)[0]
    return abs(slope) * 2 * np.pi


def get_projections_and_metadata_gen3(paths, low_signal_filter: bool = True, low_signal_photons: float = 30.0,
                                      central_column_offset: float = 0.0, angle_offset_deg: float = 0.0,
                                      table_feed: float | str | None = None, column_scale: dict | None = None):
    r"""Projections (line integrals; views, columns, rows) and :class:`~pytomography.metadata.CT.CTGen3ProjMeta` of a
    DICOM-CT-PD scan, with the views in acquisition order (``InstanceNumber``).

    By default rays that expect fewer than ``low_signal_photons`` photons (from (7033,1065) PhotonStatistics) are
    filtered with :func:`~pytomography.io.CT.preprocessing.filter_low_signal`, which removes most of the streaks of
    photon starvation (between the shoulders, for example). The scanner conventions below are off unless asked for;
    each was found by comparing reconstructions with one scanner's own images (TCIA LDCT-and-Projection-data case
    C145, GE) and is not confirmed on other scans:

    * ``central_column_offset``: channels added to the DetectorCentralElement column. C145 and C001 (GE) fit their own
      data best with -0.42.
    * ``angle_offset_deg``: added to every focal spot angle; it rotates the image. C145: 0.161.
    * ``table_feed``: rescale the focal spot z positions, about the last view, so that the table advances this much
      per rotation (mm), or ``'pitch'`` for SpiralPitchFactor (0018,9311) times the collimation. C145's images use the
      nominal feed (39.375 mm), 0.2% less than its projections imply.
    * ``column_scale``: ``dict(g0=..., g2=...)``, a per-column scale of the line integrals
      (:func:`~pytomography.io.CT.preprocessing.scale_columns`); C145: ``dict(g0=-0.0149, g2=0.0186)``.

    The metadata also carries ``photon_counts``, ``water_attenuation`` (per mm, to convert to HU), ``correction_flags``
    and ``spiral_pitch``.

    Args:
        paths (str | list): A folder of DICOM-CT-PD files, or a list of them.
        low_signal_filter (bool, optional): Filter photon-starved rays. Defaults to True.
        low_signal_photons (float, optional): Photons a filtered ray should represent. Defaults to 30.
        central_column_offset (float, optional): Defaults to 0.
        angle_offset_deg (float, optional): Defaults to 0.
        table_feed (float | str | None, optional): Defaults to None (as stored).
        column_scale (dict | None, optional): Defaults to None.

    Returns:
        tuple: Projections (torch.Tensor) and metadata (CTGen3ProjMeta).
    """
    if isinstance(paths, (str, os.PathLike)) and os.path.isdir(paths):
        paths = sorted(glob.glob(os.path.join(paths, '*.dcm')))
    paths = list(paths)
    d = _read_gen3(paths)                                       # one pass, then into acquisition order
    if d['instances'] is not None:
        order = np.argsort(d['instances'], kind='stable')
        if np.any(order != np.arange(len(order))):
            index = torch.from_numpy(order)
            d = {k: (v.index_select(0, index) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}
    projections = d['projections']
    ds = pydicom.dcmread(paths[0], stop_before_pixels=True)
    detector_tranverse_spacing = struct.unpack('<f', ds[0x7029,0x1002].value)[0]
    DSD = struct.unpack('<f', ds[0x7031,0x1031].value)[0]
    # the transverse spacing is an arc length on the cylindrical detector, which subtends spacing / DSD radians
    phi_det_spacing = detector_tranverse_spacing/DSD
    z_det_spacing = struct.unpack('<f', ds[0x7029,0x1006].value)[0]
    pitch = ds.get('SpiralPitchFactor')
    phis = d['phi'] + float(np.radians(angle_offset_deg))
    zs = d['z']
    if table_feed is not None:
        if table_feed == 'pitch':
            if pitch is None:
                raise ValueError("table_feed='pitch' needs SpiralPitchFactor (0018,9311), which these files lack")
            target = float(pitch) * projections.shape[2] * z_det_spacing * float(d['rho'].double().mean()) / DSD
        else:
            target = float(table_feed)
        feed = _table_feed(phis, zs)
        if feed > 0:
            zs = (zs[-1].double() + (zs.double() - zs[-1].double()) * (target / feed)).to(zs.dtype)
    proj_meta = CTGen3ProjMeta(phis, d['rho'], zs, d['dphi'], d['drho'], d['dz'], d['col'] + central_column_offset, d['row'],
                               phi_det_spacing, z_det_spacing, DSD, shape=projections.shape[1:], patient_position=ds.get('PatientPosition'))
    proj_meta.photon_counts = d['photons']
    proj_meta.water_attenuation = get_water_attenuation(paths[0])
    proj_meta.correction_flags = get_correction_flags(paths[0])
    proj_meta.spiral_pitch = None if pitch is None else float(pitch)
    if low_signal_filter:
        if d['photons'] is None:
            warnings.warn('these DICOM-CT-PD files carry no PhotonStatistics (7033,1065), so photon-starved rays are not filtered')
        else:
            projections = preprocessing.filter_low_signal(projections, d['photons'], low_signal_photons)
    if column_scale is not None:
        projections = preprocessing.scale_columns(projections, proj_meta, **column_scale)
    return projections, proj_meta
