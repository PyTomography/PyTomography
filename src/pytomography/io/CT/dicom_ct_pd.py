"""Read DICOM-CT-PD: the projections of a third generation (fan beam, cylindrical detector) CT scan, one DICOM file per
view, with the geometry of every view in private tags (TCIA LDCT-and-Projection-data, the DICOM-CT-PD user manual).

:func:`read_dicom_ct_pd` returns the scan in PyTomography's standard form::

    projections, object_meta, proj_meta = io.CT.read_dicom_ct_pd(folder)

The object frame is the patient's LPS (k runs foot to head), centred in x and y on the isocentre and along z on the
middle of the focal spot path; ``object_meta.centre`` is that point in the patient's coordinates.
"""
from __future__ import annotations
import glob
import os
import struct
import warnings
import pydicom
import torch
import numpy as np
import pytomography
from pytomography.io import _vendors
from pytomography.metadata import ObjectMeta
from pytomography.metadata.CT import CTGen3ProjMeta
from . import preprocessing

#: DICOM-CT-PD preprocessing flags (group 7039): what was applied to the projections before export.
CORRECTION_FLAGS = {0x1003: 'beam hardening', 0x1004: 'gain', 0x1005: 'dark field', 0x1006: 'flat field', 0x1007: 'bad pixel',
                    0x1008: 'scatter', 0x1009: 'log'}

#: The DICOM-CT-PD tags the reader uses: name -> (group, element, kind), kind 'f' (float32 values), 'H' (unsigned
#: 16-bit), 's' (text) or 'ds' (a decimal string). The files write them as private tags, whose values pydicom gives as
#: raw bytes (VR UN, or OB) or, in explicit VR files, decoded (FL, US, CS, DS).
TAGS = {
    'transverse_spacing': (0x7029, 0x1002, 'f'),   # DetectorElementTransverseSpacing (mm, an arc at the detector)
    'axial_spacing': (0x7029, 0x1006, 'f'),        # DetectorElementAxialSpacing (mm)
    'detector_shape': (0x7029, 0x100B, 's'),       # DetectorShape: CYLINDRICAL, SPHERICAL or FLAT
    'rows': (0x7029, 0x1010, 'H'),                 # NumberofDetectorRows
    'columns': (0x7029, 0x1011, 'H'),              # NumberofDetectorColumns
    'phi': (0x7031, 0x1001, 'f'),                  # DetectorFocalCenterAngularPosition (rad)
    'z': (0x7031, 0x1002, 'f'),                    # DetectorFocalCenterAxialPosition (mm, the patient's z)
    'rho': (0x7031, 0x1003, 'f'),                  # DetectorFocalCenterRadialDistance (mm)
    'dsd': (0x7031, 0x1031, 'f'),                  # ConstantRadialDistance (mm, focal spot to detector)
    'central_element': (0x7031, 0x1033, 'f'),      # DetectorCentralElement (column, row)
    'dphi': (0x7033, 0x100B, 'f'),                 # SourceAngularPositionShift (rad)
    'dz': (0x7033, 0x100C, 'f'),                   # SourceAxialPositionShift (mm)
    'drho': (0x7033, 0x100D, 'f'),                 # SourceRadialDistanceShift (mm)
    'ffs_mode': (0x7033, 0x100E, 's'),             # FlyingFocalSpotMode
    'views_per_turn': (0x7033, 0x1013, 'H'),       # NumberofSourceAngularSteps
    'photons': (0x7033, 0x1065, 'f'),              # PhotonStatistics (per column)
    'projection_type': (0x7037, 0x1009, 's'),      # TypeofProjectionData: AXIAL or HELICAL
    'geometry_type': (0x7037, 0x100A, 's'),        # TypeofProjectionGeometry: FANBEAM
    'water': (0x7041, 0x1001, 'ds'),               # WaterAttenuationCoefficient (1/mm)
}
_REQUIRED = object()
_SEQUENCES = (list, tuple, pydicom.multival.MultiValue)


def tag(ds, name: str, default=_REQUIRED):
    """A DICOM-CT-PD tag of a file, decoded whatever its VR: a tuple of floats ('f'), an int ('H'), a string ('s') or a
    float ('ds'). Raises KeyError for a missing tag without ``default``."""
    group, element, kind = TAGS[name]
    if (group, element) not in ds:
        if default is _REQUIRED:
            raise KeyError(f"the file has no {name} ({group:04X},{element:04X}): is it DICOM-CT-PD?")
        return default
    value = ds[group, element].value
    if isinstance(value, bytes):
        if kind == 'f':
            if len(value) % 4:
                raise ValueError(f"{name} ({group:04X},{element:04X}) is {len(value)} bytes, not float32 values")
            return tuple(float(v) for v in np.frombuffer(value, dtype='<f4'))
        if kind == 'H':
            return int(struct.unpack('<H', value[:2])[0])
        text = value.decode('ascii', 'replace').strip('\x00 ')
        return float(text) if kind == 'ds' else text
    if kind == 'f':
        return tuple(float(v) for v in (value if isinstance(value, _SEQUENCES) else [value]))
    if kind == 'H':
        return int(value[0] if isinstance(value, _SEQUENCES) else value)
    if kind == 'ds':
        return float(value)
    return str(value).strip('\x00 ')


def _check(ds, n_columns: int, n_rows: int) -> None:
    """Raises for a scan PyTomography cannot model (#36): a detector that is not cylindrical, a geometry that is not a
    fan beam, or rows and columns that disagree with the image."""
    shape = str(tag(ds, 'detector_shape', 'CYLINDRICAL')).upper()
    if shape != 'CYLINDRICAL':
        raise NotImplementedError(f"PyTomography models a cylindrical detector; these files' DetectorShape is {shape}")
    geometry = str(tag(ds, 'geometry_type', 'FANBEAM')).upper()
    if geometry != 'FANBEAM':
        raise NotImplementedError(f"PyTomography models fan beam projections; these files' TypeofProjectionGeometry is "
                                  f"{geometry}")
    kind = str(tag(ds, 'projection_type', 'HELICAL')).upper()
    if kind not in ('HELICAL', 'AXIAL'):
        raise NotImplementedError(f"TypeofProjectionData {kind} is neither HELICAL nor AXIAL")
    for name, n in (('columns', n_columns), ('rows', n_rows)):
        stated = tag(ds, name, None)
        if stated is not None and stated != n:
            raise ValueError(f"the files say {stated} detector {name}, and their images have {n}")


def _paths(folder) -> list:
    if isinstance(folder, (str, os.PathLike)) and os.path.isdir(folder):
        return sorted(glob.glob(os.path.join(folder, '*.dcm')))
    return [os.fspath(p) for p in folder]


def _read_gen3(paths) -> dict:
    """Projections, per-view geometry, InstanceNumber and PhotonStatistics of every file, in the order given, in one
    pass written straight into preallocated arrays (Python parses the files, so threads do not help; preallocation
    avoids the copies of stacking)."""
    first = pydicom.dcmread(paths[0])
    shape = first.pixel_array.shape
    N = len(paths)
    projections = torch.empty((N, *shape), dtype=pytomography.dtype)
    geometry = np.empty((N, 8), dtype=np.float32)
    photons = tag(first, 'photons', None)
    photons = np.empty((N, len(photons)), dtype=np.float32) if photons is not None else None
    instances = [None] * N
    for j, path in enumerate(paths):
        ds = first if j == 0 else pydicom.dcmread(path)
        geometry[j] = (tag(ds, 'phi')[0], tag(ds, 'z')[0], tag(ds, 'rho')[0], tag(ds, 'dphi', (0.0,))[0],
                       tag(ds, 'dz', (0.0,))[0], tag(ds, 'drho', (0.0,))[0], *tag(ds, 'central_element')[:2])
        if photons is not None:
            counts = tag(ds, 'photons', None)
            if counts is None:
                photons = None
            else:
                photons[j] = counts
        instances[j] = ds.get('InstanceNumber')
        projections[j] = torch.from_numpy(ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept)
    geometry = torch.from_numpy(geometry)
    out = {k: geometry[:, i].contiguous() for i, k in enumerate(('phi', 'z', 'rho', 'dphi', 'dz', 'drho', 'col', 'row'))}
    out['projections'] = projections
    out['photons'] = torch.from_numpy(photons) if photons is not None else None
    out['instances'] = None if any(n is None for n in instances) else [int(n) for n in instances]
    return out


def _correction_flags(ds) -> dict:
    """The DICOM-CT-PD preprocessing flags (7039,1003-1009): the corrections applied to the projections before they
    were exported, e.g. ``{'beam hardening': True, ..., 'scatter': True}``."""
    flags = {}
    for element, name in CORRECTION_FLAGS.items():
        if (0x7039, element) in ds:
            value = ds[0x7039, element].value
            value = value.decode('ascii', 'replace') if isinstance(value, bytes) else str(value)
            flags[name] = value.strip('\x00 ').upper() == 'YES'
    return flags


def _central_column(values: torch.Tensor, n_columns: int, vendor: str, central_column='auto') -> torch.Tensor:
    """The detector's central column (counted from 1, fractional) for every view: DetectorCentralElement as stored,
    except that GE counts it from the other end of the detector, from zero, which makes it ``n_columns - tag`` (the
    vendor table's 'ge-ct-pd-central-column'). A number for ``central_column`` gives the column itself, ``'tag'`` the
    value as stored."""
    if central_column == 'tag':
        return values
    if central_column == 'auto':
        if vendor == 'GE':
            _vendors.use('ge-ct-pd-central-column')
            return n_columns - values
        return values
    if isinstance(central_column, str):
        raise ValueError(f"central_column must be 'auto', 'tag' or a number, not {central_column!r}")
    return torch.full_like(values, float(central_column))


def _table_feed(source_phis: torch.Tensor, source_zs: torch.Tensor) -> float:
    """Advance of the focal spot (mm) per rotation, from a straight line fit of z against the unwrapped angle."""
    beta = np.unwrap(source_phis.double().numpy())
    if len(beta) < 2 or abs(beta[-1] - beta[0]) < 1e-9:
        return 0.0
    slope = np.polyfit(beta - beta[0], source_zs.double().numpy(), 1)[0]
    return abs(slope) * 2 * np.pi


def _check_patient_frame(ds, vendor: str) -> None:
    """The patient frame of DICOM-CT-PD: verified for GE scans, feet first supine; other positions raise, other vendors
    warn (Siemens images put their y origin on the table, which the projections do not record, #34)."""
    position = ds.get('PatientPosition')
    if position != 'FFS':
        raise NotImplementedError(f"the patient frame of DICOM-CT-PD is verified for feet first supine (FFS) scans only; "
                                  f"this scan is {position!r}. Compare a reconstruction with the scanner's images to "
                                  "extend it")
    if vendor == 'SIEMENS':
        _vendors.use('siemens-ct-pd-y-origin')
    elif vendor != 'GE':
        maker = vendor or 'an unknown manufacturer'
        warnings.warn(f"the patient frame of DICOM-CT-PD is verified on GE scans only, and these are from {maker}: the "
                      "isocentre is taken as the patient's (0, 0) in x and y", stacklevel=3)


def default_grid(proj_meta: CTGen3ProjMeta) -> tuple:
    """The grid :func:`read_dicom_ct_pd` uses when it is not given one: 512 x 512 over the field of view (the cylinder
    the fan of every view covers), and slices at the detector rows' spacing at the isocentre along the focal spot path.
    Returns ``(dr, shape)``."""
    fan = proj_meta.phis_det[:, 0]
    r_fov = float(proj_meta.source_rhos.min()) * float(torch.sin(torch.minimum(fan[0].abs(), fan[-1].abs())))
    dx = 2 * r_fov / 512
    dz = float(proj_meta.row_det_spacing) * float(proj_meta.source_rhos.double().mean()) / float(proj_meta.DSD)
    extent = float(proj_meta.source_zs.max() - proj_meta.source_zs.min())
    return (dx, dx, dz), (512, 512, max(1, int(extent // dz)))


def read_dicom_ct_pd(folder, *, object_meta: ObjectMeta | None = None, table_feed: float | str | None = None,
                     filter_low_signal: bool = True, low_signal_photons: float = 30.0, central_column='auto',
                     central_column_offset: float = 0.0, column_scale: dict | None = None
                     ) -> tuple[torch.Tensor, ObjectMeta, CTGen3ProjMeta]:
    r"""A DICOM-CT-PD scan in PyTomography's standard form: the projections (line integrals; views, columns, rows), the
    object grid placed in the patient, and :class:`~pytomography.metadata.CT.CTGen3ProjMeta`, with the views in
    acquisition order (``InstanceNumber``).

    The geometry comes from the files' private tags, decoded whatever their VR and checked: the detector must be
    cylindrical and the projections fan beam (#36). The detector's central column is DetectorCentralElement (7031,1033),
    except that GE counts it from the other end of the detector, from zero, so that its central column is ``n_columns -
    DetectorCentralElement`` (443.25 rather than 444.75 on TCIA LDCT-and-Projection-data cases C145 and C001, GE
    Discovery CT750 HD): ``central_column='auto'`` applies this for GE (the vendor table's 'ge-ct-pd-central-column').
    It was found by reconstructing the views with the focal spot in either half of the turn separately: the two images
    coincide with this column and lie 1.4 mm apart with the column as stored.

    The patient frame (``object_meta.centre``, ``frame='patient'``) is verified for GE scans, feet first supine: other
    patient positions raise, and other vendors warn (Siemens images put their y origin on the table, which the
    projections do not record, #34).

    By default rays that expect fewer than ``low_signal_photons`` photons (from (7033,1065) PhotonStatistics) are
    filtered with :func:`~pytomography.io.CT.preprocessing.filter_low_signal`, which removes most of the streaks of
    photon starvation (between the shoulders, for example).

    Two options reproduce the scanner's own images of C145 and are off unless asked for; they were found by comparing
    with those images, not from the files:

    * ``table_feed``: rescale the focal spot z positions, about the last view, so that the table advances this much
      per rotation (mm), or ``'pitch'`` for SpiralPitchFactor (0018,9311) times the collimation. C145's images use the
      nominal feed (39.375 mm), 0.2% less than its projections imply.
    * ``column_scale``: ``dict(g0=..., g2=...)``, a per-column scale of the line integrals
      (:func:`~pytomography.io.CT.preprocessing.scale_columns`), fitted to the scanner's own images with
      :func:`~pytomography.io.CT.preprocessing.fit_column_scale`. GE Discovery CT750 HD at 100 kV: about
      ``dict(g0=-0.012, g2=0.0165)`` (C145; C001 gives -0.0135, 0.0184).

    The metadata also carries ``photon_counts``, ``water_attenuation`` (per mm, for :func:`~pytomography.io.CT.to_hu`),
    ``correction_flags``, ``spiral_pitch`` and ``patient_position``.

    Args:
        folder (str | list): A folder of DICOM-CT-PD files (``*.dcm``), or a list of them.
        object_meta (ObjectMeta, optional): the reconstruction grid (``dr`` and ``shape``); its centre and frame are
            set by the reader. Defaults to None: 512 x 512 over the field of view, and slices at the detector rows'
            spacing at the isocentre along the focal spot path.
        table_feed (float | str | None, optional): Defaults to None (as stored).
        filter_low_signal (bool, optional): Filter photon-starved rays. Defaults to True.
        low_signal_photons (float, optional): Photons a filtered ray should represent. Defaults to 30.
        central_column (str | float, optional): ``'auto'`` (the manufacturer's convention, above), ``'tag'``
            (DetectorCentralElement as stored), or the central column itself (counted from 1). Defaults to ``'auto'``.
        central_column_offset (float, optional): Channels added to the central column. Defaults to 0.
        column_scale (dict | None, optional): Defaults to None.

    Returns:
        tuple[torch.Tensor, ObjectMeta, CTGen3ProjMeta]: the projections (float32 on the CPU), the object grid
        (modality ``'CT'``, units ``'1/mm'``, frame ``'patient'``) and the projection metadata.
    """
    paths = _paths(folder)
    if not paths:
        raise FileNotFoundError(f"no DICOM-CT-PD files (*.dcm) in {folder}")
    header = pydicom.dcmread(paths[0], stop_before_pixels=True)
    vendor = _vendors.manufacturer(header)
    _check_patient_frame(header, vendor)
    d = _read_gen3(paths)                                       # one pass, then into acquisition order
    if d['instances'] is not None:
        order = np.argsort(d['instances'], kind='stable')
        if np.any(order != np.arange(len(order))):
            index = torch.from_numpy(order)
            d = {k: (v.index_select(0, index) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}
    projections = d['projections']
    _check(header, projections.shape[1], projections.shape[2])
    for name in ('col', 'row'):
        if float(d[name].max() - d[name].min()) > 1e-3:
            raise NotImplementedError(f"DetectorCentralElement changes from view to view ({name}); PyTomography models "
                                      "one central element")
    DSD = tag(header, 'dsd')[0]
    # the transverse spacing is an arc length on the cylindrical detector, which subtends spacing / DSD radians
    col_angle_spacing = tag(header, 'transverse_spacing')[0] / DSD
    row_spacing = tag(header, 'axial_spacing')[0]
    pitch = header.get('SpiralPitchFactor')
    phis, zs = d['phi'], d['z']
    if table_feed is not None:
        if table_feed == 'pitch':
            if pitch is None:
                raise ValueError("table_feed='pitch' needs SpiralPitchFactor (0018,9311), which these files lack")
            target = float(pitch) * projections.shape[2] * row_spacing * float(d['rho'].double().mean()) / DSD
        else:
            target = float(table_feed)
        feed = _table_feed(phis, zs)
        if feed > 0:
            zs = (zs[-1].double() + (zs.double() - zs[-1].double()) * (target / feed)).to(zs.dtype)
    col = _central_column(d['col'], projections.shape[1], vendor, central_column) + central_column_offset
    proj_meta = CTGen3ProjMeta(phis, d['rho'], zs, d['dphi'], d['drho'], d['dz'], col, d['row'], col_angle_spacing,
                               row_spacing, DSD, shape=projections.shape[1:], patient_position=header.get('PatientPosition'))
    proj_meta.photon_counts = d['photons']
    proj_meta.water_attenuation = tag(header, 'water', None)
    proj_meta.correction_flags = _correction_flags(header)
    proj_meta.spiral_pitch = None if pitch is None else float(pitch)
    if filter_low_signal:
        if d['photons'] is None:
            warnings.warn('these DICOM-CT-PD files carry no PhotonStatistics (7033,1065), so photon-starved rays are not filtered')
        else:
            projections = preprocessing.filter_low_signal(projections, d['photons'], low_signal_photons, inplace=True)
    if column_scale is not None:        # as scale_columns does it, but in place: these projections are the reader's own
        projections.mul_(preprocessing.column_scale(proj_meta, **column_scale).to(projections)[None, :, None])
    dr, shape = default_grid(proj_meta) if object_meta is None else (object_meta.dr, object_meta.shape)
    object_meta = ObjectMeta(dr, shape, centre=(0.0, 0.0, proj_meta.z_center), frame='patient', units='1/mm',
                             modality='CT')
    return projections, object_meta, proj_meta
