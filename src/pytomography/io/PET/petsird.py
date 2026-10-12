"""Read list-mode data in PETSIRD, the format of the Emission Tomography Standardization Initiative (ETSI).

The header of a file is read with ETSI's own Python package, ``petsird``, version 0.9: the format that ETSI's
converters (STIR2PETSIRD, GATE to PETSIRD, CASToR) and STIR write. Install it with ``pip install
"pytomography[petsird]"``. The events are decoded by PyTomography's own compiled reader (numba), about 75 times faster
than ETSI's Python reader: the 197 million prompts of a 60-minute scan on a Siemens mMR take about 8 s.

As for every PET source, the data are converted to PyTomography's standard form before reconstruction::

    header = petsird.read_header(path)
    info = petsird.get_detector_info(header, up="+x", towards_bed="+z", patient_orientation="HFS")
    detector_ids = petsird.get_detector_ids(path, info=info, header=header)
    weights = petsird.get_sensitivity_weights(header, info=info)
    scanner_LUT = shared.get_scanner_LUT(info)

``info`` describes the scanner as the GATE and GE readers do. Detector ids are then PyTomography's crystal numbers,
ring by ring (``ring * info["NrCrystalsPerRing"] + crystal``), and the weights are in the order of
``torch.combinations`` of those ids, so the projectors, randoms and scatter estimates work as for any other source.
PETSIRD 0.9 does not say which way its gantry axes point or how the patient lies, so :func:`get_detector_info` takes
them as arguments; PyTomography's coordinates are then the patient's (see :func:`get_frame_transform`).

Without ``info``, ids are PETSIRD's own numbers of the detecting elements: those of module type 0 first, then those of
type 1, and so on. Within a type, element ``e`` of module ``m`` is number ``m * elements_per_module + e``. The
scanner lookup table of :func:`get_scanner_LUT` uses this numbering and PETSIRD's gantry coordinates.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree

from pytomography.io.PET import shared
from pytomography.metadata.PET import PETTOFMeta

SUPPORTED = (0, 9)  # petsird major, minor: the format of ETSI's converters and STIR


def _petsird():
    try:
        import petsird
        from importlib.metadata import version
    except ImportError:
        raise ImportError('Reading PETSIRD files needs ETSI\'s petsird package, version 0.9: '
                          'pip install "pytomography[petsird]"') from None
    found = tuple(int(x) for x in version("petsird").split(".")[:2])
    if found != SUPPORTED:
        raise ImportError(f"PyTomography reads PETSIRD with petsird 0.9, the format ETSI's converters and STIR write,"
                          f" and petsird {version('petsird')} is installed. Run: pip install \"petsird>=0.9,<0.10\"")
    return petsird


def _scanner(header_or_scanner):
    if isinstance(header_or_scanner, (str, Path)):
        header_or_scanner = read_header(header_or_scanner)
    return getattr(header_or_scanner, "scanner", header_or_scanner)


_STREAM_STARTS: dict = {}  # (file, size, modification time): the byte at which the file's time blocks start


def _file_key(petsird_file) -> tuple:
    path = Path(petsird_file).resolve()
    status = path.stat()
    return str(path), status.st_size, status.st_mtime_ns


def read_header(petsird_file: str):
    """The header of a PETSIRD file: the scanner (``header.scanner``) and the exam.

    Pass it on to :func:`get_detector_ids` (``header=``) so that the file's header is not read again: the
    normalisation tables of a clinical scanner take seconds to read, and gigabytes.
    """
    return _read_header_and_stream_start(petsird_file)[0]


def _read_header_and_stream_start(petsird_file):
    """The header of a PETSIRD file, read with ETSI's package, and the byte at which its time blocks start."""
    petsird = _petsird()
    reader = petsird.BinaryPETSIRDReader(str(petsird_file), skip_completed_check=True)
    try:
        header = reader.read_header()
        # yardl's CodedInputStream reads the file ahead into a buffer: the header ends where the unread part of that
        # buffer starts (petsird is pinned to 0.9, whose reader has these attributes)
        coded = reader._stream
        start = coded._stream.tell() - (coded._last_read_count - coded._offset)
    finally:
        reader.close()
    _STREAM_STARTS[_file_key(petsird_file)] = start
    return header, start


def get_element_offsets(scanner) -> list[int]:
    """The number of the first detecting element of each type of module, then the total number of elements."""
    offsets = [0]
    for rep in _scanner(scanner).scanner_geometry.replicated_modules:
        offsets.append(offsets[-1] + len(rep.transforms) * len(rep.object.detecting_elements.transforms))
    return offsets


def _element_positions(scanner) -> np.ndarray:
    """``N x 3`` centres of the detecting elements' boxes, in mm in the gantry coordinate system (float64)."""
    positions = []
    for rep in _scanner(scanner).scanner_geometry.replicated_modules:
        elements = rep.object.detecting_elements
        centre = np.mean([corner.c for corner in elements.object.shape.corners], axis=0)
        element_mats = np.stack([t.matrix for t in elements.transforms]).astype(np.float64)  # E x 3 x 4
        module_mats = np.stack([t.matrix for t in rep.transforms]).astype(np.float64)  # M x 3 x 4
        in_module = element_mats[:, :, :3] @ centre + element_mats[:, :, 3]  # E x 3
        in_gantry = np.einsum("mij,ej->mei", module_mats[:, :, :3], in_module) + module_mats[:, None, :, 3]
        positions.append(in_gantry.reshape(-1, 3))
    return np.concatenate(positions)


def get_scanner_LUT(scanner) -> torch.Tensor:
    """Position of every detecting element, the centre of its box, in mm in the gantry coordinate system.

    This is PETSIRD's own numbering and coordinates; for reconstruction, use ``shared.get_scanner_LUT(info)`` with
    ``info`` from :func:`get_detector_info`.

    Args:
        scanner: a PETSIRD ``ScannerInformation``, or a header.

    Returns:
        torch.Tensor: ``N x 3`` positions, numbered as described at the top of this module.
    """
    return torch.tensor(_element_positions(scanner), dtype=torch.float32)


def get_scanner_LUT_from_header(header) -> torch.Tensor:
    """Same as :func:`get_scanner_LUT`; the name the PETSIRD tutorial used before PyTomography 4."""
    return get_scanner_LUT(header)


def get_TOF_meta(scanner, n_sigmas: float = 3., type_of_module_pair: tuple[int, int] = (0, 0)) -> PETTOFMeta:
    """Time-of-flight metadata of one pair of module types (for scanners with one type of module, the only pair).

    PETSIRD bins (t1 - t2) c/2 in mm, so low bins are near the first detector of an event. PyTomography's GATE reader
    bins the same quantity in the same order (``gate.get_detector_ids``), so PETSIRD's TOF bins are used as they are.
    PyTomography needs the bins to be of equal width and symmetric about 0.

    Args:
        scanner: a PETSIRD ``ScannerInformation``, or a header.
        n_sigmas (float, optional): how far, in standard deviations, the TOF kernel reaches. Defaults to 3.
        type_of_module_pair (tuple[int, int], optional): the two module types. Defaults to (0, 0).
    """
    scanner = _scanner(scanner)
    t0, t1 = type_of_module_pair
    edges = np.asarray(scanner.tof_bin_edges[t0][t1].edges, dtype=np.float64)
    widths = np.diff(edges)
    if not (np.allclose(widths, widths[0], rtol=1e-4) and np.isclose(edges[0], -edges[-1], atol=1e-3 * widths[0])):
        raise ValueError(f"PyTomography needs TOF bins of equal width, symmetric about 0; this file's are {edges}")
    return PETTOFMeta(num_bins=len(edges) - 1, tof_range=float(edges[-1] - edges[0]),
                      fwhm=float(scanner.tof_resolution[t0][t1]), n_sigmas=n_sigmas)


def get_TOF_meta_from_header(header, n_sigmas: float = 3.) -> PETTOFMeta:
    """Same as :func:`get_TOF_meta`; the name the PETSIRD tutorial used before PyTomography 4."""
    return get_TOF_meta(header, n_sigmas)


_AXES = {"+x": (1., 0., 0.), "-x": (-1., 0., 0.), "+y": (0., 1., 0.), "-y": (0., -1., 0.), "+z": (0., 0., 1.),
         "-z": (0., 0., -1.)}
# patient orientations: the directions of superior and of posterior, as multiples of towards_bed and up. Head first,
# superior points into the gantry, away from the bed; supine, posterior points down.
_ORIENTATIONS = {"HFS": (-1, -1), "FFS": (1, -1), "HFP": (-1, 1), "FFP": (1, 1)}


def _patient_matrix(up: str, towards_bed: str, patient_orientation: str) -> np.ndarray:
    """3 x 3 rotation from PETSIRD's gantry axes to the patient's: rows are left, posterior, superior (DICOM's LPS)."""
    for name, axis in (("up", up), ("towards_bed", towards_bed)):
        if axis not in _AXES:
            raise ValueError(f"{name} must be one of {list(_AXES)}, not {axis!r}")
    if patient_orientation not in _ORIENTATIONS:
        raise ValueError(f"patient_orientation must be one of {list(_ORIENTATIONS)}, not {patient_orientation!r}")
    up_axis, bed_axis = np.array(_AXES[up]), np.array(_AXES[towards_bed])
    if up_axis @ bed_axis != 0:
        raise ValueError("up and towards_bed must be different axes")
    s, p = _ORIENTATIONS[patient_orientation]
    superior, posterior = s * bed_axis, p * up_axis
    return np.stack([np.cross(posterior, superior), posterior, superior])  # LPS is right-handed: L = P x S


def get_frame_transform(info: dict) -> np.ndarray:
    """The 3 x 4 matrix ``[R | t]`` that takes positions in PETSIRD's gantry coordinates (mm) to PyTomography's:
    ``R @ position + t``.

    PyTomography's coordinates for PETSIRD data are the patient's, as in DICOM's LPS: x to the patient's left, y
    posterior and z superior, with the origin in the middle of the scanner, so images shown as ``image[:, :, z].T``
    have the bed at the bottom. They are then turned about the axis by ``info["petsird_alignment_deg"]`` (a fraction
    of a module, which puts the scanner's crystals where PyTomography's description of the scanner has them). To bring
    an attenuation map from LPS into the reconstruction's grid, turn it by that angle about the axis.

    Args:
        info (dict): from :func:`get_detector_info`.
    """
    patient = _patient_matrix(info["petsird_up"], info["petsird_towards_bed"], info["patient_orientation"])
    a = np.radians(info["petsird_alignment_deg"])
    turn = np.array([[np.cos(a), np.sin(a), 0.], [-np.sin(a), np.cos(a), 0.], [0., 0., 1.]])  # by -a about z
    rotation = turn @ patient
    shift = -rotation @ np.array([0., 0., info["petsird_axial_centre"]])
    return np.hstack([rotation, shift[:, None]])


def _levels(values: np.ndarray, gap: float = 0.05) -> np.ndarray:
    """The distinct values, up to differences smaller than ``gap`` (mm): the mean of each group, in increasing order."""
    values = np.sort(values)
    return np.array([group.mean() for group in np.split(values, np.nonzero(np.diff(values) > gap)[0] + 1)])


def get_detector_info(
    scanner,
    up: str = "+x",
    towards_bed: str = "+z",
    patient_orientation: str = "HFS",
    tolerance_mm: float = 0.25,
) -> dict:
    """PyTomography's description of a PETSIRD scanner: the ``info`` dictionary of its standard PET form.

    The scanner must be a cylinder of rings around PETSIRD's z axis, made of one type of module with one layer of
    crystals: modules around the axis and along it, each with rows and columns of crystals. ``info`` holds what the
    GATE and GE readers give (rings, crystals per ring, the spacings of modules and crystals, the radius of the crystal
    centres), the energy window and resolution from the file, and how PETSIRD's coordinates map to PyTomography's
    (keys starting with ``petsird_``, and ``patient_orientation``; see :func:`get_frame_transform`).

    PETSIRD 0.9 does not define which way its gantry axes point, nor how the patient lies, so these are arguments.
    Files written by ETSI's STIR2PETSIRD have ``up="+x"`` and ``towards_bed="+z"``: STIR's gantry axes are x to the
    right looking into the gantry from the bed, y down and z towards the bed, and STIR2PETSIRD writes them as
    (x, y, z) = (-y, -x, z) of STIR.

    Args:
        scanner: a PETSIRD ``ScannerInformation``, a header, or a PETSIRD file.
        up (str, optional): the gantry axis that points up: "+x", "-x", "+y" or "-y". Defaults to "+x".
        towards_bed (str, optional): the scanner's axis, pointing out of the gantry towards the bed: "+z" or "-z".
            Defaults to "+z".
        patient_orientation (str, optional): "HFS", "FFS", "HFP" or "FFP" (head or feet first, supine or prone).
            Defaults to "HFS".
        tolerance_mm (float, optional): the largest distance allowed between a detecting element and the crystal
            PyTomography's description puts there. Defaults to 0.25.

    Raises:
        ValueError: if PyTomography's description cannot place a crystal within ``tolerance_mm`` of every element.

    Returns:
        dict: the scanner's ``info``.
    """
    scanner = _scanner(scanner)
    if towards_bed not in ("+z", "-z"):
        raise ValueError("the scanner's axis must be PETSIRD's z axis: towards_bed '+z' or '-z'")
    patient = _patient_matrix(up, towards_bed, patient_orientation)
    geometry = scanner.scanner_geometry.replicated_modules
    if len(geometry) != 1:
        raise ValueError(f"PyTomography describes scanners with one type of module; this one has {len(geometry)}")
    rep = geometry[0]
    gantry = _element_positions(scanner)
    n = len(gantry)
    n_modules = len(rep.transforms)
    n_elements = len(rep.object.detecting_elements.transforms)
    module = np.arange(n) // n_elements
    axial_centre = float((gantry[:, 2].min() + gantry[:, 2].max()) / 2)
    position = (gantry - np.array([0., 0., axial_centre])) @ patient.T

    # along the axis: rings, modules along the axis, and the rings of one module
    rings = _levels(position[:, 2])
    module_z = np.bincount(module, weights=position[:, 2]) / n_elements
    module_levels = _levels(module_z)
    module_rings = _levels(position[module == 0, 2])
    n_axial_modules, rings_per_module = len(module_levels), len(module_rings)
    if n_modules % n_axial_modules or n_elements % rings_per_module:
        raise ValueError("the modules of this scanner are not arranged in rings around its axis")
    n_trans_modules = n_modules // n_axial_modules
    crystals_per_module_row = n_elements // rings_per_module
    if n_trans_modules * crystals_per_module_row * len(rings) != n:
        raise ValueError("the detecting elements of this scanner do not form complete rings around its axis")

    # around the axis: one row of a module's crystals lies along the module's face, at the radius
    row = position[(module == 0) & (np.abs(position[:, 2] - module_rings[0]) < 0.05), :2]
    if len(row) > 1:
        along = np.linalg.svd(row - row.mean(axis=0))[2][0]
        normal = np.array([-along[1], along[0]])
        radius = abs(float(row.mean(axis=0) @ normal))
        crystal_trans_spacing = float(np.mean(np.diff(np.sort(row @ along))))
    else:
        radius = float(np.linalg.norm(row[0]))
        normal = row[0] / radius
        crystal_trans_spacing = 0.0
    element_shape = rep.object.detecting_elements
    corners = np.array([corner.c for corner in element_shape.object.shape.corners], dtype=np.float64)
    e_mat = np.asarray(element_shape.transforms[0].matrix, dtype=np.float64)
    m_mat = np.asarray(rep.transforms[0].matrix, dtype=np.float64)
    corners = (m_mat[:, :3] @ (e_mat[:, :3] @ corners.T + e_mat[:, 3:]) + m_mat[:, 3:]).T
    crystal_length = float(np.ptp((corners @ patient.T)[:, :2] @ normal))

    # PyTomography centres module k on the angle 2 pi k / modules around; align the modules' crystal centroids (not
    # their faces: STIR2PETSIRD, for one, places each crystal by its corner, half a crystal off the face's centre)
    centroids = np.stack([np.bincount(module, weights=position[:, k]) / n_elements for k in (0, 1)], axis=1)
    azimuths = np.arctan2(centroids[:, 1], centroids[:, 0])
    alignment = float(np.angle(np.mean(np.exp(1j * azimuths * n_trans_modules))) / n_trans_modules)

    energy_edges = np.asarray(scanner.event_energy_bin_edges[0].edges, dtype=np.float64)
    info = {
        "min_rsector_difference": 0,
        "crystal_length": crystal_length,
        "radius": radius,
        "crystalTransNr": crystals_per_module_row,
        "crystalTransSpacing": crystal_trans_spacing,
        "crystalAxialNr": rings_per_module,
        "crystalAxialSpacing": float(np.mean(np.diff(module_rings))) if rings_per_module > 1 else 0.0,
        "submoduleTransNr": 1,
        "submoduleTransSpacing": 0.0,
        "submoduleAxialNr": 1,
        "submoduleAxialSpacing": 0.0,
        "moduleTransNr": 1,
        "moduleTransSpacing": 0.0,
        "moduleAxialNr": n_axial_modules,
        "moduleAxialSpacing": float(np.mean(np.diff(module_levels))) if n_axial_modules > 1 else 0.0,
        "rsectorTransNr": n_trans_modules,
        "rsectorAxialNr": 1,
        "NrCrystalsPerRing": n_trans_modules * crystals_per_module_row,
        "NrRings": len(rings),
        "firstCrystalAxis": 1,
        "TOF": int(scanner.tof_bin_edges[0][0].number_of_bins() > 1),
        "energy_window_low": float(energy_edges[0]),
        "energy_window_high": float(energy_edges[-1]),
        "energy_resolution": float(scanner.energy_resolution_at_511[0]),
        "petsird_up": up,
        "petsird_towards_bed": towards_bed,
        "patient_orientation": patient_orientation,
        "petsird_axial_centre": axial_centre,
        "petsird_alignment_deg": math.degrees(alignment),
    }
    get_detector_id_map(scanner, info, tolerance_mm)  # checks that every element has its crystal
    return info


def get_detector_id_map(scanner, info: dict, tolerance_mm: float = 0.25) -> torch.Tensor:
    """PyTomography's detector id of every PETSIRD detecting element: the crystal of ``shared.get_scanner_LUT(info)``
    at the element's place.

    Args:
        scanner: a PETSIRD ``ScannerInformation``, a header, or a PETSIRD file.
        info (dict): from :func:`get_detector_info`.
        tolerance_mm (float, optional): the largest distance allowed between an element and its crystal. Defaults to
            0.25.

    Returns:
        torch.Tensor: one int64 id per element, in PETSIRD's numbering of the elements.
    """
    frame = get_frame_transform(info)
    position = _element_positions(_scanner(scanner)) @ frame[:, :3].T + frame[:, 3]
    crystals = shared.get_scanner_LUT(info).numpy().astype(np.float64)
    if len(crystals) != len(position):
        raise ValueError(f"info describes {len(crystals)} crystals, and the scanner has {len(position)} elements")
    distance, ids = cKDTree(crystals).query(position)
    if distance.max() > tolerance_mm or len(np.unique(ids)) != len(ids):
        raise ValueError(f"PyTomography's description of this scanner puts crystals up to {distance.max():.2f} mm "
                         f"from PETSIRD's detecting elements (tolerance_mm={tolerance_mm}): PyTomography can only "
                         "describe cylinders of rings of one type of module")
    return torch.from_numpy(ids.astype(np.int64))


def _time_window(time_interval) -> tuple[np.uint64, np.uint64]:
    """[t0, t1) in whole ms: a time block starting at s ms (an integer) is in [start, stop) if t0 <= s < t1."""
    top = 2**64 - 1
    if time_interval is None:
        return np.uint64(0), np.uint64(top)

    def edge(t):
        t = float(t)
        if math.isnan(t):
            raise ValueError("time_interval holds NaN")
        return 0 if t <= 0 else top if t >= top else math.ceil(t)

    return np.uint64(edge(time_interval[0])), np.uint64(edge(time_interval[1]))


def get_detector_ids(
    petsird_file: str,
    read_tof: bool | None = None,
    read_energy: bool = False,
    time_block_ids: Sequence[int] | None = None,
    return_header: bool = False,
    delayeds: bool = False,
    time_interval: tuple[float, float] | None = None,
    max_events: int | None = None,
    info: dict | None = None,
    header=None,
) -> torch.Tensor | tuple[torch.Tensor, object]:
    """Read the coincidences of a PETSIRD file as detector ids.

    The events are decoded by PyTomography's compiled reader, in two passes over the file: one counts the events, the
    other writes them, so memory holds only the result (16 to 40 bytes per event). The first call in a new
    environment compiles the reader, which takes about 10 s; the compiled code is cached for later sessions.

    The file's header is read too, unless it is given (``header``): for a clinical scanner, its normalisation tables
    take seconds to read and gigabytes of memory.

    Args:
        petsird_file (str): the PETSIRD file.
        read_tof (bool | None, optional): add each event's TOF bin as a third column. None (the default) reads it if
            the scanner has more than one TOF bin.
        read_energy (bool, optional): add the energy bin of each detector as two more columns. Defaults to False.
        time_block_ids (Sequence[int] | None, optional): read only these event time blocks, counted from 0.
        return_header (bool, optional): also return the file's header. Defaults to False.
        delayeds (bool, optional): read the delayed coincidences instead of the prompts. Defaults to False.
        time_interval (tuple[float, float] | None, optional): read only the time blocks that start in
            [start, stop), in ms.
        max_events (int | None, optional): stop after the time block in which this many events have been read.
        info (dict | None, optional): from :func:`get_detector_info`: give PyTomography's crystal ids (the standard
            form) instead of PETSIRD's element numbers.
        header (optional): this file's header, from :func:`read_header`, so that it is not read again.

    Returns:
        torch.Tensor: ``N x 2`` detector ids, ``N x 3`` with TOF bins, plus two columns with energy bins; int64. With
        ``return_header``, also the header.
    """
    from pytomography.io.PET._petsird_binary import (BAD_BIN, BAD_TAG, BAD_TYPES, MIXED_TOF, OK, TRUNCATED,
                                                      walk_time_blocks)
    start = _STREAM_STARTS.get(_file_key(petsird_file)) if header is not None else None
    if start is None:  # the file changed, or this header came from elsewhere
        header, start = _read_header_and_stream_start(petsird_file)
    scanner = header.scanner
    n_types = scanner.scanner_geometry.number_of_module_types()
    if delayeds and not scanner.delayed_events_are_stored:
        raise ValueError("this PETSIRD file stores no delayed coincidences")
    offsets = np.array(get_element_offsets(scanner), dtype=np.int64)
    n_energy = np.array([scanner.event_energy_bin_edges[t].number_of_bins() for t in range(n_types)], dtype=np.int64)
    n_tof = np.array([[scanner.tof_bin_edges[a][b].number_of_bins() for b in range(n_types)] for a in range(n_types)],
                     dtype=np.int64)
    if read_tof is None:
        read_tof = bool(n_tof.max() > 1)
    width = 2 + bool(read_tof) + 2 * bool(read_energy)
    detector_ids = np.zeros((0, width), dtype=np.int64)
    if time_block_ids is not None:
        blocks = np.array([int(b) for b in time_block_ids if int(b) >= 0], dtype=np.int64)
        wanted = np.zeros(blocks.max() + 1 if len(blocks) else 0, dtype=np.uint8)
        wanted[blocks] = 1
    else:
        wanted = np.zeros(0, dtype=np.uint8)
    if time_block_ids is None or len(wanted):
        if info is None:
            id_map = np.arange(offsets[-1], dtype=np.int64)
        else:
            id_map = get_detector_id_map(scanner, info).numpy()
        t0, t1 = _time_window(time_interval)
        limit = -1 if max_events is None else int(max_events)
        data = np.memmap(petsird_file, dtype=np.uint8, mode="r")
        n, status = walk_time_blocks(data, start, t0, t1, wanted, limit, bool(delayeds), False, offsets, n_energy,
                                     n_tof, bool(read_tof), bool(read_energy), id_map, detector_ids)
        problem = {
            BAD_TAG: "it holds a time block of a type that PETSIRD 0.9 does not have",
            TRUNCATED: "it ends inside its time blocks, so it is incomplete (still being written, or partly copied)",
            BAD_BIN: "one of its events has a detection bin outside the scanner of its header",
            BAD_TYPES: "one of its time blocks has more types of module than its scanner",
        }
        if status == MIXED_TOF:
            raise ValueError("the module types of this scanner have different TOF bins; read the events without TOF"
                             " (read_tof=False)")
        if status != OK:
            raise ValueError(f"cannot read {petsird_file}: {problem[status]} (after {n:,} events)")
        detector_ids = np.empty((n, width), dtype=np.int64)
        walk_time_blocks(data, start, t0, t1, wanted, limit, bool(delayeds), True, offsets, n_energy, n_tof,
                         bool(read_tof), bool(read_energy), id_map, detector_ids)
        del data
    detector_ids = torch.from_numpy(detector_ids)
    return (detector_ids, header) if return_header else detector_ids


def _pair_index(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Index of the pair (a, b), a < b, in the order of torch.combinations(torch.arange(n), 2)."""
    return a * (2 * n - a - 1) // 2 + (b - a - 1)


def get_sensitivity_weights(scanner, energy_bins: Sequence[int] | None = None, info: dict | None = None
                            ) -> torch.Tensor:
    """Detection efficiency of every pair of detecting elements, for the sensitivity image (``weights_sensitivity``).

    As in PETSIRD, the efficiency of a coincidence between two detection bins is the calibration factor, times the
    efficiencies of the two bins, times the efficiency of their module pair, which is 0 for module pairs that are not
    in coincidence. For a pair of elements this is summed over the energy bins, and the two orders of the elements
    are averaged when the file gives both.

    Args:
        scanner: a PETSIRD ``ScannerInformation``, or a header.
        energy_bins (Sequence[int] | None, optional): count only these energy bins; all of them if None.
        info (dict | None, optional): from :func:`get_detector_info`: number the pairs by PyTomography's crystal ids
            (the standard form) instead of PETSIRD's element numbers.

    Returns:
        torch.Tensor: one float32 weight per pair of elements, in the order of
        ``torch.combinations(torch.arange(N), 2)``, which :class:`~pytomography.metadata.PET.PETLMProjMeta` expects.
        5 bytes per pair while it is computed (4 after): 2.6 GB for a scanner of 32,256 elements.
    """
    from pytomography.io.PET._petsird_binary import add_module_pairs
    scanner = _scanner(scanner)
    eff = scanner.detection_efficiencies
    geometry = scanner.scanner_geometry.replicated_modules
    offsets = get_element_offsets(scanner)
    n = offsets[-1]
    id_map = np.arange(n, dtype=np.int64) if info is None else get_detector_id_map(scanner, info).numpy()
    shape = [(len(r.transforms), len(r.object.detecting_elements.transforms),
              scanner.event_energy_bin_edges[t].number_of_bins()) for t, r in enumerate(geometry)]
    if eff is None or eff.detection_bin_efficiencies is None:
        bin_eff = [np.ones(s) for s in shape]
    else:
        bin_eff = [np.asarray(eff.detection_bin_efficiencies[t], dtype=np.float64).reshape(s)
                   for t, s in enumerate(shape)]
    if energy_bins is not None:
        for t, (_, _, ne) in enumerate(shape):
            keep = np.isin(np.arange(ne), list(energy_bins))
            bin_eff[t] = bin_eff[t] * keep
    calibration = float(eff.calibration_factor) if eff is not None else 1.0
    total = np.zeros(n * (n - 1) // 2, dtype=np.float32)
    orders = np.zeros(n * (n - 1) // 2, dtype=np.uint8)  # 2 where the file gives both orders of a pair
    no_tables = np.zeros((0, 1, 1, 1, 1), dtype=np.float32)
    for a, (ma, ea, na) in enumerate(shape):
        for b, (mb, eb, nb) in enumerate(shape):
            if eff is None or eff.module_pair_sgidlut is None:
                sgid, matrices = np.where(np.eye(ma, mb, dtype=bool) & (a == b), -1, 0), None
            else:
                sgid, matrices = np.asarray(eff.module_pair_sgidlut[a][b]), eff.module_pair_efficiencies_vectors[a][b]
            pairs = np.argwhere(sgid >= 0).astype(np.int64)
            for s in range(0, len(pairs), 4096):  # module pairs in groups, so their tables take at most ~100 MB
                group = pairs[s:s + 4096]
                if matrices is None:
                    tables = no_tables
                else:
                    tables = np.stack([np.asarray(matrices[sgid[m0, m1]].values, dtype=np.float32)
                                       for m0, m1 in group]).reshape(len(group), ea, na, eb, nb)
                add_module_pairs(total, orders, np.ascontiguousarray(group[:, 0]), np.ascontiguousarray(group[:, 1]),
                                 tables, np.ascontiguousarray(bin_eff[a]), np.ascontiguousarray(bin_eff[b]),
                                 offsets[a], offsets[b], id_map, n, calibration)
    np.divide(total, orders, out=total, where=orders > 0)
    return torch.from_numpy(total)
