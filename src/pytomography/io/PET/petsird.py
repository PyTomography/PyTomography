"""Read list-mode data in PETSIRD, the format of the Emission Tomography Standardization Initiative (ETSI).

The header of a file is read with ETSI's own Python package, ``petsird``, version 0.9: the format that ETSI's
converters (STIR2PETSIRD, GATE to PETSIRD, CASToR) and STIR write. Install it with ``pip install
"pytomography[petsird]"``. The events are decoded by PyTomography's own compiled reader (numba), about 75 times faster
than ETSI's Python reader: the 197 million prompts of a 60-minute scan on a Siemens mMR take about 8 s.

:func:`read_petsird` returns the data in PyTomography's standard form, and :func:`read_header` reads a file's header
once, to pass on to it::

    header = io.PET.petsird.read_header(path)
    prompts, object_meta, proj_meta = io.PET.read_petsird(path, up="+x", towards_bed="+z", patient_position="HFS",
                                                          header=header)

``proj_meta.info`` describes the scanner as the GATE and GE readers do. Detector ids are PyTomography's crystal
numbers, ring by ring (``ring * info["NrCrystalsPerRing"] + crystal``), and the sensitivity weights are in the order of
``torch.combinations`` of those ids, so the projectors, randoms and scatter estimates work as for any other source.
PETSIRD 0.9 does not say which way its gantry axes point or how the patient lies, so :func:`read_petsird` takes them as
arguments. The object frame is then the patient's LPS, exactly: the scanner lookup table holds any turn of the
scanner's modules about the axis (``info["rsector0_angle"]``).

PETSIRD's own numbers of the detecting elements, which the functions here use without ``info``, run over the elements
of module type 0 first, then those of type 1, and so on; within a type, element ``e`` of module ``m`` is number
``m * elements_per_module + e``. :func:`element_positions` gives their positions in PETSIRD's gantry coordinates.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree

from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta
from .scanner import keep_rsector_difference, scanner_info, scanner_lut
from ._standard import default_grid, reader_object_meta

SUPPORTED = (0, 9)
"""The version (major, minor) of ETSI's ``petsird`` package that PyTomography reads PETSIRD files with, the format
that ETSI's converters and STIR write."""


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
    """Reads the header of a PETSIRD file with ETSI's ``petsird`` package: its scanner (``header.scanner``) and exam.

    Pass it on to :func:`read_petsird` (``header=``) so that the file's header is not read again: the normalisation
    tables of a clinical scanner take seconds to read, and gigabytes.

    Args:
        petsird_file (str): The PETSIRD file.

    Returns:
        petsird.Header: The header, as ETSI's package reads it.

    Raises:
        ImportError: If ETSI's ``petsird`` package, version 0.9, is not installed
            (``pip install "pytomography[petsird]"``).
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


def _element_offsets(scanner) -> list[int]:
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


def element_positions(scanner) -> torch.Tensor:
    """Returns the position of every detecting element, the centre of its box, in PETSIRD's gantry coordinates in mm.

    These are PETSIRD's own numbering and coordinates; the scanner lookup table of the standard form, in the object
    frame, is :func:`~pytomography.io.PET.scanner_lut` of ``proj_meta.info``.

    Args:
        scanner (petsird.ScannerInformation | petsird.Header | str): A PETSIRD ``ScannerInformation``, a header, or a
            PETSIRD file.

    Returns:
        torch.Tensor: float32 positions, ``[N, 3]`` in mm, numbered as described at the top of this module.
    """
    return torch.tensor(_element_positions(scanner), dtype=torch.float32)


def _tof_meta(scanner, n_sigmas: float = 3., type_of_module_pair: tuple[int, int] = (0, 0)) -> PETTOFMeta:
    """Time-of-flight metadata of one pair of module types (for scanners with one type of module, the only pair).

    PETSIRD bins (t1 - t2) c/2 in mm, so low bins are near the first detector of an event. That is PyTomography's
    definition (:class:`~pytomography.metadata.PET.PETTOFMeta`), so PETSIRD's TOF bins are used as they are.
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


_AXES = {"+x": (1., 0., 0.), "-x": (-1., 0., 0.), "+y": (0., 1., 0.), "-y": (0., -1., 0.), "+z": (0., 0., 1.),
         "-z": (0., 0., -1.)}
# patient positions: the directions of superior and of posterior, as multiples of towards_bed and up. Head first,
# superior points into the gantry, away from the bed; supine, posterior points down.
_ORIENTATIONS = {"HFS": (-1, -1), "FFS": (1, -1), "HFP": (-1, 1), "FFP": (1, 1)}


def _patient_matrix(up: str, towards_bed: str, patient_position: str) -> np.ndarray:
    """3 x 3 rotation from PETSIRD's gantry axes to the patient's: rows are left, posterior, superior (DICOM's LPS)."""
    for name, axis in (("up", up), ("towards_bed", towards_bed)):
        if axis not in _AXES:
            raise ValueError(f"{name} must be one of {list(_AXES)}, not {axis!r}")
    if patient_position not in _ORIENTATIONS:
        raise ValueError(f"patient_position must be one of {list(_ORIENTATIONS)}, not {patient_position!r}")
    up_axis, bed_axis = np.array(_AXES[up]), np.array(_AXES[towards_bed])
    if up_axis @ bed_axis != 0:
        raise ValueError("up and towards_bed must be different axes")
    s, p = _ORIENTATIONS[patient_position]
    superior, posterior = s * bed_axis, p * up_axis
    return np.stack([np.cross(posterior, superior), posterior, superior])  # LPS is right-handed: L = P x S


def _frame_transform(info: dict) -> np.ndarray:
    """The 3 x 4 matrix ``[R | t]`` that takes positions in PETSIRD's gantry coordinates (mm) to PyTomography's object
    frame: ``R @ position + t``. That frame is the patient's, as in DICOM's LPS (x to the patient's left, y posterior
    and z superior), with the origin in the middle of the scanner.

    Args:
        info (dict): from :func:`_detector_info`.
    """
    rotation = _patient_matrix(info["petsird_up"], info["petsird_towards_bed"], info["patient_position"])
    shift = -rotation @ np.array([0., 0., info["petsird_axial_centre"]])
    return np.hstack([rotation, shift[:, None]])


def _levels(values: np.ndarray, gap: float = 0.05) -> np.ndarray:
    """The distinct values, up to differences smaller than ``gap`` (mm): the mean of each group, in increasing order."""
    values = np.sort(values)
    return np.array([group.mean() for group in np.split(values, np.nonzero(np.diff(values) > gap)[0] + 1)])


def _detector_info(
    scanner,
    up: str = "+x",
    towards_bed: str = "+z",
    patient_position: str = "HFS",
    tolerance: float = 0.25,
) -> dict:
    """PyTomography's description of a PETSIRD scanner: the ``info`` dictionary of its standard PET form.

    The scanner must be a cylinder of rings around PETSIRD's z axis, made of one type of module with one layer of
    crystals: modules around the axis and along it, each with rows and columns of crystals. ``info`` holds what the
    GATE and GE readers give (rings, crystals per ring, the spacings of modules and crystals, the radius of the LORs'
    end points, the centres of the crystals, with ``doi`` half the crystals' depth), the frame keys, the energy
    window and resolution from the file, and how PETSIRD's coordinates map to the patient's (keys starting with
    ``petsird_``, and ``patient_position``).

    PETSIRD 0.9 does not define which way its gantry axes point, nor how the patient lies, so these are arguments.
    Files written by ETSI's STIR2PETSIRD have ``up="+x"`` and ``towards_bed="+z"``: STIR's gantry axes are x to the
    right looking into the gantry from the bed, y down and z towards the bed, and STIR2PETSIRD writes them as
    (x, y, z) = (-y, -x, z) of STIR.

    Args:
        scanner: a PETSIRD ``ScannerInformation``, a header, or a PETSIRD file.
        up (str, optional): the gantry axis that points up: "+x", "-x", "+y" or "-y". Defaults to "+x".
        towards_bed (str, optional): the scanner's axis, pointing out of the gantry towards the bed: "+z" or "-z".
            Defaults to "+z".
        patient_position (str, optional): "HFS", "FFS", "HFP" or "FFP" (head or feet first, supine or prone).
            Defaults to "HFS".
        tolerance (float, optional): the largest distance allowed between a detecting element and the crystal
            PyTomography's description puts there. Defaults to 0.25.

    Raises:
        ValueError: if PyTomography's description cannot place a crystal within ``tolerance`` of every element.

    Returns:
        dict: the scanner's ``info``.
    """
    scanner = _scanner(scanner)
    if towards_bed not in ("+z", "-z"):
        raise ValueError("the scanner's axis must be PETSIRD's z axis: towards_bed '+z' or '-z'")
    patient = _patient_matrix(up, towards_bed, patient_position)
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

    # PyTomography centres module k on the angle rsector0_angle + 360 k / modules around: the angle that puts
    # each module on the modules' crystal centroids (not their faces: STIR2PETSIRD, for one, places each crystal by its
    # corner, half a crystal off the face's centre, a turn of -0.353 degrees for the mMR)
    centroids = np.stack([np.bincount(module, weights=position[:, k]) / n_elements for k in (0, 1)], axis=1)
    azimuths = np.arctan2(centroids[:, 1], centroids[:, 0])
    alignment = float(np.angle(np.mean(np.exp(1j * azimuths * n_trans_modules))) / n_trans_modules)

    energy_edges = np.asarray(scanner.event_energy_bin_edges[0].edges, dtype=np.float64)
    info = scanner_info(**{
        "min_rsector_difference": 0,
        "crystal_length": crystal_length,
        "radius": radius,
        "doi": crystal_length / 2,
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
        "rsector0_angle": math.degrees(alignment) % 360.0,
        "transaxial_direction": 1,
        "axial_direction": 1,
        "energy_window_low": float(energy_edges[0]),
        "energy_window_high": float(energy_edges[-1]),
        "energy_resolution": float(scanner.energy_resolution_at_511[0]),
        "petsird_up": up,
        "petsird_towards_bed": towards_bed,
        "patient_position": patient_position,
        "petsird_axial_centre": axial_centre,
    })
    _detector_id_map(scanner, info, tolerance)  # checks that every element has its crystal
    return info


def _detector_id_map(scanner, info: dict, tolerance: float = 0.25) -> torch.Tensor:
    """PyTomography's detector id of every PETSIRD detecting element: the crystal of ``scanner_lut(info)`` at the
    element's place.

    Args:
        scanner: a PETSIRD ``ScannerInformation``, a header, or a PETSIRD file.
        info (dict): from :func:`_detector_info`.
        tolerance (float, optional): the largest distance allowed between an element and its crystal. Defaults to
            0.25.

    Returns:
        torch.Tensor: one int64 id per element, in PETSIRD's numbering of the elements.
    """
    frame = _frame_transform(info)
    position = _element_positions(_scanner(scanner)) @ frame[:, :3].T + frame[:, 3]
    crystals = scanner_lut(info).numpy().astype(np.float64)
    if len(crystals) != len(position):
        raise ValueError(f"info describes {len(crystals)} crystals, and the scanner has {len(position)} elements")
    distance, ids = cKDTree(crystals).query(position)
    if distance.max() > tolerance or len(np.unique(ids)) != len(ids):
        raise ValueError(f"PyTomography's description of this scanner puts crystals up to {distance.max():.2f} mm "
                         f"from PETSIRD's detecting elements (tolerance={tolerance}): PyTomography can only "
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


def _detector_ids(
    petsird_file: str,
    read_tof: bool | None = None,
    read_energy: bool = False,
    time_block_ids: Sequence[int] | None = None,
    return_header: bool = False,
    delayeds: bool = False,
    time_interval_ms: tuple[float, float] | None = None,
    max_events: int | None = None,
    info: dict | None = None,
    header=None,
) -> torch.Tensor | tuple[torch.Tensor, object]:
    """Read the coincidences of a PETSIRD file as detector ids.

    The events are decoded by PyTomography's compiled reader, in two passes over the file: one counts the events, the
    other writes them, so memory holds only the result (8 to 20 bytes per event). The first call in a new
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
        time_interval_ms (tuple[float, float] | None, optional): read only the time blocks that start in
            [start, stop), in ms.
        max_events (int | None, optional): stop after the time block in which this many events have been read.
        info (dict | None, optional): from :func:`_detector_info`: give PyTomography's crystal ids (the standard
            form) instead of PETSIRD's element numbers.
        header (optional): this file's header, from :func:`read_header`, so that it is not read again.

    Returns:
        torch.Tensor: ``N x 2`` detector ids, ``N x 3`` with TOF bins, plus two columns with energy bins; int32. With
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
    offsets = np.array(_element_offsets(scanner), dtype=np.int64)
    n_energy = np.array([scanner.event_energy_bin_edges[t].number_of_bins() for t in range(n_types)], dtype=np.int64)
    n_tof = np.array([[scanner.tof_bin_edges[a][b].number_of_bins() for b in range(n_types)] for a in range(n_types)],
                     dtype=np.int64)
    if read_tof is None:
        read_tof = bool(n_tof.max() > 1)
    width = 2 + bool(read_tof) + 2 * bool(read_energy)
    detector_ids = np.zeros((0, width), dtype=np.int32)
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
            id_map = _detector_id_map(scanner, info).numpy()
        t0, t1 = _time_window(time_interval_ms)
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
        detector_ids = np.empty((n, width), dtype=np.int32)
        walk_time_blocks(data, start, t0, t1, wanted, limit, bool(delayeds), True, offsets, n_energy, n_tof,
                         bool(read_tof), bool(read_energy), id_map, detector_ids)
        del data
    detector_ids = torch.from_numpy(detector_ids)
    return (detector_ids, header) if return_header else detector_ids


def _sensitivity_weights(scanner, energy_bins: Sequence[int] | None = None, info: dict | None = None
                            ) -> torch.Tensor:
    """Detection efficiency of every pair of detecting elements, for the sensitivity image (``weights_sensitivity``).

    As in PETSIRD, the efficiency of a coincidence between two detection bins is the calibration factor, times the
    efficiencies of the two bins, times the efficiency of their module pair, which is 0 for module pairs that are not
    in coincidence. For a pair of elements this is summed over the energy bins, and the two orders of the elements
    are averaged when the file gives both.

    Args:
        scanner: a PETSIRD ``ScannerInformation``, or a header.
        energy_bins (Sequence[int] | None, optional): count only these energy bins; all of them if None.
        info (dict | None, optional): from :func:`_detector_info`: number the pairs by PyTomography's crystal ids
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
    offsets = _element_offsets(scanner)
    n = offsets[-1]
    id_map = np.arange(n, dtype=np.int64) if info is None else _detector_id_map(scanner, info).numpy()
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


def read_petsird(
    path: str | os.PathLike,
    *,
    object_meta: ObjectMeta | None = None,
    events: str = "prompts",
    up: str = "+x",
    towards_bed: str = "+z",
    patient_position: str = "HFS",
    time_interval: tuple[float, float] | None = None,
    header=None,
    min_rsector_difference: int = 0,
    tolerance: float = 0.25,
) -> tuple[torch.Tensor, ObjectMeta, PETLMProjMeta]:
    """Reads the coincidences of a PETSIRD 0.9 file in PyTomography's standard form, with the scanner and its TOF bins.

    The scanner must be a cylinder of rings around PETSIRD's z axis, made of one type of module with one layer of
    crystals. Each detecting element is matched to the crystal of PyTomography's description of the scanner
    (``proj_meta.info``) at its place, within ``tolerance``. PETSIRD 0.9 does not say which way its gantry axes
    point, nor how the patient lies, so ``up``, ``towards_bed`` and ``patient_position`` give them. The object frame is
    the patient's LPS, with the origin at the centre of the rings (``frame='scanner'``). Files written by ETSI's
    STIR2PETSIRD have ``up="+x"`` and ``towards_bed="+z"``: STIR's gantry axes are x to the right looking into the
    gantry from the bed, y down and z towards the bed, and STIR2PETSIRD writes them as (x, y, z) = (-y, -x, z) of STIR.
    For the prompts, the file's detection efficiencies give the sensitivity weight of every crystal pair.

    Args:
        path (str | os.PathLike): The PETSIRD file.
        object_meta (ObjectMeta, optional): The reconstruction grid (``dr`` in mm, and ``shape``); its centre and
            frame are set by the reader. Defaults to None: voxels half the crystal pitch, a square covering 80 % of
            the ring's diameter, and every ring.
        events (str, optional): ``'prompts'`` or ``'delayeds'``. Defaults to ``'prompts'``.
        up (str, optional): The gantry axis that points up: ``'+x'``, ``'-x'``, ``'+y'`` or ``'-y'``. Defaults to
            ``'+x'``.
        towards_bed (str, optional): The scanner's axis, pointing out of the gantry towards the bed: ``'+z'`` or
            ``'-z'``. Defaults to ``'+z'``.
        patient_position (str, optional): ``'HFS'``, ``'FFS'``, ``'HFP'`` or ``'FFP'`` (head or feet first, supine or
            prone). Defaults to ``'HFS'``.
        time_interval (tuple[float, float], optional): Read only the time blocks that start in [start, stop), in
            s. Defaults to None: every time block.
        header (petsird.Header, optional): This file's header, from :func:`read_header`, so that it is not read again
            (a clinical scanner's normalisation tables take seconds to read, and gigabytes). Defaults to None.
        min_rsector_difference (int, optional): Keep only events whose crystals are at least this many modules apart
            around the ring. Defaults to 0.
        tolerance (float, optional): The largest distance allowed between a detecting element and the crystal
            PyTomography's description puts there, in mm. Defaults to 0.25.

    Returns:
        tuple[torch.Tensor, ObjectMeta, PETLMProjMeta]: The events (int32 ``[N, 2]`` detector ids, or ``[N, 3]`` with
        the TOF bin as stored, when the scanner has TOF bins), the object grid (modality ``'PT'``, units
        ``'counts'``), and the projection metadata: ``info``, ``tof_meta`` and, for the prompts, the detection
        efficiency of every crystal pair as ``weights_sensitivity`` (in ``torch.combinations`` order; 4 bytes per
        pair, 2 GB for the mMR).

    Raises:
        ValueError: If ``events``, ``up``, ``towards_bed`` or ``patient_position`` is not one of the values above,
            the scanner is not one PyTomography can describe (within ``tolerance``), its TOF bins are not of equal
            width and symmetric about 0, the file stores no delayed coincidences (with ``'delayeds'``), or the file
            cannot be read (incomplete, or with events outside its scanner).
        ImportError: If ETSI's ``petsird`` package, version 0.9, is not installed.
    """
    if events not in ("prompts", "delayeds"):
        raise ValueError(f"events must be 'prompts' or 'delayeds', not {events!r}")
    if header is None:
        header = read_header(path)
    scanner = header.scanner
    info = _detector_info(scanner, up=up, towards_bed=towards_bed, patient_position=patient_position,
                          tolerance=tolerance)
    if min_rsector_difference:
        info = scanner_info(info, min_rsector_difference=min_rsector_difference)
    window_ms = None if time_interval is None else (1000 * time_interval[0], 1000 * time_interval[1])  # PETSIRD's ms
    tof = scanner.tof_bin_edges[0][0].number_of_bins() > 1
    detector_ids = _detector_ids(path, read_tof=tof, delayeds=events == "delayeds", time_interval_ms=window_ms,
                                 info=info, header=header)
    detector_ids = keep_rsector_difference(detector_ids, info)
    tof_meta = _tof_meta(scanner) if tof else None
    weights = _sensitivity_weights(scanner, info=info) if events == "prompts" else None
    object_meta = reader_object_meta(object_meta, default_grid(info), centre=(0.0, 0.0, 0.0), frame="scanner",
                                     modality="PT", units="counts")
    proj_meta = PETLMProjMeta(detector_ids, info, tof_meta=tof_meta, weights_sensitivity=weights)
    return detector_ids, object_meta, proj_meta
