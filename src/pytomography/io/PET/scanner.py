"""The PET scanner in PyTomography's standard form: the ``info`` dictionary and the position of every crystal.

Every PET reader describes its scanner as a parametric ring with :func:`scanner_info`, and the projectors, sinograms,
randoms and scatter estimates work from that description alone. :func:`scanner_lut` places every crystal in the object
frame: LPS axes (x to the patient's left, y posterior, z superior), in mm, with the origin at the centre of the rings,
which is the centre of the reconstruction grid.

**The ring.** Crystals are grouped into submodules, submodules into modules and modules into rsectors, as in GATE's
cylindrical PET system:

* ``{crystal, submodule, module}{Trans, Axial}Nr``: the number of each part around the ring (``Trans``) and along the
  axis (``Axial``) within its parent; ``rsectorTransNr`` rsectors make up the ring, and ``rsectorAxialNr`` sit along
  the axis;
* ``{crystal, submodule, module, rsector}{Trans, Axial}Spacing``: their centre-to-centre pitch, in mm (rsectors around
  the ring are placed by angle, so they have no transaxial spacing);
* ``radius``: the distance from the axis to the LOR end points of rsector 0, in mm: the centre of the module's face
  plus ``doi``, the depth of interaction inside the crystal that the LORs end at;
* ``doi`` and ``crystal_length``: that depth, and the crystals' radial length, in mm (``radius`` already includes
  ``doi``);
* ``NrCrystalsPerRing`` and ``NrRings``, computed from the counts.

A detector id is ``ring * NrCrystalsPerRing + crystal``, where ``crystal`` counts the crystals of a ring (crystal,
then submodule, module, rsector, fastest first) and ``ring`` the rings (crystal, submodule, module, rsector along the
axis).

**The frame keys** say where that ring lies in the object frame:

* ``rsector0_angle``: the azimuth of the centre of rsector 0, from +x towards +y;
* ``transaxial_direction``: +1 when crystal numbers run from +x towards +y (counter-clockwise seen from +z), -1 the
  other way;
* ``axial_direction``: +1 when ring numbers run towards +z (superior), -1 towards -z.

They replace ``firstCrystalAxis`` and the negation of z that :func:`scanner_lut` applied to every scanner before
PyTomography 4 (``firstCrystalAxis=1`` is ``rsector0_angle=0, transaxial_direction=1``; ``firstCrystalAxis=0`` is
``rsector0_angle=90, transaxial_direction=-1``; the old negation of z is ``axial_direction=-1``). Each reader sets
them from what it knows of its source: GATE from ``world_axes``, PETSIRD from the patient's position in the gantry, GE
from its vendor rule.

**The energy window**: ``energy_window_low``, ``energy_window_high`` and ``energy_resolution`` (the FWHM of the
photopeak at 511 keV as a fraction of 511 keV), or None where the source does not give them. The scatter estimate
reads them.

**Events** outside ``min_rsector_difference`` (pairs of crystals in rsectors that are fewer than this many rsectors
apart around the ring) are dropped by the readers; the sinogram's geometry does not depend on it.
"""
from __future__ import annotations

import csv
import math
import os
from collections.abc import Mapping

import numpy as np
import torch

#: (key, default): the counts of the parts of the ring. None means the key is required.
_COUNTS = (("crystalTransNr", None), ("crystalAxialNr", None), ("submoduleTransNr", 1), ("submoduleAxialNr", 1),
           ("moduleTransNr", None), ("moduleAxialNr", None), ("rsectorTransNr", None), ("rsectorAxialNr", 1))
#: the pitch of each part, in mm, required where there is more than one of the part
_SPACINGS = ("crystalTransSpacing", "crystalAxialSpacing", "submoduleTransSpacing", "submoduleAxialSpacing",
             "moduleTransSpacing", "moduleAxialSpacing", "rsectorAxialSpacing")
FRAME_KEYS = {"rsector0_angle": 0.0, "transaxial_direction": 1, "axial_direction": 1}
"""The frame keys of ``info`` and their defaults, which are GATE's world (rsector 0 on +x, crystal numbers running
counter-clockwise seen from +z, ring numbers running towards +z)."""
ENERGY_KEYS = ("energy_window_low", "energy_window_high", "energy_resolution")
"""The energy keys of ``info``, the energy window's limits in keV and the FWHM of the photopeak at 511 keV as a
fraction of 511 keV, each None where the source does not give it."""
#: keys of ``info`` before PyTomography 4, and what replaced them
_RENAMED = {
    "firstCrystalAxis": "the frame keys rsector0_angle, transaxial_direction and axial_direction: "
                        "firstCrystalAxis=1 is rsector0_angle=0, transaxial_direction=1, and firstCrystalAxis=0 is "
                        "rsector0_angle=90, transaxial_direction=-1; the z that scanner LUTs used to negate is "
                        "axial_direction=-1",
    "mean_interaction_depth": "doi (radius includes it)",
    "TOF": "the reader's tof_meta (proj_meta.tof_meta)",
}

_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "pet_scanner_info.txt")


def _table_row(name: str) -> dict:
    """The row of PyTomography's table of clinical scanners (``data/pet_scanner_info.txt``) for one scanner."""
    with open(_TABLE, newline="") as f:
        rows = {row["scanner_name"].strip(): {k.strip(): v.strip() for k, v in row.items() if k != "scanner_name"}
                for row in csv.DictReader(f, skipinitialspace=True)}
    if name not in rows:
        raise ValueError(f"PyTomography's table of PET scanners has no {name!r}; it has {sorted(rows)}")
    return rows[name]


def _scalar(key, value):
    """``value`` as a Python scalar: info is hashed by the geometry caches, so its values must be hashable."""
    if isinstance(value, (np.generic, torch.Tensor)):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"info[{key!r}] must be a number or a string, not {type(value).__name__}")


def _as_int(key, value) -> int:
    value = _scalar(key, value)
    if isinstance(value, str):
        value = float(value)
    if isinstance(value, bool) or value is None or not float(value).is_integer():
        raise ValueError(f"info[{key!r}] must be a whole number, not {value!r}")
    return int(value)


def _as_float(key, value) -> float | None:
    value = _scalar(key, value)
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "none", "nan")):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"info[{key!r}] must be a number, not {value!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"info[{key!r}] must be finite, not {value}")
    return value


def scanner_info(scanner: str | Mapping | None = None, **keys) -> dict:
    """Builds the ``info`` dictionary of a PET scanner, PyTomography's description of it as a parametric ring.

    The keys are described at the top of this module (:mod:`pytomography.io.PET.scanner`). Every value is checked and
    converted to a Python ``int`` or ``float`` (or None for a missing energy key), and ``NrCrystalsPerRing`` and
    ``NrRings`` are computed from the counts. Every PET reader builds its ``info`` with this function; use it to
    describe a scanner by hand, or to change keys of an ``info`` a reader returned.

    Args:
        scanner (str | Mapping | None, optional): The name of a scanner in PyTomography's table of clinical scanners
            (``data/pet_scanner_info.txt``, e.g. ``'discovery_MI'``), or an ``info`` dictionary to start from.
            Defaults to None: every required key is in ``keys``.
        **keys: Keys of ``info``, which replace those of ``scanner``. Between them, ``scanner`` and ``keys`` must give
            the counts of crystals, modules and rsectors around the ring and of crystals and modules along the axis,
            ``radius`` (mm), and the spacing (mm) of every part of which there is more than one. ``submodule*Nr``
            and ``rsectorAxialNr`` default to 1, ``doi``, ``crystal_length`` and ``min_rsector_difference`` to 0,
            the frame keys to :data:`FRAME_KEYS` and the energy keys to None. Any other key (a string or a number) is
            kept as it is.

    Returns:
        dict: The scanner's ``info``.

    Raises:
        TypeError: If ``scanner`` is not a name, a mapping or None, or a value is not a number or a string.
        ValueError: If a key is missing or has a value it cannot have, ``scanner`` names no scanner of the table, or a
            key is one that ``info`` had before PyTomography 4 (``firstCrystalAxis``, ``energy_window_low``, ...): the
            message says what replaced it.

    Example:
        >>> info = io.PET.scanner_info(crystalTransNr=8, crystalTransSpacing=4.0, crystalAxialNr=8,
        ...                            crystalAxialSpacing=4.0, moduleTransNr=1, moduleAxialNr=8,
        ...                            moduleAxialSpacing=32.25, rsectorTransNr=56, radius=337.0, doi=9.0)
        >>> info = io.PET.scanner_info(info, min_rsector_difference=2)
        >>> info = io.PET.scanner_info("discovery_MI")
    """
    if isinstance(scanner, str):
        row = _table_row(scanner)
        start = {k: v for k, v in row.items() if k not in ("num_tof_bins", "tof_range", "tof_fwhm")}
    elif scanner is None:
        start = {}
    elif isinstance(scanner, Mapping):
        start = dict(scanner)
    else:
        raise TypeError(f"scanner must be a scanner's name or an info dictionary, not {type(scanner).__name__}")
    given = {**start, **keys}
    for old, new in _RENAMED.items():
        if old in given:
            raise ValueError(f"info[{old!r}] is no longer used; PyTomography 4 replaced it with {new}")
    info = {}
    for key, default in _COUNTS:
        if key not in given and default is None:
            raise ValueError(f"info needs {key!r}")
        info[key] = _as_int(key, given.get(key, default))
        if info[key] < 1:
            raise ValueError(f"info[{key!r}] must be at least 1, not {info[key]}")
    for key in _SPACINGS:
        count = info[key.replace("Spacing", "Nr")]
        value = _as_float(key, given.get(key))
        if value is None:
            if count > 1:
                raise ValueError(f"info needs {key!r}: there are {count} of these parts")
            value = 0.0
        if value < 0:
            raise ValueError(f"info[{key!r}] is a pitch, which cannot be negative ({value})")
        info[key] = value
    info["radius"] = _as_float("radius", given.get("radius"))
    if info["radius"] is None or info["radius"] <= 0:
        raise ValueError(f"info needs a positive 'radius' (mm), not {given.get('radius')!r}")
    for key in ("doi", "crystal_length"):
        info[key] = _as_float(key, given.get(key, 0.0)) or 0.0
        if info[key] < 0:
            raise ValueError(f"info[{key!r}] cannot be negative ({info[key]})")
    info["min_rsector_difference"] = _as_int("min_rsector_difference", given.get("min_rsector_difference", 0))
    if not 0 <= info["min_rsector_difference"] <= info["rsectorTransNr"] // 2:
        raise ValueError(f"info['min_rsector_difference'] must be between 0 and half the rsectors around the ring "
                         f"({info['rsectorTransNr'] // 2}), not {info['min_rsector_difference']}")
    info["rsector0_angle"] = _as_float("rsector0_angle", given.get("rsector0_angle", 0.0)) or 0.0
    for key in ("transaxial_direction", "axial_direction"):
        info[key] = _as_int(key, given.get(key, FRAME_KEYS[key]))
        if info[key] not in (1, -1):
            raise ValueError(f"info[{key!r}] must be 1 or -1, not {info[key]}")
    for key in ENERGY_KEYS:
        info[key] = _as_float(key, given.get(key))
    low, high, resolution = (info[k] for k in ENERGY_KEYS)
    if low is not None and high is not None and not high > low:
        raise ValueError(f"the energy window's upper limit ({high} keV) must be above its lower limit ({low} keV)")
    if resolution is not None and not 0 < resolution < 1:
        raise ValueError(f"info['energy_resolution'] is the FWHM at 511 keV as a fraction of 511 keV (0.15 for 15%), "
                         f"between 0 and 1, not {resolution}")
    info["NrCrystalsPerRing"] = (info["crystalTransNr"] * info["submoduleTransNr"] * info["moduleTransNr"]
                                 * info["rsectorTransNr"])
    info["NrRings"] = info["crystalAxialNr"] * info["submoduleAxialNr"] * info["moduleAxialNr"] * info["rsectorAxialNr"]
    for key in ("NrCrystalsPerRing", "NrRings"):
        if key in given and _as_int(key, given[key]) != info[key]:
            raise ValueError(f"info[{key!r}] is {given[key]}, but the parts of the ring make {info[key]}")
    for key, value in given.items():
        if key not in info:
            info[key] = _scalar(key, value)
    return info


def _check_frame(info: Mapping) -> None:
    """Raises if ``info`` was not built by :func:`scanner_info` (it has keys from before PyTomography 4, or lacks the
    frame keys), so that a scanner is never placed in a frame it did not state."""
    for old, new in _RENAMED.items():
        if old in info:
            raise ValueError(f"info[{old!r}] is no longer used; PyTomography 4 replaced it with {new}. Build info with "
                             "pytomography.io.PET.scanner_info")
    missing = [k for k in FRAME_KEYS if k not in info]
    if missing:
        raise ValueError(f"info lacks the frame keys {missing}: build it with pytomography.io.PET.scanner_info")


def crystal_parts(info: Mapping, sort_by_detector_ids: bool = False) -> tuple[torch.Tensor, ...]:
    """Returns the part ids of every crystal of the scanner: its index within each part, around the ring and along it.

    Args:
        info (Mapping): The scanner's ``info`` (:func:`scanner_info`).
        sort_by_detector_ids (bool, optional): Give the crystals in the order of their detector ids. Defaults to
            False: the order of ``torch.cartesian_prod`` of the parts' ranges, in the order of the tuple returned (the
            axial rsector varies fastest).

    Returns:
        tuple[torch.Tensor, ...]: Eight 1D long tensors, one entry per crystal: the transaxial and axial index of the
        crystal within its submodule, of the submodule within its module, of the module within its rsector, and of
        the rsector in the scanner.
    """
    parts = torch.cartesian_prod(*(torch.arange(info[f"{part}{d}Nr"]) for part in ("crystal", "submodule", "module",
                                                                                 "rsector") for d in ("Trans", "Axial")))
    trans_crystal, axial_crystal, trans_submodule, axial_submodule, trans_module, axial_module, trans_rsector, \
        axial_rsector = parts.T
    parts = (trans_crystal, axial_crystal, trans_submodule, axial_submodule, trans_module, axial_module, trans_rsector,
             axial_rsector)
    if sort_by_detector_ids:
        order = torch.argsort(detector_ids_from_parts(*parts, info=info))
        parts = tuple(p[order] for p in parts)
    return parts


def detector_ids_from_parts(trans_crystal, axial_crystal, trans_submodule, axial_submodule, trans_module, axial_module,
                            trans_rsector, axial_rsector, info: Mapping):
    """Returns the detector id ``ring * NrCrystalsPerRing + crystal`` of crystals given by their part ids.

    The part ids are those :func:`crystal_parts` returns, in the same order: tensors, arrays or integers of the same
    shape.

    Args:
        trans_crystal (torch.Tensor | np.ndarray | int): Transaxial index of the crystal within its submodule.
        axial_crystal (torch.Tensor | np.ndarray | int): Axial index of the crystal within its submodule.
        trans_submodule (torch.Tensor | np.ndarray | int): Transaxial index of the submodule within its module.
        axial_submodule (torch.Tensor | np.ndarray | int): Axial index of the submodule within its module.
        trans_module (torch.Tensor | np.ndarray | int): Transaxial index of the module within its rsector.
        axial_module (torch.Tensor | np.ndarray | int): Axial index of the module within its rsector.
        trans_rsector (torch.Tensor | np.ndarray | int): Index of the rsector around the ring.
        axial_rsector (torch.Tensor | np.ndarray | int): Index of the rsector along the axis.
        info (Mapping): The scanner's ``info`` (:func:`scanner_info`).

    Returns:
        torch.Tensor | np.ndarray | int: The detector ids, of the type of the part ids.
    """
    ring = axial_crystal + info["crystalAxialNr"] * (axial_submodule + info["submoduleAxialNr"] * (
        axial_module + info["moduleAxialNr"] * axial_rsector))
    crystal = trans_crystal + info["crystalTransNr"] * (trans_submodule + info["submoduleTransNr"] * (
        trans_module + info["moduleTransNr"] * trans_rsector))
    return ring * info["NrCrystalsPerRing"] + crystal


def _centred(index, n, spacing):
    return (index - (n - 1) / 2) * spacing


def scanner_lut(info: Mapping) -> torch.Tensor:
    """Returns the position of every crystal in the object frame, in mm: the scanner lookup table.

    The object frame has LPS axes, with the origin at the centre of the rings. Within rsector 0, crystals sit on the
    plane at ``radius`` from the axis, at their transaxial offset along the rsector's face (centred on the rsector) and
    their axial offset along the axis (centred on the rings). Rsector ``k`` is at the azimuth
    ``rsector0_angle + transaxial_direction * 360 k / rsectorTransNr`` degrees, from +x towards +y; transaxial
    offsets run in the direction of ``transaxial_direction`` around the ring, and axial offsets in the direction of
    ``axial_direction``.

    Args:
        info (Mapping): The scanner's ``info`` (:func:`scanner_info`).

    Returns:
        torch.Tensor: float32 positions (x, y, z) in mm, ``[N_crystals, 3]``; row ``d`` is detector id ``d``.

    Raises:
        ValueError: If ``info`` has keys from before PyTomography 4 or lacks the frame keys: build it with
            :func:`scanner_info`.
    """
    _check_frame(info)
    tc, ac, ts, as_, tm, am, tr, ar = (p.double() for p in crystal_parts(info))
    ids = detector_ids_from_parts(*crystal_parts(info), info=info)
    axial = (_centred(ar, info["rsectorAxialNr"], info.get("rsectorAxialSpacing", 0.0))
             + _centred(am, info["moduleAxialNr"], info["moduleAxialSpacing"])
             + _centred(as_, info["submoduleAxialNr"], info["submoduleAxialSpacing"])
             + _centred(ac, info["crystalAxialNr"], info["crystalAxialSpacing"]))
    trans = (_centred(tm, info["moduleTransNr"], info["moduleTransSpacing"])
             + _centred(ts, info["submoduleTransNr"], info["submoduleTransSpacing"])
             + _centred(tc, info["crystalTransNr"], info["crystalTransSpacing"]))
    direction = info["transaxial_direction"]
    angle = math.radians(info["rsector0_angle"]) + direction * 2 * math.pi * tr / info["rsectorTransNr"]
    radius = float(info["radius"])
    x = radius * torch.cos(angle) - direction * trans * torch.sin(angle)
    y = radius * torch.sin(angle) + direction * trans * torch.cos(angle)
    z = info["axial_direction"] * axial
    lut = torch.empty((ids.shape[0], 3), dtype=torch.float64)
    lut[ids] = torch.stack([x, y, z], dim=1)
    return lut.to(torch.float32)


def frame_keys_after(matrix, rsector0_angle: float = 0.0, transaxial_direction: int = 1,
                     axial_direction: int = 1) -> dict:
    """Returns the frame keys of a ring in a new frame, from its keys in the old frame and the matrix between them.

    ``matrix`` is a rotation or a reflection that keeps the scanner's axis along z: the ring that the given keys
    describe in the old frame is described by the returned keys in the new one.

    Args:
        matrix (array-like): 3 x 3 orthogonal matrix: positions in the new frame are ``matrix @`` positions in the old
            one. Its third row and third column must be (0, 0, 1) or (0, 0, -1).
        rsector0_angle (float, optional): The ring's ``rsector0_angle`` in the old frame, in degrees.
            Defaults to 0.
        transaxial_direction (int, optional): Its ``transaxial_direction`` in the old frame (1 or -1). Defaults to 1.
        axial_direction (int, optional): Its ``axial_direction`` in the old frame (1 or -1). Defaults to 1.

    Returns:
        dict: ``rsector0_angle`` (degrees, in [0, 360)), ``transaxial_direction`` and ``axial_direction`` in the
        new frame.

    Raises:
        ValueError: If ``matrix`` is not a 3 x 3 orthogonal matrix, or does not keep the scanner's axis along z.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3) or not np.allclose(m @ m.T, np.eye(3), atol=1e-9):
        raise ValueError(f"matrix must be a 3 x 3 rotation or reflection, not {matrix!r}")
    if not (np.allclose(m[2, :2], 0, atol=1e-9) and np.allclose(m[:2, 2], 0, atol=1e-9)):
        raise ValueError("the scanner's axis must stay along z (the patient's head-foot axis)")
    a = m[:2, :2]
    alpha = math.degrees(math.atan2(a[1, 0], a[0, 0]))
    if np.linalg.det(a) > 0:  # a turn by alpha
        angle, direction = alpha + rsector0_angle, transaxial_direction
    else:  # a turn by alpha after mirroring y: azimuths change sign, and the crystals run the other way
        angle, direction = alpha - rsector0_angle, -transaxial_direction
    angle = angle % 360.0
    if math.isclose(angle, 360.0, abs_tol=1e-9) or math.isclose(angle, 0.0, abs_tol=1e-9):
        angle = 0.0
    return {"rsector0_angle": angle, "transaxial_direction": int(direction),
            "axial_direction": int(round(m[2, 2])) * int(axial_direction)}


def pair_index(low, high, n):
    """Returns the index of the crystal pair (``low``, ``high``), ``low < high``, in ``torch.combinations`` order.

    Plain arithmetic, so that numba's compiled loops can call it. On tensors and arrays use :func:`crystal_pair_index`,
    which computes in 64-bit integers.

    Args:
        low (int | torch.Tensor | np.ndarray): The lower crystal id of each pair.
        high (int | torch.Tensor | np.ndarray): The higher crystal id of each pair.
        n (int): Number of crystals.

    Returns:
        int | torch.Tensor | np.ndarray: Index of each pair in ``torch.combinations(torch.arange(n), 2)``.
    """
    return low * (2 * n - low - 1) // 2 + (high - low - 1)


def crystal_pair_index(crystal_1, crystal_2, n_crystals: int):
    """Returns the index of each crystal pair ``(crystal_1, crystal_2)`` in ``torch.combinations`` order.

    The order is that of ``torch.combinations(torch.arange(n_crystals), 2)``: the order of ``weights_sensitivity``, of
    :func:`~pytomography.io.PET.shared.crystal_pair_blocks` and of every reader's normalisation weights. The index is
    computed in 64-bit integers: in float32 the products round (for the 28,672 crystals of the Siemens Biograph mMR,
    about 9 pairs in 10 would get a neighbouring pair's index), and in int32 they overflow for larger scanners.

    Args:
        crystal_1 (torch.Tensor | np.ndarray): Lower crystal id of each pair.
        crystal_2 (torch.Tensor | np.ndarray): Higher crystal id of each pair (``crystal_1 < crystal_2``; this is not
            checked).
        n_crystals (int): Number of crystals.

    Returns:
        torch.Tensor | np.ndarray: int64 index of each pair: a tensor if either input is a tensor, else an array.
    """
    if isinstance(crystal_1, torch.Tensor) or isinstance(crystal_2, torch.Tensor):
        return pair_index(torch.as_tensor(crystal_1).to(torch.long), torch.as_tensor(crystal_2).to(torch.long),
                          int(n_crystals))
    return pair_index(np.asarray(crystal_1, dtype=np.int64), np.asarray(crystal_2, dtype=np.int64), int(n_crystals))


def rsector_difference(detector_ids: torch.Tensor, info: Mapping) -> torch.Tensor:
    """Returns how many rsectors apart, around the ring, the two crystals of each event are.

    Args:
        detector_ids (torch.Tensor): Events, ``[N, 2]`` detector ids, or ``[N, 3]`` with the TOF bin.
        info (Mapping): The scanner's ``info`` (:func:`scanner_info`).

    Returns:
        torch.Tensor: int64 ``[N]``: each event's rsector difference, from 0 to ``rsectorTransNr // 2``.
    """
    per_rsector = info["crystalTransNr"] * info["submoduleTransNr"] * info["moduleTransNr"]
    sectors = (detector_ids[:, :2].to(torch.long) % info["NrCrystalsPerRing"]) // per_rsector
    difference = (sectors[:, 0] - sectors[:, 1]).abs()
    return torch.minimum(difference, info["rsectorTransNr"] - difference)


def keep_rsector_difference(detector_ids: torch.Tensor, info: Mapping) -> torch.Tensor:
    """Returns the events whose two crystals are at least ``info['min_rsector_difference']`` rsectors apart.

    ``min_rsector_difference`` is an event filter, which every reader applies. When it is 0 (or missing), the events
    are returned as they are.

    Args:
        detector_ids (torch.Tensor): Events, ``[N, 2]`` detector ids, or ``[N, 3]`` with the TOF bin.
        info (Mapping): The scanner's ``info`` (:func:`scanner_info`).

    Returns:
        torch.Tensor: The events kept, in their order.
    """
    if not info.get("min_rsector_difference", 0):
        return detector_ids
    return detector_ids[rsector_difference(detector_ids, info) >= info["min_rsector_difference"]]
