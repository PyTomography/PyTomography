"""The scanner of PET data in PyTomography's standard form: the ``info`` dictionary and the position of every crystal.

Every PET reader describes its scanner as a parametric ring with :func:`scanner_info`, and the projectors, sinograms,
randoms and scatter estimates work from that description alone. :func:`scanner_lut` places every crystal in the object
frame: LPS axes (x to the patient's left, y posterior, z superior), in mm, with the origin at the centre of the rings,
which is the centre of the reconstruction grid.

**The ring.** Crystals are grouped into modules, submodules and rsectors, as GATE's cylindrical PET system does:

* ``{crystal, submodule, module}{Trans, Axial}Nr``: the number of each part around the ring (``Trans``) and along the
  axis (``Axial``) within its parent; ``rsectorTransNr`` rsectors make up the ring, and ``rsectorAxialNr`` sit along
  the axis;
* ``{crystal, submodule, module, rsector}{Trans, Axial}Spacing``: their centre-to-centre pitch, in mm (rsectors around
  the ring are placed by angle, so they have no transaxial spacing);
* ``radius``: the distance from the axis to the LOR end points of rsector 0, in mm: the centre of the module's face
  plus ``doi_mm``, the depth of interaction inside the crystal that the LORs end at;
* ``NrCrystalsPerRing`` and ``NrRings``, computed from the counts.

A detector id is ``ring * NrCrystalsPerRing + crystal``, where ``crystal`` counts the crystals of a ring (crystal,
then submodule, module, rsector, fastest first) and ``ring`` the rings (crystal, submodule, module, rsector along the
axis).

**The frame keys** say where that ring lies in the object frame:

* ``rsector0_angle_deg``: the azimuth of the centre of rsector 0, from +x towards +y;
* ``transaxial_direction``: +1 when crystal numbers run from +x towards +y (counter-clockwise seen from +z), -1 the
  other way;
* ``axial_direction``: +1 when ring numbers run towards +z (superior), -1 towards -z.

They replace ``firstCrystalAxis`` and the negation of z that :func:`scanner_lut` applied to every scanner before
PyTomography 4 (``firstCrystalAxis=1`` is ``rsector0_angle_deg=0, transaxial_direction=1``; ``firstCrystalAxis=0`` is
``rsector0_angle_deg=90, transaxial_direction=-1``; the old negation of z is ``axial_direction=-1``). Each reader sets
them from what it knows of its source: GATE from ``world_axes``, PETSIRD from the patient's position in the gantry, GE
from its vendor rule.

**The energy window**: ``energy_window_low_keV``, ``energy_window_high_keV`` and ``energy_resolution`` (the FWHM of the
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
#: frame keys and their defaults (GATE's world: rsector 0 on +x, crystals counter-clockwise, rings towards +z)
FRAME_KEYS = {"rsector0_angle_deg": 0.0, "transaxial_direction": 1, "axial_direction": 1}
#: energy keys: None where the source does not give them
ENERGY_KEYS = ("energy_window_low_keV", "energy_window_high_keV", "energy_resolution")
#: keys of ``info`` before PyTomography 4, and what replaced them
_RENAMED = {
    "firstCrystalAxis": "the frame keys rsector0_angle_deg, transaxial_direction and axial_direction: "
                        "firstCrystalAxis=1 is rsector0_angle_deg=0, transaxial_direction=1, and firstCrystalAxis=0 is "
                        "rsector0_angle_deg=90, transaxial_direction=-1; the z that scanner LUTs used to negate is "
                        "axial_direction=-1",
    "energy_window_low": "energy_window_low_keV",
    "energy_window_high": "energy_window_high_keV",
    "mean_interaction_depth": "doi_mm (radius includes it)",
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
    """The ``info`` dictionary of a PET scanner, PyTomography's description of it as a parametric ring (see the top of
    this module), with every value checked and converted to a Python ``int`` or ``float``.

    Every PET reader builds its ``info`` with this function. Use it to describe a scanner by hand, or to change keys of
    an ``info`` a reader returned::

        info = io.PET.scanner_info(crystalTransNr=8, crystalTransSpacing=4.0, crystalAxialNr=8,
                                   crystalAxialSpacing=4.0, moduleTransNr=1, moduleAxialNr=8, moduleAxialSpacing=32.25,
                                   rsectorTransNr=56, radius=337.0, doi_mm=9.0)
        info = io.PET.scanner_info(info, min_rsector_difference=2)
        info = io.PET.scanner_info("discovery_MI")      # PyTomography's table of clinical scanners

    Args:
        scanner (str | Mapping | None, optional): the name of a scanner in PyTomography's table
            (``data/pet_scanner_info.txt``: ``'discovery_MI'``), or an ``info`` dictionary to start from. Defaults to
            None.
        **keys: keys of ``info``, which replace those of ``scanner``. ``submodule*`` keys default to one submodule,
            ``rsectorAxialNr`` to 1, ``doi_mm``, ``crystal_length`` and ``min_rsector_difference`` to 0, the frame
            keys to GATE's world (``rsector0_angle_deg=0``, ``transaxial_direction=1``, ``axial_direction=1``) and the
            energy keys to None. Other keys (strings or numbers) are kept as they are.

    Raises:
        ValueError: if a key is missing, has a value it cannot have, or is a key of ``info`` from before PyTomography
            4 (``firstCrystalAxis``, ``energy_window_low``, ...): the message says what replaced it.

    Returns:
        dict: the scanner's ``info``.
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
    for key in ("doi_mm", "crystal_length"):
        info[key] = _as_float(key, given.get(key, 0.0)) or 0.0
        if info[key] < 0:
            raise ValueError(f"info[{key!r}] cannot be negative ({info[key]})")
    info["min_rsector_difference"] = _as_int("min_rsector_difference", given.get("min_rsector_difference", 0))
    if not 0 <= info["min_rsector_difference"] <= info["rsectorTransNr"] // 2:
        raise ValueError(f"info['min_rsector_difference'] must be between 0 and half the rsectors around the ring "
                         f"({info['rsectorTransNr'] // 2}), not {info['min_rsector_difference']}")
    info["rsector0_angle_deg"] = _as_float("rsector0_angle_deg", given.get("rsector0_angle_deg", 0.0)) or 0.0
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
    """The part ids of every crystal of the scanner: (trans crystal, axial crystal, trans submodule, axial submodule,
    trans module, axial module, trans rsector, axial rsector), as 1D long tensors, in the order of
    ``torch.cartesian_prod`` of the parts, or of the detector ids with ``sort_by_detector_ids``."""
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
    """The detector id ``ring * NrCrystalsPerRing + crystal`` of crystals given by their part ids (tensors or arrays,
    in the order :func:`crystal_parts` returns them)."""
    ring = axial_crystal + info["crystalAxialNr"] * (axial_submodule + info["submoduleAxialNr"] * (
        axial_module + info["moduleAxialNr"] * axial_rsector))
    crystal = trans_crystal + info["crystalTransNr"] * (trans_submodule + info["submoduleTransNr"] * (
        trans_module + info["moduleTransNr"] * trans_rsector))
    return ring * info["NrCrystalsPerRing"] + crystal


def _centred(index, n, spacing):
    return (index - (n - 1) / 2) * spacing


def scanner_lut(info: Mapping) -> torch.Tensor:
    """The position of every crystal: the scanner lookup table, in the object frame (LPS axes, mm, origin at the centre
    of the rings).

    Within rsector 0, crystals sit at ``radius`` from the axis, at their transaxial offset along the rsector's face and
    their axial offset along the axis, both centred on the rsector. Rsector ``k`` is at the azimuth
    ``rsector0_angle_deg + transaxial_direction * 360 k / rsectorTransNr`` degrees; transaxial offsets run in the
    direction of ``transaxial_direction`` around the ring and axial offsets in the direction of ``axial_direction``.

    Args:
        info (Mapping): the scanner's ``info`` (:func:`scanner_info`).

    Returns:
        torch.Tensor: float32 positions, ``[N_crystals, 3]``; row ``d`` is detector id ``d``.
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
    angle = math.radians(info["rsector0_angle_deg"]) + direction * 2 * math.pi * tr / info["rsectorTransNr"]
    radius = float(info["radius"])
    x = radius * torch.cos(angle) - direction * trans * torch.sin(angle)
    y = radius * torch.sin(angle) + direction * trans * torch.cos(angle)
    z = info["axial_direction"] * axial
    lut = torch.empty((ids.shape[0], 3), dtype=torch.float64)
    lut[ids] = torch.stack([x, y, z], dim=1)
    return lut.to(torch.float32)


def frame_keys_after(matrix, rsector0_angle_deg: float = 0.0, transaxial_direction: int = 1,
                     axial_direction: int = 1) -> dict:
    """The frame keys of a ring after its crystals' positions are taken through ``matrix`` (3 x 3, a rotation or
    reflection that keeps the scanner's axis along z: a ring described in one frame by the given keys, seen in
    another).

    Args:
        matrix: 3 x 3 orthogonal matrix, positions in the new frame = ``matrix @`` positions in the old one. Its third
            row and column must be (0, 0, +-1).
        rsector0_angle_deg (float, optional): the ring's ``rsector0_angle_deg`` in the old frame. Defaults to 0.
        transaxial_direction (int, optional): its ``transaxial_direction`` in the old frame. Defaults to 1.
        axial_direction (int, optional): its ``axial_direction`` in the old frame. Defaults to 1.

    Returns:
        dict: ``rsector0_angle_deg`` (in [0, 360)), ``transaxial_direction`` and ``axial_direction`` in the new frame.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3) or not np.allclose(m @ m.T, np.eye(3), atol=1e-9):
        raise ValueError(f"matrix must be a 3 x 3 rotation or reflection, not {matrix!r}")
    if not (np.allclose(m[2, :2], 0, atol=1e-9) and np.allclose(m[:2, 2], 0, atol=1e-9)):
        raise ValueError("the scanner's axis must stay along z (the patient's head-foot axis)")
    a = m[:2, :2]
    alpha = math.degrees(math.atan2(a[1, 0], a[0, 0]))
    if np.linalg.det(a) > 0:  # a turn by alpha
        angle, direction = alpha + rsector0_angle_deg, transaxial_direction
    else:  # a turn by alpha after mirroring y: azimuths change sign, and the crystals run the other way
        angle, direction = alpha - rsector0_angle_deg, -transaxial_direction
    angle = angle % 360.0
    if math.isclose(angle, 360.0, abs_tol=1e-9) or math.isclose(angle, 0.0, abs_tol=1e-9):
        angle = 0.0
    return {"rsector0_angle_deg": angle, "transaxial_direction": int(direction),
            "axial_direction": int(round(m[2, 2])) * int(axial_direction)}


def pair_index(low, high, n):
    """The index of the crystal pair (``low``, ``high``), ``low < high``, in the order of
    ``torch.combinations(torch.arange(n), 2)``: plain arithmetic, for numba's compiled loops (``_petsird_binary``).
    Use :func:`crystal_pair_index` on tensors and arrays."""
    return low * (2 * n - low - 1) // 2 + (high - low - 1)


def crystal_pair_index(crystal_1, crystal_2, n_crystals: int):
    """Index of each pair of crystals ``(crystal_1, crystal_2)``, ``crystal_1 < crystal_2``, in the order of
    ``torch.combinations(torch.arange(n_crystals), 2)`` (the order of ``weights_sensitivity``, of
    :func:`~pytomography.io.PET.shared.crystal_pair_blocks` and of every reader's normalisation weights). Computed with
    64-bit integers: in float32 the products round, and for the mMR (28,672 crystals) about 9 in 10 pairs were given a
    neighbouring pair's index; in int32 they overflow for larger scanners.

    Args:
        crystal_1 (torch.Tensor | np.ndarray): Lower crystal of each pair.
        crystal_2 (torch.Tensor | np.ndarray): Higher crystal of each pair.
        n_crystals (int): Number of crystals.

    Returns:
        torch.Tensor | np.ndarray: Index of each pair (int64), a tensor for tensors and an array otherwise.
    """
    if isinstance(crystal_1, torch.Tensor) or isinstance(crystal_2, torch.Tensor):
        return pair_index(torch.as_tensor(crystal_1).to(torch.long), torch.as_tensor(crystal_2).to(torch.long),
                          int(n_crystals))
    return pair_index(np.asarray(crystal_1, dtype=np.int64), np.asarray(crystal_2, dtype=np.int64), int(n_crystals))


def rsector_difference(detector_ids: torch.Tensor, info: Mapping) -> torch.Tensor:
    """How many rsectors apart, around the ring, the two crystals of each event are (0 to ``rsectorTransNr // 2``)."""
    per_rsector = info["crystalTransNr"] * info["submoduleTransNr"] * info["moduleTransNr"]
    sectors = (detector_ids[:, :2].to(torch.long) % info["NrCrystalsPerRing"]) // per_rsector
    difference = (sectors[:, 0] - sectors[:, 1]).abs()
    return torch.minimum(difference, info["rsectorTransNr"] - difference)


def keep_rsector_difference(detector_ids: torch.Tensor, info: Mapping) -> torch.Tensor:
    """The events whose two crystals are at least ``info['min_rsector_difference']`` rsectors apart around the ring
    (all of them when it is 0): ``min_rsector_difference`` is an event filter, which every reader applies."""
    if not info.get("min_rsector_difference", 0):
        return detector_ids
    return detector_ids[rsector_difference(detector_ids, info) >= info["min_rsector_difference"]]
