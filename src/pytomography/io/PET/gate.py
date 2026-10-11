"""Read GATE simulations of PET scanners: the coincidences of GATE's ROOT output, with the scanner described by the
simulation's geometry macro.

:func:`read_gate` returns the events in PyTomography's standard form, with the object grid placed in the patient and
the projection metadata (:class:`~pytomography.metadata.PET.PETLMProjMeta`)::

    events, object_meta, proj_meta = io.PET.read_gate(root_files, 'mMR_Geometry.mac', doi_mm=9,
                                                      object_meta=ObjectMeta(dr=(2, 2, 2), shape=(128, 128, 96)))

GATE's world frame is whatever the simulation made it: the phantom sits in it wherever the simulation placed it. Two
arguments say how: ``world_axes``, the patient direction (L or R, P or A, S or I) that GATE's +x, +y and +z point to,
and ``world_origin``, where GATE's origin is in the patient's LPS coordinates (mm). The datasets PyTomography hosts
record theirs in the dataset registry (``pytomography/datasets/registry.py``, a dataset's ``geometry``), and the reader
uses them when it reads those files.

:func:`gate_normalization` computes the normalisation weights of every crystal pair from a simulated calibration
scan of a cylindrical shell.
"""
from __future__ import annotations

import math
import os
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import uproot

import pytomography
from pytomography.io import _vendors
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta
from pytomography.utils.memory import block_size
from .scanner import (crystal_parts, detector_ids_from_parts, frame_keys_after, keep_rsector_difference,
                      scanner_info, scanner_lut)
from .shared import SPEED_OF_LIGHT_MM_PER_PS, crystal_pair_blocks
from ._standard import default_grid, normalization_weights, reader_object_meta

#: Patient directions, as unit vectors in LPS
_DIRECTIONS = {"L": (1, 0, 0), "R": (-1, 0, 0), "P": (0, 1, 0), "A": (0, -1, 0), "S": (0, 0, 1), "I": (0, 0, -1)}
#: GATE length units in mm
_UNITS = {"nm": 1e-6, "um": 1e-3, "mm": 1.0, "cm": 10.0, "m": 1000.0}
#: what ``events`` can ask for
_KINDS = ("trues", "scatters", "randoms")


def world_matrix(world_axes: str) -> np.ndarray:
    """The 3 x 3 matrix that takes positions in GATE's world (mm) to the patient's LPS axes: column ``i`` is the
    patient direction GATE's axis ``i`` points to.

    Args:
        world_axes (str): three letters, one per GATE axis (x, y, z), from L or R, P or A, and S or I: ``'LPS'`` when
            GATE's axes are the patient's, ``'LPI'`` when GATE's z runs towards the feet. GATE's z, the scanner's axis,
            must be the patient's head-foot axis.

    Returns:
        np.ndarray: the matrix (a rotation or a reflection).
    """
    letters = str(world_axes).upper()
    if len(letters) != 3 or any(c not in _DIRECTIONS for c in letters):
        raise ValueError(f"world_axes must be three letters from L/R, P/A and S/I (e.g. 'LPS'), not {world_axes!r}")
    matrix = np.array([_DIRECTIONS[c] for c in letters], dtype=np.float64).T
    if abs(abs(np.linalg.det(matrix)) - 1) > 1e-9:
        raise ValueError(f"world_axes {world_axes!r} names one patient axis twice")
    if letters[2] not in "SI":
        raise ValueError(f"GATE's z is the scanner's axis, which PyTomography takes as the patient's head-foot axis: "
                         f"the third letter of world_axes must be S or I, not {letters[2]!r}")
    return matrix


def _parse_macro(macro) -> dict:
    """The commands of a GATE macro: ``{command: [arguments of each time it is given]}``, comments dropped."""
    commands: dict = {}
    for line in Path(macro).read_text(errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line.startswith("/"):
            continue
        command, *arguments = line.split()
        commands.setdefault(command, []).append(arguments)
    return commands


def _lengths(arguments: Sequence[str], n: int, command: str) -> list[float]:
    """The first ``n`` numbers of a GATE command, in mm (a unit after them, mm if there is none)."""
    try:
        values = [float(a) for a in arguments[:n]]
    except ValueError:
        raise ValueError(f"cannot read {n} numbers from {command} {' '.join(arguments)}") from None
    unit = arguments[n] if len(arguments) > n else "mm"
    if unit not in _UNITS:
        raise ValueError(f"unknown length unit {unit!r} in {command} {' '.join(arguments)}")
    return [v * _UNITS[unit] for v in values]


def _last(commands: dict, command: str):
    return commands[command][-1] if command in commands else None


def _macro_info(macro, doi_mm: float = 0.0, crystal_volume: str = "crystal") -> tuple[dict, float]:
    """The ring of a GATE cylindrical PET system, as ``info`` keys in GATE's world, from its geometry macro; and the
    axial position of the rings' centre in GATE's world (mm)."""
    commands = _parse_macro(macro)
    levels = ("crystal", "submodule", "module", "rsector")
    translation = np.zeros(3)
    for level in levels:
        arguments = _last(commands, f"/gate/{level}/placement/setTranslation")
        if arguments is not None:
            translation += _lengths(arguments, 3, f"/gate/{level}/placement/setTranslation")
    if abs(translation[1]) > 1e-6:
        raise NotImplementedError("PyTomography reads GATE scanners whose rsector is placed along +x (its translation"
                                  f" has y = {translation[1]} mm)")
    if translation[0] <= 0:
        raise ValueError(f"the crystals of this macro are not placed away from the axis along +x ({translation} mm)")
    length = _last(commands, f"/gate/{crystal_volume}/geometry/setXLength")
    if length is None:
        raise ValueError(f"the macro does not give the depth of its crystals (/gate/{crystal_volume}/geometry/setXLength)")
    crystal_length = _lengths(length, 1, "setXLength")[0]
    keys = {"crystal_length": crystal_length, "radius": translation[0] - crystal_length / 2 + doi_mm, "doi_mm": doi_mm}
    first_angle = 0.0
    for level in levels:
        repeaters = [a[0] for a in commands.get(f"/gate/{level}/repeaters/insert", [])]
        if not repeaters and level == "submodule":
            keys.update(submoduleTransNr=1, submoduleTransSpacing=0.0, submoduleAxialNr=1, submoduleAxialSpacing=0.0)
        for repeater in repeaters:
            base = f"/gate/{level}/{repeater}"
            if repeater == "cubicArray":
                numbers = [int(float(_last(commands, f"{base}/setRepeatNumber{c}")[0]))
                           if f"{base}/setRepeatNumber{c}" in commands else 1 for c in "XYZ"]
                vector = _lengths(_last(commands, f"{base}/setRepeatVector"), 3, f"{base}/setRepeatVector") \
                    if f"{base}/setRepeatVector" in commands else [0.0, 0.0, 0.0]
                if numbers[0] > 1:
                    raise NotImplementedError(f"{level}s repeated along x (the depth of the crystals) are not supported")
                keys.update({f"{level}TransNr": numbers[1], f"{level}TransSpacing": vector[1],
                             f"{level}AxialNr": numbers[2], f"{level}AxialSpacing": vector[2]})
            elif repeater == "linear":
                number = int(float(_last(commands, f"{base}/setRepeatNumber")[0]))
                vector = _lengths(_last(commands, f"{base}/setRepeatVector"), 3, f"{base}/setRepeatVector")
                if abs(vector[0]) > 1e-9 or abs(vector[1]) > 1e-9:
                    raise NotImplementedError(f"a linear repeater of {level}s must repeat along z, not {vector}")
                keys.update({f"{level}AxialNr": number, f"{level}AxialSpacing": vector[2]})
                keys.setdefault(f"{level}TransNr", 1)
            elif repeater == "ring":
                keys.update({f"{level}TransNr": int(float(_last(commands, f"{base}/setRepeatNumber")[0])),
                             f"{level}AxialNr": 1})
                if f"{base}/setFirstAngle" in commands:
                    arguments = _last(commands, f"{base}/setFirstAngle")
                    first_angle = float(arguments[0]) * (1.0 if len(arguments) < 2 or arguments[1].startswith("deg")
                                                        else 180 / math.pi)
                if f"{base}/setAngularSpan" in commands:
                    arguments = _last(commands, f"{base}/setAngularSpan")
                    span = float(arguments[0]) * (1.0 if len(arguments) < 2 or arguments[1].startswith("deg")
                                                 else 180 / math.pi)
                    if not math.isclose(span, 360.0):
                        raise NotImplementedError(f"a ring of rsectors over {span} degrees is not a full ring")
            else:
                raise NotImplementedError(f"GATE's {repeater!r} repeater ({level}) is not supported")
    keys["rsector0_angle_deg"] = first_angle
    for level in levels[1:]:
        keys.setdefault(f"{level}TransNr", 1)
        keys.setdefault(f"{level}AxialNr", 1)
    return keys, float(translation[2])


def _registry_placement(root_files: Sequence) -> dict | None:
    """How the phantom of a dataset PyTomography hosts sat in GATE's world, from the dataset registry: the ``gate``
    entry of the dataset's ``geometry``, when the first ROOT file is one of the dataset's (by its name and size, or by
    its folder in the data folder)."""
    from pytomography.datasets import data_dir, registry
    first = Path(root_files[0])
    try:
        size = first.stat().st_size
        resolved = first.resolve()
    except OSError:
        return None
    for name, entry in registry.DATASETS.items():
        placement = entry.get("geometry", {}).get("gate")
        if not placement:
            continue
        for path, pinned in entry.get("files", {}).items():
            if Path(path).name == first.name and int(pinned[0]) == size:
                return placement
        try:
            if resolved.is_relative_to((data_dir() / name).resolve()):
                return placement
        except OSError:
            continue
    return None


def _branch(tree, name: str) -> np.ndarray:
    return tree[name].array(library="np")


def _event_kinds(tree) -> dict:
    """Which coincidences of a tree are randoms (their two photons come from different annihilations) and scatters
    (not randoms, with a Compton or Rayleigh interaction in the phantom), as boolean arrays."""
    random = ~((_branch(tree, "sourcePosX1") == _branch(tree, "sourcePosX2"))
               & (_branch(tree, "sourcePosY1") == _branch(tree, "sourcePosY2"))
               & (_branch(tree, "sourcePosZ1") == _branch(tree, "sourcePosZ2")))
    scattered = np.zeros(random.shape, dtype=bool)
    for name in ("comptonPhantom1", "comptonPhantom2", "RayleighPhantom1", "RayleighPhantom2"):
        scattered |= _branch(tree, name) != 0
    return {"randoms": random, "scatters": scattered & ~random, "trues": ~scattered & ~random}


def _crystal_ids(tree, info: dict, j: int) -> np.ndarray:
    """Detector ids of single ``j`` (1 or 2) of every coincidence, from GATE's copy numbers (``crystalID`` is the
    cubic array's copy number, transaxial fastest)."""
    rsector, module, submodule, crystal = (_branch(tree, f"{level}ID{j}").astype(np.int64)
                                           for level in ("rsector", "module", "submodule", "crystal"))
    parts = []
    for ids, level in ((crystal, "crystal"), (submodule, "submodule"), (module, "module"), (rsector, "rsector")):
        parts += [ids % info[f"{level}TransNr"], ids // info[f"{level}TransNr"]]
    return detector_ids_from_parts(*parts, info=info)


def _kinds(events) -> tuple[str, Sequence[str] | None]:
    """The tree and the kinds of coincidence ``events`` asks for (None: all of them)."""
    if isinstance(events, str):
        if events == "prompts":
            return "Coincidences", None
        if events == "delayeds":
            return "delay", None
        events = (events,)
    kinds = tuple(events)
    unknown = [k for k in kinds if k not in _KINDS]
    if unknown or not kinds:
        raise ValueError(f"events must be 'prompts', 'delayeds', or one or more of {_KINDS}, not {events!r}")
    return "Coincidences", kinds


def _read_events(root_files: Sequence, info: dict, tree_name: str, kinds, tof_meta: PETTOFMeta | None) -> torch.Tensor:
    """int32 events [N, 2] (or [N, 3] with TOF bins) of the ROOT files, the coincidences of the kinds asked for."""
    if tof_meta is not None:
        edges = tof_meta.bin_edges.double().numpy()
    columns = []
    for path in root_files:
        with uproot.open(path) as f:
            tree = f[tree_name]
            keep = None
            if kinds is not None:
                flags = _event_kinds(tree)
                keep = np.zeros(flags["trues"].shape, dtype=bool)
                for kind in kinds:
                    keep |= flags[kind]
            ids = [_crystal_ids(tree, info, j) for j in (1, 2)]
            if tof_meta is not None:
                # (t1 - t2) c / 2: the position along the LOR from its midpoint towards column 1 (the second single)
                position = (_branch(tree, "time1") - _branch(tree, "time2")) * 1e12 * SPEED_OF_LIGHT_MM_PER_PS / 2
                tof_bin = np.digitize(position, edges) - 1
                inside = (tof_bin >= 0) & (tof_bin < tof_meta.num_bins)
                keep = inside if keep is None else keep & inside
                ids.append(tof_bin)
            block = np.stack(ids, axis=1)
            if keep is not None:
                block = block[keep]
            columns.append(block.astype(np.int32))
    width = 2 if tof_meta is None else 3
    events = np.concatenate(columns) if columns else np.zeros((0, width), dtype=np.int32)
    return torch.from_numpy(events)


def read_gate(
    root_files: str | os.PathLike | Sequence,
    macro: str | os.PathLike,
    *,
    object_meta: ObjectMeta | None = None,
    events: str | Sequence[str] = "prompts",
    tof_meta: PETTOFMeta | None = None,
    world_axes: str | None = None,
    world_origin: Sequence[float] | None = None,
    doi_mm: float = 0.0,
    energy_window_keV: Sequence[float] | None = None,
    energy_resolution: float | None = None,
    normalization=None,
    min_rsector_difference: int = 0,
    crystal_volume: str = "crystal",
) -> tuple[torch.Tensor, ObjectMeta, PETLMProjMeta]:
    r"""The coincidences of a GATE simulation of a cylindrical PET scanner, in PyTomography's standard form.

    The scanner is read from the simulation's geometry macro: a ring of rsectors (GATE's ``ring`` repeater), each made
    of modules, submodules and crystals (``cubicArray`` or ``linear`` repeaters), placed along +x. Each event's two
    detector ids come from GATE's copy numbers. The scanner lookup table is in the object frame: GATE's world turned
    to the patient's LPS axes by ``world_axes``. The grid is centred on the rings' centre, and ``object_meta.centre``
    says where that is in the patient.

    Args:
        root_files (str | os.PathLike | Sequence): GATE's ROOT output, one file or several (their events are joined
            in order).
        macro (str | os.PathLike): the GATE macro that describes the scanner (``.mac``).
        object_meta (ObjectMeta, optional): the reconstruction grid (``dr`` and ``shape``); its centre and frame are
            set by the reader. Defaults to None: voxels half the crystal pitch, a square covering 80 % of the ring's
            diameter, and every ring.
        events (str | Sequence[str], optional): ``'prompts'`` (every coincidence), ``'delayeds'`` (the delayed
            coincidence window, GATE's ``delay`` tree), or one or more of ``'trues'``, ``'scatters'`` (not random,
            with a Compton or Rayleigh interaction in the phantom) and ``'randoms'`` (two photons from different
            annihilations) of the prompts. Defaults to ``'prompts'``.
        tof_meta (PETTOFMeta, optional): TOF bins to sort the events into, by (t1 - t2) c / 2; events outside the bins
            are dropped. Defaults to None: no TOF.
        world_axes (str, optional): the patient directions GATE's +x, +y and +z point to, e.g. ``'LPS'`` (GATE's world
            is the patient's frame) or ``'LPI'`` (GATE's z runs towards the feet). Defaults to None: the placement the
            dataset registry records for a hosted dataset, else ``'LPS'``.
        world_origin (Sequence[float], optional): where GATE's origin lies in the patient's LPS coordinates (mm). With
            it, ``object_meta.frame`` is ``'patient'``. Defaults to None: the registry's placement for a hosted
            dataset; else GATE's origin is the origin, with ``frame='scanner'``.
        doi_mm (float, optional): depth of interaction: the LORs end this far inside the crystals' inner face (mm).
            Defaults to 0.
        energy_window_keV (Sequence[float], optional): the simulation's energy window (low, high) in keV, which is in
            the digitizer, not in the output (for the scatter estimate). Defaults to None.
        energy_resolution (float, optional): the digitizer's energy resolution, the FWHM at 511 keV as a fraction of
            511 keV. Defaults to None.
        normalization (str | os.PathLike | np.ndarray | torch.Tensor, optional): normalisation weights of every
            crystal pair, in ``torch.combinations`` order (a ``.npy`` file, or the array; see
            :func:`gate_normalization`). Pairs whose weight is NaN (no calibration data) get 0, with a warning. They
            become ``proj_meta.weights_sensitivity``. Defaults to None.
        min_rsector_difference (int, optional): keep only events whose crystals are at least this many rsectors apart
            around the ring (GATE's ``setMinSectorDifference``). Defaults to 0.
        crystal_volume (str, optional): the GATE volume of the crystals, whose ``setXLength`` is their depth. Defaults
            to ``'crystal'``.

    Returns:
        tuple[torch.Tensor, ObjectMeta, PETLMProjMeta]: the events (int32 ``[N, 2]``, or ``[N, 3]`` with the TOF bin;
        column 0 is the first single), the object grid (modality ``'PT'``, units ``'counts'``) and the projection
        metadata (the events, ``info``, ``tof_meta`` and the normalisation weights).
    """
    if isinstance(root_files, (str, os.PathLike)):
        root_files = [root_files]
    root_files = list(root_files)
    if not root_files:
        raise ValueError("no ROOT files given")
    gate_keys, axial_centre = _macro_info(macro, doi_mm=doi_mm, crystal_volume=crystal_volume)
    placement = _registry_placement(root_files) if world_axes is None and world_origin is None else None
    if placement is not None:
        world_axes, world_origin = placement["world_axes"], placement.get("world_origin")
    _vendors.use("gate-world-frame", str(root_files[0]))
    matrix = world_matrix(world_axes or "LPS")
    frame = frame_keys_after(matrix, gate_keys.pop("rsector0_angle_deg"), 1, 1)
    low, high = (None, None) if energy_window_keV is None else (float(energy_window_keV[0]), float(energy_window_keV[1]))
    info = scanner_info(gate_keys, **frame, min_rsector_difference=min_rsector_difference,
                        energy_window_low_keV=low, energy_window_high_keV=high, energy_resolution=energy_resolution)
    tree, kinds = _kinds(events)
    detector_ids = keep_rsector_difference(_read_events(root_files, info, tree, kinds, tof_meta), info)
    centre = matrix @ np.array([0.0, 0.0, axial_centre]) + (np.zeros(3) if world_origin is None else np.asarray(world_origin, dtype=np.float64))
    object_meta = reader_object_meta(object_meta, default_grid(info), centre=centre,
                                     frame="scanner" if world_origin is None else "patient", modality="PT", units="counts")
    weights = None if normalization is None else normalization_weights(normalization, info)
    proj_meta = PETLMProjMeta(detector_ids, info, tof_meta=tof_meta, weights_sensitivity=weights)
    # how GATE's world sat in the patient, for images in GATE's world (gate_image)
    proj_meta.world_axes = "LPS" if world_axes is None else str(world_axes).upper()
    proj_meta.world_origin = None if world_origin is None else tuple(float(v) for v in world_origin)
    return detector_ids, object_meta, proj_meta


def gate_image(source, proj_meta: PETLMProjMeta, *, units: str | None = None,
               placement_mm: Sequence[float] = (0.0, 0.0, 0.0)):
    """An image in a GATE simulation's world (a voxelised phantom or attenuation map, e.g. GATE/STIR Interfile ``.hv``),
    placed in the patient frame of :func:`read_gate`'s ``object_meta``: through ``world_axes`` and ``world_origin``,
    which ``proj_meta`` keeps.

    GATE centres a voxelised phantom on its placement (``/gate/<phantom>/placement/setTranslation``), with its axes
    along GATE's x, y and z; :func:`~pytomography.io.load_image` reads such a file centred, with its axes as LPS
    (the vendor rule 'gate-stir-interfile-image'), in the frame ``'scanner'``. Here it is moved to where the simulation
    put it. Use the image with :func:`~pytomography.io.attenuation_map` or :meth:`~pytomography.io.Image.on`. (An image
    with its own patient geometry, such as the phantom's NIfTI, needs none of this: load it with ``load_image``.)

    Args:
        source: the image file (anything :func:`~pytomography.io.load_image` reads), or an
            :class:`~pytomography.io.Image` read from it.
        proj_meta (PETLMProjMeta): from :func:`read_gate`.
        units (str, optional): the units of the image's values when the file does not say (``'1/cm'`` for a GATE mu
            map). Defaults to None.
        placement_mm (Sequence[float], optional): where the image's centre was in GATE's world (the phantom's
            placement). Defaults to GATE's origin.

    Returns:
        Image: the image, in the patient frame when ``read_gate`` had a ``world_origin`` (else ``'scanner'``).
    """
    from pytomography.io import Image, load_image
    image = source if isinstance(source, Image) else load_image(source, units=units)
    if units is not None:
        image.units = units
    matrix = world_matrix(getattr(proj_meta, "world_axes", "LPS"))
    origin = getattr(proj_meta, "world_origin", None)
    to_patient = np.eye(4)
    to_patient[:3, :3] = matrix
    to_patient[:3, 3] = matrix @ np.asarray(placement_mm, dtype=np.float64) + (0 if origin is None else np.asarray(origin))
    centred = image.affine.copy()
    centred[:3, 3] -= image.centre                        # the image centred on GATE's origin
    return Image(image.data, to_patient @ centred, units=image.units, modality=image.modality, meta=image.meta,
                 frame="scanner" if origin is None else "patient", labels=image.labels)


# ---------- normalisation from a calibration scan ----------

def _symmetry_histogram_values(f, info: dict, tree: str = "Coincidences", include_randoms: bool = True) -> torch.Tensor:
    """The seven symmetry-class coordinates of the coincidences of an opened ROOT file (as :func:`_symmetry_values`
    gives them for crystal pairs), without the randoms unless ``include_randoms``."""
    t = f[tree]
    parts = []
    for j in (1, 2):
        rsector, module, submodule, crystal = (torch.from_numpy(_branch(t, f"{level}ID{j}").astype(np.int64))
                                               for level in ("rsector", "module", "submodule", "crystal"))
        p = []
        for ids, level in ((crystal, "crystal"), (submodule, "submodule"), (module, "module"), (rsector, "rsector")):
            p += [ids % info[f"{level}TransNr"], ids // info[f"{level}TransNr"]]
        parts.append(p)
    ids = torch.stack([detector_ids_from_parts(*p, info=info) for p in parts])
    first = ids.argmin(dim=0)   # the crystal with the smaller id first, as in torch.combinations
    a = [torch.where(first == 0, p0, p1) for p0, p1 in zip(*parts)]
    b = [torch.where(first == 0, p1, p0) for p0, p1 in zip(*parts)]
    values = _symmetry_values_of_parts(a, b, info)
    if not include_randoms:
        same = ((_branch(t, "sourcePosX1") == _branch(t, "sourcePosX2")) & (_branch(t, "sourcePosY1") == _branch(t, "sourcePosY2"))
                & (_branch(t, "sourcePosZ1") == _branch(t, "sourcePosZ2")))
        values = values[torch.from_numpy(same)]
    return values


def _symmetry_values_of_parts(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], info: dict) -> torch.Tensor:
    """Symmetry-class coordinates of crystal pairs (``a`` the crystal with the smaller id), from their part ids in the
    order of :func:`~pytomography.io.PET.scanner.crystal_parts`."""
    tc_a, ac_a, ts_a, as_a, tm_a, am_a, tr_a, ar_a = a
    tc_b, ac_b, ts_b, as_b, tm_b, am_b, tr_b, ar_b = b
    return torch.vstack([ac_a, ac_b, tc_a, tc_b,
                         (as_b - as_a) + (info["submoduleAxialNr"] - 1),
                         (am_b - am_a) + (info["moduleAxialNr"] - 1),
                         (tr_b - tr_a) % info["rsectorTransNr"]]).T


def _symmetry_values(pairs: torch.Tensor, components: Sequence[torch.Tensor], info: dict) -> torch.Tensor:
    """The seven symmetry-histogram coordinates of crystal pairs ``(a, b)``, ``a < b``: ``components`` are the
    crystals' part ids, ``crystal_parts(info, sort_by_detector_ids=True)``."""
    return _symmetry_values_of_parts([c[pairs[:, 0]] for c in components], [c[pairs[:, 1]] for c in components], info)


def _symmetry_shape(info: dict) -> list[int]:
    return [info["crystalAxialNr"], info["crystalAxialNr"], info["crystalTransNr"], info["crystalTransNr"],
            info["submoduleAxialNr"] * 2 - 1, info["moduleAxialNr"] * 2 - 1, info["rsectorTransNr"]]


def lor_radius(detector_ids: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Signed distance of each LOR from the scanner's axis, in the transaxial plane (mm): for a pair of crystals at
    the same transaxial position, the distance of that position from the axis."""
    x1, y1, _ = lut[detector_ids[:, 0]].T
    x2, y2, _ = lut[detector_ids[:, 1]].T
    return torch.where((x1 == x2) & (y1 == y2), torch.sqrt(x1**2 + y1**2),
                       (x1 * y2 - y1 * x2) / torch.sqrt((x1 - x2)**2 + (y1 - y2)**2))


def _weights_from_symmetry_histogram(histo: torch.Tensor, info: dict, cylinder_radius_mm: float,
                                     pairs_per_block: int | None = None) -> torch.Tensor:
    """Normalization weight of every crystal pair (``torch.combinations`` order) from the calibration scan's symmetry
    histogram: the counts of the pair's symmetry class over the number of pairs in it, times the correction for the
    cylinder's non-uniform exposure. NaN where the pair's LOR passes outside the cylinder.

    All pairs at once took about 75 GB for the mMR (eight 6.6 GB lists of all pairs and a 23 GB table of their symmetry
    classes), so the pairs are gone through a block at a time, twice: to count the pairs in each class, then to compute
    their weights.

    Args:
        histo (torch.Tensor): Coincidences of the calibration scan per symmetry class.
        info (dict): the scanner's ``info``.
        cylinder_radius_mm (float): Radius of cylindrical phantom used in scan
        pairs_per_block (int | None, optional): Pairs processed at once. Defaults to None: as many as fit in an eighth
            of the memory budget (about 200 bytes each), or 2**22 without a budget.

    Returns:
        torch.Tensor: Weight of every crystal pair.
    """
    lut = scanner_lut(info)
    n_crystals = lut.shape[0]
    if pairs_per_block is None:
        pairs_per_block = block_size(200, default=2**22)
    components = crystal_parts(info, sort_by_detector_ids=True)
    shape = histo.shape

    def flat_class(vals):
        index = torch.zeros(vals.shape[0], dtype=torch.long)
        for k, n in enumerate(shape):
            index = index * n + vals[:, k]
        return index
    # number of crystal pairs in each symmetry class
    n_bins = torch.zeros(histo.numel(), dtype=torch.long)
    for _, pairs in crystal_pair_blocks(n_crystals, pairs_per_block):
        n_bins += torch.bincount(flat_class(_symmetry_values(pairs, components, info)), minlength=histo.numel())
    counts_per_pair = histo / n_bins.to(histo.dtype).reshape(shape)
    weights = torch.empty(n_crystals * (n_crystals - 1) // 2, dtype=counts_per_pair.dtype)
    for offset, pairs in crystal_pair_blocks(n_crystals, pairs_per_block):
        vals = _symmetry_values(pairs, components, info)
        # the cylindrical shell exposes the LORs that cross it at a slant for longer
        correction = 1 / (torch.sqrt(1 - (torch.abs(lor_radius(pairs, lut)) / cylinder_radius_mm)**2) + pytomography.delta)
        weights[offset:offset + pairs.shape[0]] = counts_per_pair[tuple(vals.T)] * correction
    return weights


def gate_normalization(root_files: str | os.PathLike | Sequence, info: dict, cylinder_radius_mm: float, *,
                       include_randoms: bool = False) -> torch.Tensor:
    """Normalisation weights of every crystal pair from a GATE simulation of a calibration scan: a uniform cylindrical
    shell of radius ``cylinder_radius_mm`` centred on the scanner's axis.

    The coincidences are counted per symmetry class of crystal pairs (the crystals' positions within their modules, and
    their modules' and rsectors' differences), divided by the number of pairs in the class, and corrected for the
    longer path of slanted LORs through the shell. The weight of every crystal pair (411 million for the Siemens
    Biograph mMR) is computed a block of pairs at a time, sized from the memory budget
    (:func:`pytomography.set_memory_budget`).

    Pairs whose LOR passes outside the shell have no calibration data: their weight is 0, and a warning says how many
    there are.

    Args:
        root_files (str | os.PathLike | Sequence): the calibration scan's ROOT files.
        info (dict): the scanner's ``info`` (``proj_meta.info`` from :func:`read_gate`).
        cylinder_radius_mm (float): radius of the calibration shell (mm).
        include_randoms (bool, optional): count the random coincidences too. Defaults to False.

    Returns:
        torch.Tensor: float32 weight of every crystal pair, in ``torch.combinations`` order: pass it to
        :func:`read_gate` as ``normalization``.
    """
    if isinstance(root_files, (str, os.PathLike)):
        root_files = [root_files]
    shape = _symmetry_shape(info)
    histo = torch.zeros(shape)
    edges = [torch.arange(n + 1).to(torch.float32) - 0.5 for n in shape]
    for path in root_files:
        with uproot.open(path) as f:
            values = _symmetry_histogram_values(f, info, include_randoms=include_randoms)
            histo += torch.histogramdd(values.to(torch.float32), edges)[0]
    weights = _weights_from_symmetry_histogram(histo, info, cylinder_radius_mm)
    return normalization_weights(weights, info, in_place=True)
