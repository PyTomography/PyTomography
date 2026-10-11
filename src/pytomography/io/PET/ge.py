"""Read GE PET list mode (the HDF5 RDF files of the Discovery MI) and GE's corrections, in PyTomography's standard
form::

    events, object_meta, proj_meta = io.PET.read_ge_hdf5('LIST0000.BLF')
    event_weights, additive_counts, pair_ids, pair_weights = io.PET.read_ge_corrections('corrections.h5')

The scanner's geometry comes from PyTomography's table of clinical scanners (``data/pet_scanner_info.txt``), which
agrees with the file's ``SystemGeometry``; its TOF bins too (GE's file does not give the TOF resolution). The energy
window comes from the file.

**The frame (GE's vendor rule, "check").** A detector id is ``544 ring + crystal``. Crystal 0 is at the top of the
gantry, and the rings and crystals are placed so that the bed lies posterior (+y): the object frame is turned 180
degrees about the axis from the geometry PyTomography used before version 4, in which images were upside down. That
the bed is posterior is verified (GE's attenuation factors back-projected); left-right and head-foot are not (there is
no GE reconstruction or CT of the local data), so the reader warns.
"""
from __future__ import annotations

import os
import warnings

import h5py
import numpy as np
import torch

from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta
from .scanner import _table_row, keep_rsector_difference, scanner_info
from ._standard import default_grid, reader_object_meta

#: rows of events read at once (the file's chunks hold 8 million)
_BLOCK = 2**23


def _vendor_rule(rule_id: str, source: str = ""):
    """Looks a rule up in the vendor table (which warns when the rule is not verified)."""
    try:
        from pytomography.io import _vendors
    except ImportError:   # the vendor table is not in this build yet
        return None
    return _vendors.use(rule_id, source)


def _tof_meta(name: str) -> PETTOFMeta:
    """The TOF bins of a scanner of PyTomography's table."""
    row = _table_row(name)
    return PETTOFMeta(num_bins=int(float(row["num_tof_bins"])), tof_range=float(row["tof_range"]),
                      fwhm=float(row["tof_fwhm"]))


def _header_value(f: h5py.File, *paths: str):
    """The first of these header entries that the file has, as a Python number; None if it has none."""
    for path in paths:
        if path in f:
            value = np.asarray(f[path][()]).ravel()
            if value.size:
                return value[0].item()
    return None


def _check_geometry(f: h5py.File, info: dict, path: str) -> None:
    """Raises if the file's SystemGeometry is not the scanner of ``info`` (crystals around a ring, and rings)."""
    geometry = "HeaderData/SystemGeometry/"
    around = [_header_value(f, geometry + k) for k in ("radialModulesPerSystem", "radialBlocksPerModule",
                                                       "radialCrystalsPerBlock")]
    along = [_header_value(f, geometry + k) for k in ("axialModulesPerSystem", "axialBlocksPerModule",
                                                      "axialCrystalsPerBlock")]
    if None in around or None in along:
        return   # no SystemGeometry: nothing to check against
    crystals, rings = int(np.prod(around)), int(np.prod(along))
    if (crystals, rings) != (info["NrCrystalsPerRing"], info["NrRings"]):
        raise ValueError(f"{path} is from a scanner with {crystals} crystals around {rings} rings, not the "
                         f"{info['NrCrystalsPerRing']} x {info['NrRings']} of PyTomography's table entry")


def read_ge_hdf5(listmode_file: str | os.PathLike, *, object_meta: ObjectMeta | None = None,
                 scanner: str = "discovery_MI", min_rsector_difference: int = 0
                 ) -> tuple[torch.Tensor, ObjectMeta, PETLMProjMeta]:
    """The coincidences of a GE HDF5 list mode file (RDF, e.g. ``LIST0000.BLF``) in PyTomography's standard form.

    Each event of ``MiceList/TofCoinc`` is (ring, crystal, ring, crystal, TOF): its detector ids are ``ring * 544 +
    crystal`` and its TOF bin ``14 - raw / 13`` (the raw values are multiples of 13 fine units of 13.02 ps), so that
    TOF bins increase from column 0's crystal to column 1's. Events outside the TOF bins are dropped. The events are
    read a block at a time (8 million rows).

    Args:
        listmode_file (str | os.PathLike): the list mode file.
        object_meta (ObjectMeta, optional): the reconstruction grid (``dr`` and ``shape``); its centre and frame are
            set by the reader. Defaults to None: voxels half the crystal pitch, a square covering 80 % of the ring's
            diameter, and every ring.
        scanner (str, optional): the scanner, by its name in PyTomography's table. Defaults to ``'discovery_MI'``.
        min_rsector_difference (int, optional): keep only events whose crystals are at least this many modules apart
            around the ring. Defaults to 0.

    Returns:
        tuple[torch.Tensor, ObjectMeta, PETLMProjMeta]: the events (int32 ``[N, 3]``), the object grid (modality
        ``'PT'``, units ``'counts'``, frame ``'scanner'``) and the projection metadata (``info``, with the energy
        window from the file, and ``tof_meta``). GE's corrections (event weights, the additive term, the valid pairs
        and their weights) are read by :func:`read_ge_corrections`.
    """
    _vendor_rule("ge-dmi-listmode", str(listmode_file))
    with h5py.File(listmode_file, "r") as f:
        low = _header_value(f, "HeaderData/AcqParameters/EDCATParameters/lower_energy_limit",
                            "HeaderData/AcqParameters/BackEndAcqFilters/minEnergyKeV")
        high = _header_value(f, "HeaderData/AcqParameters/EDCATParameters/upper_energy_limit",
                             "HeaderData/AcqParameters/BackEndAcqFilters/maxEnergyKeV")
        info = scanner_info(scanner, min_rsector_difference=min_rsector_difference,
                            energy_window_low_keV=None if low is None else float(low),
                            energy_window_high_keV=None if high is None else float(high))
        _check_geometry(f, info, str(listmode_file))
        duration = _header_value(f, "HeaderData/AcqParameters/RxScanParameters/acqTime")
        tof_meta = _tof_meta(scanner)
        coincidences = f["MiceList/TofCoinc"]
        blocks = []
        crystals = info["NrCrystalsPerRing"]
        for start in range(0, coincidences.shape[0], _BLOCK):
            raw = torch.from_numpy(coincidences[start:start + _BLOCK].astype(np.int32))
            tof = (tof_meta.num_bins // 2) - torch.div(raw[:, 4], 13, rounding_mode="floor")
            events = torch.stack([crystals * raw[:, 0] + raw[:, 1], crystals * raw[:, 2] + raw[:, 3], tof], dim=1)
            blocks.append(events[(tof >= 0) & (tof < tof_meta.num_bins)])
            del raw, tof, events
    events = keep_rsector_difference(torch.cat(blocks).to(torch.int32) if blocks else
                                     torch.zeros((0, 3), dtype=torch.int32), info)
    object_meta = reader_object_meta(object_meta, default_grid(info), centre=(0.0, 0.0, 0.0), frame="scanner",
                                     modality="PT", units="counts")
    proj_meta = PETLMProjMeta(events, info, tof_meta=tof_meta)
    proj_meta.duration_s = None if duration is None else float(duration)
    return events, object_meta, proj_meta


def read_ge_corrections(corrections_file: str | os.PathLike, *, pairs: bool = True,
                        crystals_per_ring: int = 544) -> tuple:
    """GE's corrections for a list mode file (``corrections.h5``), in PyTomography's standard form.

    * ``event_weights``: the multiplicative factor of each event, attenuation x sensitivity
      (``correction_lists/atten * sens``): the ``weights`` of :class:`~pytomography.metadata.PET.PETLMProjMeta`. They
      include attenuation, so no attenuation map is passed with them.
    * ``additive_counts``: the expected randoms and scatter of each event, in counts (``correction_lists/contam``).
      The likelihood's additive term is these divided by the event's sensitivity.
    * ``pair_ids``: every pair of crystals GE counts as valid, int32 ``[M, 2]`` detector ids (``all_xtals/xtal_ids``
      stores (crystal, ring, crystal, ring), the other order from the list mode): ``detector_ids_sensitivity``.
    * ``pair_weights``: their attenuation x sensitivity: ``weights_sensitivity``.

    The 146 million valid pairs of the Discovery MI are read a block of 8 million at a time (2 GB in all for the ids and
    weights).

    Args:
        corrections_file (str | os.PathLike): GE's corrections file.
        pairs (bool, optional): read the valid pairs and their weights too. Defaults to True.
        crystals_per_ring (int, optional): crystals in a ring of the scanner. Defaults to 544 (the Discovery MI).

    Returns:
        tuple: ``event_weights``, ``additive_counts`` (float32 ``[N]``), ``pair_ids`` (int32 ``[M, 2]``) and
        ``pair_weights`` (float32 ``[M]``); the last two are None without ``pairs``.
    """
    with h5py.File(corrections_file, "r") as f:
        lists = f["correction_lists"]
        event_weights = torch.from_numpy(lists["atten"][:].astype(np.float32))
        event_weights.mul_(torch.from_numpy(lists["sens"][:].astype(np.float32)))
        additive_counts = torch.from_numpy(lists["contam"][:].astype(np.float32))
        if not pairs:
            return event_weights, additive_counts, None, None
        ids = f["all_xtals/xtal_ids"]
        n = ids.shape[0]
        pair_ids = torch.empty((n, 2), dtype=torch.int32)
        pair_weights = torch.empty(n, dtype=torch.float32)
        for start in range(0, n, _BLOCK):
            stop = min(start + _BLOCK, n)
            block = torch.from_numpy(ids[start:stop].astype(np.int32))
            pair_ids[start:stop, 0] = crystals_per_ring * block[:, 1] + block[:, 0]
            pair_ids[start:stop, 1] = crystals_per_ring * block[:, 3] + block[:, 2]
            pair_weights[start:stop] = torch.from_numpy(f["all_xtals/atten"][start:stop] * f["all_xtals/sens"][start:stop])
    return event_weights, additive_counts, pair_ids, pair_weights
