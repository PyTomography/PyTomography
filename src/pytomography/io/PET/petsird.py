"""Read list-mode data in PETSIRD, the format of the Emission Tomography Standardization Initiative (ETSI).

Files are read with ETSI's own Python package, ``petsird``, version 0.9: the format that ETSI's converters
(STIR2PETSIRD, GATE to PETSIRD, CASToR) and STIR write. Install it with ``pip install "pytomography[petsird]"``.

The detecting elements of every type of module are numbered one after the other: those of module type 0 first, then
those of type 1, and so on. Within a type, element ``e`` of module ``m`` is number ``m * elements_per_module + e``.
The scanner lookup table, the detector ids of events and the sensitivity weights all use this numbering.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

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
    return getattr(header_or_scanner, "scanner", header_or_scanner)


def read_header(petsird_file: str):
    """The header of a PETSIRD file: the scanner (``header.scanner``) and the exam."""
    petsird = _petsird()
    with petsird.BinaryPETSIRDReader(str(petsird_file), skip_completed_check=True) as reader:
        return reader.read_header()


def get_element_offsets(scanner) -> list[int]:
    """The number of the first detecting element of each type of module, then the total number of elements."""
    offsets = [0]
    for rep in _scanner(scanner).scanner_geometry.replicated_modules:
        offsets.append(offsets[-1] + len(rep.transforms) * len(rep.object.detecting_elements.transforms))
    return offsets


def get_scanner_LUT(scanner) -> torch.Tensor:
    """Position of every detecting element, the centre of its box, in mm in the gantry coordinate system.

    Args:
        scanner: a PETSIRD ``ScannerInformation``, or a header.

    Returns:
        torch.Tensor: ``N x 3`` positions, numbered as described at the top of this module.
    """
    positions = []
    for rep in _scanner(scanner).scanner_geometry.replicated_modules:
        elements = rep.object.detecting_elements
        centre = np.mean([corner.c for corner in elements.object.shape.corners], axis=0)
        element_mats = np.stack([t.matrix for t in elements.transforms]).astype(np.float64)  # E x 3 x 4
        module_mats = np.stack([t.matrix for t in rep.transforms]).astype(np.float64)  # M x 3 x 4
        in_module = element_mats[:, :, :3] @ centre + element_mats[:, :, 3]  # E x 3
        in_gantry = np.einsum("mij,ej->mei", module_mats[:, :, :3], in_module) + module_mats[:, None, :, 3]
        positions.append(in_gantry.reshape(-1, 3))
    return torch.tensor(np.concatenate(positions), dtype=torch.float32)


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


def get_detector_ids(
    petsird_file: str,
    read_tof: bool | None = None,
    read_energy: bool = False,
    time_block_ids: Sequence[int] | None = None,
    return_header: bool = False,
    delayeds: bool = False,
    time_interval: tuple[float, float] | None = None,
    max_events: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, object]:
    """Read the coincidences of a PETSIRD file as detector ids.

    The file is read one time block at a time, and only the ids are kept: 16 to 40 bytes per event.

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

    Returns:
        torch.Tensor: ``N x 2`` detector ids, ``N x 3`` with TOF bins, plus two columns with energy bins; int64. With
        ``return_header``, also the header.
    """
    petsird = _petsird()
    with petsird.BinaryPETSIRDReader(str(petsird_file), skip_completed_check=True) as reader:
        header = reader.read_header()
        scanner = header.scanner
        n_types = scanner.scanner_geometry.number_of_module_types()
        if delayeds and not scanner.delayed_events_are_stored:
            raise ValueError("this PETSIRD file stores no delayed coincidences")
        offsets = get_element_offsets(scanner)
        n_energy = [scanner.event_energy_bin_edges[t].number_of_bins() for t in range(n_types)]
        n_tof = {(a, b): scanner.tof_bin_edges[a][b].number_of_bins() for a in range(n_types) for b in range(n_types)}
        if read_tof is None:
            read_tof = max(n_tof.values()) > 1
        chunks, count, block = [], 0, -1
        for time_block in reader.read_time_blocks():
            if not isinstance(time_block, petsird.TimeBlock.EventTimeBlock):
                continue
            block += 1
            start = time_block.value.time_interval.start
            if time_block_ids is not None and block not in time_block_ids:
                continue
            if time_interval is not None:
                if start >= time_interval[1]:
                    break
                if start < time_interval[0]:
                    continue
            events = time_block.value.delayed_events if delayeds else time_block.value.prompt_events
            for a in range(n_types):
                for b in range(n_types):
                    pair = events[a][b]
                    if not pair:
                        continue
                    if read_tof and n_tof[a, b] != n_tof[0, 0]:
                        raise ValueError("the module types of this scanner have different TOF bins; read the events"
                                         " without TOF (read_tof=False)")
                    bins = np.array([e.detection_bins for e in pair], dtype=np.int64)
                    columns = [offsets[a] + bins[:, 0] // n_energy[a], offsets[b] + bins[:, 1] // n_energy[b]]
                    if read_tof:
                        columns.append(np.fromiter((e.tof_idx for e in pair), dtype=np.int64, count=len(pair)))
                    if read_energy:
                        columns += [bins[:, 0] % n_energy[a], bins[:, 1] % n_energy[b]]
                    chunks.append(np.stack(columns, axis=1))
                    count += len(pair)
            if max_events is not None and count >= max_events:
                break
    width = 2 + bool(read_tof) + 2 * bool(read_energy)
    detector_ids = torch.from_numpy(np.concatenate(chunks) if chunks else np.zeros((0, width), dtype=np.int64))
    return (detector_ids, header) if return_header else detector_ids


def _pair_index(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Index of the pair (a, b), a < b, in the order of torch.combinations(torch.arange(n), 2)."""
    return a * (2 * n - a - 1) // 2 + (b - a - 1)


def get_sensitivity_weights(scanner, energy_bins: Sequence[int] | None = None) -> torch.Tensor:
    """Detection efficiency of every pair of detecting elements, for the sensitivity image (``weights_sensitivity``).

    As in PETSIRD, the efficiency of a coincidence between two detection bins is the calibration factor, times the
    efficiencies of the two bins, times the efficiency of their module pair, which is 0 for module pairs that are not
    in coincidence. For a pair of elements this is summed over the energy bins, and the two orders of the elements
    are averaged when the file gives both.

    Args:
        scanner: a PETSIRD ``ScannerInformation``, or a header.
        energy_bins (Sequence[int] | None, optional): count only these energy bins; all of them if None.

    Returns:
        torch.Tensor: one float32 weight per pair of elements, in the order of
        ``torch.combinations(torch.arange(N), 2)``, which :class:`~pytomography.metadata.PET.PETLMProjMeta` expects.
        4 bytes per pair: 1.6 GB for a scanner of 28,672 elements.
    """
    scanner = _scanner(scanner)
    eff = scanner.detection_efficiencies
    geometry = scanner.scanner_geometry.replicated_modules
    offsets = get_element_offsets(scanner)
    n = offsets[-1]
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
    orders = np.zeros(n * (n - 1) // 2, dtype=np.uint8)
    for a, (ma, ea, na) in enumerate(shape):
        for b, (mb, eb, nb) in enumerate(shape):
            if eff is None or eff.module_pair_sgidlut is None:
                sgid, matrices = np.where(np.eye(ma, mb, dtype=bool) & (a == b), -1, 0), None
            else:
                sgid, matrices = np.asarray(eff.module_pair_sgidlut[a][b]), eff.module_pair_efficiencies_vectors[a][b]
            for m0, m1 in zip(*np.nonzero(sgid >= 0)):
                d0, d1 = bin_eff[a][m0], bin_eff[b][m1]  # elements x energy bins
                if matrices is None:
                    block = np.einsum("ie,jf->ij", d0, d1)
                else:
                    values = np.asarray(matrices[sgid[m0, m1]].values, dtype=np.float64).reshape(ea, na, eb, nb)
                    block = np.einsum("ie,jf,iejf->ij", d0, d1, values)
                g0 = offsets[a] + m0 * ea + np.arange(ea)[:, None]
                g1 = offsets[b] + m1 * eb + np.arange(eb)[None, :]
                lo, hi = np.minimum(g0, g1), np.maximum(g0, g1)
                valid = lo != hi
                index = _pair_index(lo[valid], hi[valid], n)
                if a == b and m0 == m1:  # (i, j) and (j, i) of one module are the same pair
                    np.add.at(total, index, calibration * block[valid])
                    np.add.at(orders, index, 1)
                else:
                    total[index] += calibration * block[valid]
                    orders[index] += 1
    np.divide(total, orders, out=total, where=orders > 0)
    return torch.from_numpy(total)
