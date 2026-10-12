"""Compiled loops (numba) of :mod:`pytomography.io.PET.petsird`: decoding the time blocks of PETSIRD 0.9 binary files, and
adding up the detection efficiencies of crystal pairs.

ETSI's ``petsird`` package turns every event into a Python object, which takes about 10 s per minute of a clinical
scan. Here the time blocks are decoded straight from the file by compiled loops, which follow yardl's binary encoding
of the PETSIRD 0.9 model:

- unsigned integers are LEB128 varints; float32 values are 4 little-endian bytes;
- a stream is blocks of [count > 0, count items], ended by a count of 0;
- a union is a one-byte case index; a vector is a varint length, then its items; a fixed vector has no length;
- an array is one varint per dimension, then its data (raw bytes for float32, varints for integers); a fixed array is
  its data only; an optional is a one-byte flag, then the value.

The cases of a time block are 0 events, 1 external signal, 2 bed movement, 3 gantry movement, 4 dead time and 5
singles histograms. Only event blocks are read; the others are stepped over. Every read is checked against the end of
the file, so a truncated or damaged file gives an error status instead of reading past the end.
"""
import numba
import numpy as np

OK = 0
BAD_TAG = 1  # a time block of a type PETSIRD 0.9 does not have
TRUNCATED = 2  # the file ends inside the time blocks
BAD_BIN = 3  # an event's detection bin is outside the scanner
MIXED_TOF = 4  # TOF bins asked for, but the module types have different TOF bins
BAD_TYPES = 5  # more module types than the scanner has


@numba.njit(cache=True, inline="always")
def _uvarint(buf, pos):
    """The unsigned varint at ``pos``: (value, next position). Past the end of ``buf``, or for a varint longer than
    64 bits, (0, len(buf) + 1)."""
    size = buf.shape[0]
    result = np.uint64(0)
    shift = np.uint64(0)
    while pos < size and shift < np.uint64(64):
        b = np.uint64(buf[pos])
        pos += 1
        result |= (b & np.uint64(0x7F)) << shift
        if b < np.uint64(0x80):
            return result, pos
        shift += np.uint64(7)
    return np.uint64(0), size + 1


@numba.njit(cache=True, inline="always")
def _count(buf, pos):
    """A length or count: (value as int64, next position). No count can be larger than the file, so a larger one
    means a damaged file: (0, len(buf) + 1)."""
    value, pos = _uvarint(buf, pos)
    if value > np.uint64(buf.shape[0]):
        return np.int64(0), buf.shape[0] + 1
    return np.int64(value), pos


@numba.njit(cache=True, inline="always")
def _skip_varints(buf, pos, n):
    """The position after ``n`` varints, or len(buf) + 1 if the file ends first."""
    size = buf.shape[0]
    for _ in range(n):
        while pos < size and buf[pos] >= 0x80:
            pos += 1
        pos += 1
        if pos > size:
            return size + 1
    return pos


@numba.njit(cache=True)
def walk_time_blocks(buf, pos, t0, t1, wanted, max_events, delayeds, fill, offsets, n_energy, n_tof, read_tof,
                     read_energy, id_map, out):
    """One pass over the time blocks that start at byte ``pos`` of ``buf`` (the file, as uint8).

    Counts (``fill`` False) or writes into ``out`` (``fill`` True) the prompt coincidences, or the delayed ones if
    ``delayeds``, of the event blocks that start in [t0, t1) ms. If ``wanted`` is not empty, only event blocks whose
    index among the event blocks (from 0) has ``wanted[index] != 0`` are read. Reading stops after the block in which
    ``max_events`` events are reached, if ``max_events >= 0``.

    The columns of ``out`` are the detector ids of the two elements, ``id_map[element]``, where an element is numbered
    ``offsets[type] + detection bin // n_energy[type]``; then the TOF bin if ``read_tof``; then the energy bins of the
    two elements (``detection bin % n_energy[type]``) if ``read_energy``.

    Returns (number of events, status), where the status is one of the constants above.
    """
    size = buf.shape[0]
    n_types = n_energy.shape[0]
    n = 0
    block = -1
    while True:
        count, pos = _count(buf, pos)
        if pos > size:
            return n, TRUNCATED
        if count == 0:
            return n, OK
        for _ in range(count):
            if pos >= size:
                return n, TRUNCATED
            tag = buf[pos]
            pos += 1
            start, pos = _uvarint(buf, pos)
            _stop, pos = _uvarint(buf, pos)
            if tag == 0:
                block += 1
                if start >= t1:
                    return n, OK
                keep = start >= t0
                if wanted.shape[0] > 0:
                    if block >= wanted.shape[0]:
                        return n, OK  # no wanted block is left
                    keep = keep and wanted[block] != 0
                # single events: vector<vector<SingleEvent{detection bin, time offset}>>
                n_a, pos = _count(buf, pos)
                for _a in range(n_a):
                    n_e, pos = _count(buf, pos)
                    pos = _skip_varints(buf, pos, 2 * n_e)
                if pos > size:
                    return n, TRUNCATED
                # prompts, then delayeds: vector<vector<vector<CoincidenceEvent{bins[2], tof}>>> by module types
                for kind in range(2):
                    take = keep and ((kind == 1) == delayeds)
                    n_a, pos = _count(buf, pos)
                    if n_a > n_types:
                        return n, BAD_TYPES
                    for a in range(n_a):
                        n_b, pos = _count(buf, pos)
                        if n_b > n_types:
                            return n, BAD_TYPES
                        for b in range(n_b):
                            n_e, pos = _count(buf, pos)
                            if not take or n_e == 0:
                                pos = _skip_varints(buf, pos, 3 * n_e)
                                if pos > size:
                                    return n, TRUNCATED
                                continue
                            if read_tof and n_tof[a, b] != n_tof[0, 0]:
                                return n, MIXED_TOF
                            elements_a = offsets[a + 1] - offsets[a]
                            elements_b = offsets[b + 1] - offsets[b]
                            for _e in range(n_e):
                                bin_a, pos = _uvarint(buf, pos)
                                bin_b, pos = _uvarint(buf, pos)
                                tof, pos = _uvarint(buf, pos)
                                if pos > size:
                                    return n, TRUNCATED
                                element_a = np.int64(bin_a // np.uint64(n_energy[a]))
                                element_b = np.int64(bin_b // np.uint64(n_energy[b]))
                                if not (0 <= element_a < elements_a and 0 <= element_b < elements_b):
                                    return n, BAD_BIN
                                if fill:
                                    out[n, 0] = id_map[offsets[a] + element_a]
                                    out[n, 1] = id_map[offsets[b] + element_b]
                                    column = 2
                                    if read_tof:
                                        out[n, 2] = np.int64(tof)
                                        column = 3
                                    if read_energy:
                                        out[n, column] = np.int64(bin_a % np.uint64(n_energy[a]))
                                        out[n, column + 1] = np.int64(bin_b % np.uint64(n_energy[b]))
                                n += 1
                # triple events: vector^4 of TripleEvent{bins[3], tof indices[2]}
                n_1, pos = _count(buf, pos)
                for _1 in range(n_1):
                    n_2, pos = _count(buf, pos)
                    for _2 in range(n_2):
                        n_3, pos = _count(buf, pos)
                        for _3 in range(n_3):
                            n_4, pos = _count(buf, pos)
                            pos = _skip_varints(buf, pos, 5 * n_4)
                        if pos > size:
                            return n, TRUNCATED
                # quadruple events: vector^5 of TripleEvent (so in the 0.9 model)
                n_1, pos = _count(buf, pos)
                for _1 in range(n_1):
                    n_2, pos = _count(buf, pos)
                    for _2 in range(n_2):
                        n_3, pos = _count(buf, pos)
                        for _3 in range(n_3):
                            n_4, pos = _count(buf, pos)
                            for _4 in range(n_4):
                                n_5, pos = _count(buf, pos)
                                pos = _skip_varints(buf, pos, 5 * n_5)
                            if pos > size:
                                return n, TRUNCATED
                if pos > size:
                    return n, TRUNCATED
                if keep and max_events >= 0 and n >= max_events:
                    return n, OK
            elif tag == 1:  # external signal: signal id, vector<float32>
                pos = _skip_varints(buf, pos, 1)
                n_v, pos = _count(buf, pos)
                pos += 4 * n_v
            elif tag == 2 or tag == 3:  # bed or gantry movement: a fixed 3 x 4 float32 matrix
                pos += 48
            elif tag == 4:  # dead time: vector<NDArray<float32, 1>>, optional<vector<vector<NDArray<float32, 2>>>>
                n_v, pos = _count(buf, pos)
                for _v in range(n_v):
                    d, pos = _count(buf, pos)
                    pos += 4 * d
                    if pos > size:
                        return n, TRUNCATED
                if pos >= size:
                    return n, TRUNCATED
                has_value = buf[pos]
                pos += 1
                if has_value != 0:
                    n_1, pos = _count(buf, pos)
                    for _1 in range(n_1):
                        n_2, pos = _count(buf, pos)
                        for _2 in range(n_2):
                            d_1, pos = _count(buf, pos)
                            d_2, pos = _count(buf, pos)
                            if d_2 > 0 and d_1 > size // d_2:
                                return n, TRUNCATED
                            pos += 4 * d_1 * d_2
                            if pos > size:
                                return n, TRUNCATED
            elif tag == 5:  # singles histograms: vector<NDArray<uint64, 1>>
                n_v, pos = _count(buf, pos)
                for _v in range(n_v):
                    d, pos = _count(buf, pos)
                    pos = _skip_varints(buf, pos, d)
                    if pos > size:
                        return n, TRUNCATED
            else:
                return n, BAD_TAG
            if pos > size:
                return n, TRUNCATED


@numba.njit(cache=True)
def add_module_pairs(total, orders, modules_a, modules_b, tables, efficiencies_a, efficiencies_b, offset_a, offset_b,
                     id_map, n, calibration):
    """Adds the detection efficiency of every pair of elements of the module pairs (``modules_a[k]``,
    ``modules_b[k]``) to ``total``, at the pair's index in the order of ``torch.combinations(torch.arange(n), 2)``, and
    counts it in ``orders``.

    The efficiency of elements i and j is ``calibration`` times the sum over energy bins e, f of
    ``efficiencies_a[module a, i, e] * efficiencies_b[module b, j, f] * tables[k, i, e, j, f]``; with an empty
    ``tables``, the module pair's efficiencies are 1. Element i of module m of type a is
    ``id_map[offset_a + m * elements + i]``. Pairs of an element with itself are left out.
    """
    elements_a, energies_a = efficiencies_a.shape[1], efficiencies_a.shape[2]
    elements_b, energies_b = efficiencies_b.shape[1], efficiencies_b.shape[2]
    use_tables = tables.shape[0] > 0
    for k in range(modules_a.shape[0]):
        m_a, m_b = modules_a[k], modules_b[k]
        for i in range(elements_a):
            id_a = id_map[offset_a + m_a * elements_a + i]
            for j in range(elements_b):
                id_b = id_map[offset_b + m_b * elements_b + j]
                if id_a == id_b:
                    continue
                value = 0.0
                for e in range(energies_a):
                    for f in range(energies_b):
                        product = efficiencies_a[m_a, i, e] * efficiencies_b[m_b, j, f]
                        if use_tables:
                            product = product * tables[k, i, e, j, f]
                        value += product
                low, high = min(id_a, id_b), max(id_a, id_b)
                index = low * (2 * n - low - 1) // 2 + (high - low - 1)
                total[index] += calibration * value
                orders[index] += 1
