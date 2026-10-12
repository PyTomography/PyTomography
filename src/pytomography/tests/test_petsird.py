"""Tests of pytomography.io.PET.petsird on small PETSIRD files written here with ETSI's petsird package (CPU only).

The scanners come from petsird's own example generator, with random efficiencies, and every result is compared with
petsird's helpers (geometry, detection bins, detection efficiency) or with the events as ETSI's own reader reads them."""
import math
import os

import numpy as np
import pytest
import torch

petsird = pytest.importorskip("petsird")
from petsird.helpers import (generator, geometry, get_detection_efficiency, get_num_detection_bins,  # noqa: E402
                             make_detection_bin)

from pytomography.io.PET import petsird as reader  # noqa: E402
from pytomography.io.PET import shared  # noqa: E402


def module_def(**kwargs):
    """A small cylindrical module type: 6 modules of 1 x 2 x 2 crystals, 2 energy bins, 5 TOF bins."""
    values = dict(radius=100, crystal_length=(10, 4, 4), num_crystals_per_module=(1, 2, 2), num_modules_along_ring=6,
                  num_modules_along_axis=1, arc=2 * math.pi, start_angle=0, module_spacing_along_axis=0.,
                  number_of_tof_bins=5, tof_resolution=60., LLD=430, ULD=650, number_of_event_energy_bins=2,
                  energy_resolution=0.1, material_id=1)
    values.update(kwargs)
    return generator.CylindricalBlocksInfo(**values)


def scanner_with_random_efficiencies(defs, seed=0):
    rng = np.random.default_rng(seed)
    scanner = generator.get_scanner_info(defs)
    eff = scanner.detection_efficiencies
    eff.detection_bin_efficiencies = [rng.uniform(0.5, 1.5, size=np.shape(d)).astype(np.float32)
                                      for d in eff.detection_bin_efficiencies]
    for row in eff.module_pair_efficiencies_vectors:
        for vector in row:
            for matrix in vector:
                matrix.values = rng.uniform(0.5, 1.5, size=np.shape(matrix.values)).astype(np.float32)
    eff.calibration_factor = 2.0
    return scanner


def expanded(module, element, energy):
    return petsird.ExpandedDetectionBin(module_index=module, element_index=element, energy_index=energy)


def write(path, scanner, blocks):
    """blocks: [(start ms, stop ms, {(type0, type1): [(bin0, bin1, tof), ...]})]"""
    n_types = scanner.scanner_geometry.number_of_module_types()
    with petsird.BinaryPETSIRDWriter(str(path)) as writer:
        writer.write_header(petsird.Header(scanner=scanner))
        time_blocks = []
        for start, stop, events in blocks:
            prompts = [[[petsird.CoincidenceEvent(detection_bins=[b0, b1], tof_idx=tof)
                         for b0, b1, tof in events.get((a, b), [])] for b in range(n_types)] for a in range(n_types)]
            time_blocks.append(petsird.TimeBlock.EventTimeBlock(petsird.EventTimeBlock(
                time_interval=petsird.TimeInterval(start=start, stop=stop), prompt_events=prompts)))
        writer.write_time_blocks(time_blocks)


@pytest.fixture
def one_type():
    return scanner_with_random_efficiencies([module_def()])


def test_scanner_LUT_is_the_centre_of_each_detecting_box(one_type):
    lut = reader.get_scanner_LUT(one_type)
    rep = one_type.scanner_geometry.replicated_modules[0]
    n_elements = len(rep.object.detecting_elements.transforms)
    assert lut.shape == (len(rep.transforms) * n_elements, 3)
    for module in range(len(rep.transforms)):
        for element in range(n_elements):
            box = geometry.get_detecting_box(one_type, 0, expanded(module, element, 0))
            centre = np.mean([corner.c for corner in box.corners], axis=0)
            assert np.allclose(lut[module * n_elements + element].numpy(), centre, atol=1e-4)
    assert np.allclose(lut.norm(dim=1)[:4].numpy().min(), 105, atol=3)  # crystals 10 mm deep at radius 100


def test_detector_ids_TOF_and_energy_bins(one_type, tmp_path):
    bin_ = lambda m, e, en: make_detection_bin(one_type, 0, expanded(m, e, en))
    events = [(bin_(0, 1, 0), bin_(3, 2, 1), 4), (bin_(5, 3, 1), bin_(1, 0, 0), 0), (bin_(2, 0, 1), bin_(4, 1, 1), 2)]
    write(tmp_path / "f.petsird", one_type, [(0, 1000, {(0, 0): events})])
    ids, header = reader.get_detector_ids(tmp_path / "f.petsird", return_header=True)
    assert header.scanner.model_name == one_type.model_name
    assert ids.dtype == torch.int64
    assert ids.tolist() == [[1, 14, 4], [23, 4, 0], [8, 17, 2]]  # element = module * 4 + element; TOF bin
    with_energy = reader.get_detector_ids(tmp_path / "f.petsird", read_tof=False, read_energy=True)
    assert with_energy.tolist() == [[1, 14, 0, 1], [23, 4, 1, 0], [8, 17, 1, 1]]


def test_time_blocks_time_interval_and_early_stop(one_type, tmp_path):
    bin_ = lambda m: make_detection_bin(one_type, 0, expanded(m, 0, 0))
    blocks = [(t * 100, (t + 1) * 100, {(0, 0): [(bin_(t % 6), bin_((t + 3) % 6), 0)] * (t + 1)}) for t in range(5)]
    path = tmp_path / "f.petsird"
    write(path, one_type, blocks)
    assert len(reader.get_detector_ids(path)) == 1 + 2 + 3 + 4 + 5
    assert len(reader.get_detector_ids(path, time_block_ids=[0, 4])) == 1 + 5
    assert len(reader.get_detector_ids(path, time_interval=(100, 300))) == 2 + 3  # blocks starting at 100 and 200
    assert len(reader.get_detector_ids(path, max_events=4)) == 1 + 2 + 3  # stops reading after the third block


def test_two_module_types(tmp_path):
    scanner = scanner_with_random_efficiencies([module_def(), module_def(radius=150, num_modules_along_ring=4,
                                                                         number_of_event_energy_bins=1,
                                                                         number_of_tof_bins=3)])
    offsets = reader.get_element_offsets(scanner)
    assert offsets == [0, 24, 40]
    assert reader.get_scanner_LUT(scanner).shape == (40, 3)
    b0 = make_detection_bin(scanner, 0, expanded(2, 1, 1))
    b1 = make_detection_bin(scanner, 1, expanded(3, 2, 0))
    write(tmp_path / "f.petsird", scanner, [(0, 10, {(0, 1): [(b0, b1, 0)]})])
    assert reader.get_detector_ids(tmp_path / "f.petsird", read_tof=False).tolist() == [[9, 24 + 3 * 4 + 2]]
    with pytest.raises(ValueError, match="different TOF bins"):
        reader.get_detector_ids(tmp_path / "f.petsird", read_tof=True)


def test_TOF_metadata(one_type):
    meta = reader.get_TOF_meta(one_type)
    edges = one_type.tof_bin_edges[0][0].edges
    assert meta.num_bins == 5 and math.isclose(float(meta.bin_width), (edges[-1] - edges[0]) / 5, rel_tol=1e-5)
    assert math.isclose(float(meta.sigma) * 2.355, 60.0, rel_tol=1e-4)
    one_type.tof_bin_edges[0][0] = petsird.BinEdges(edges=np.array([-100, -40, 30, 100], dtype=np.float32))
    with pytest.raises(ValueError, match="symmetric about 0"):
        reader.get_TOF_meta(one_type)


def brute_force_weights(scanner, energy_bins=None):
    """Sensitivity weights from petsird's own get_detection_efficiency, one pair at a time."""
    offsets = reader.get_element_offsets(scanner)
    geometry_ = scanner.scanner_geometry.replicated_modules
    elements = []  # (type, module, element) of every element, in PyTomography's numbering
    for t, rep in enumerate(geometry_):
        n_el = len(rep.object.detecting_elements.transforms)
        elements += [(t, m, e) for m in range(len(rep.transforms)) for e in range(n_el)]
    luts = scanner.detection_efficiencies.module_pair_sgidlut
    n = offsets[-1]
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            values = []
            for (ta, ma, ea), (tb, mb, eb) in [(elements[i], elements[j]), (elements[j], elements[i])]:
                if luts[ta][tb][ma, mb] < 0:
                    continue
                pair = petsird.TypeOfModulePair((ta, tb))
                total = 0.0
                for na in range(scanner.event_energy_bin_edges[ta].number_of_bins()):
                    for nb in range(scanner.event_energy_bin_edges[tb].number_of_bins()):
                        if energy_bins is not None and (na not in energy_bins or nb not in energy_bins):
                            continue
                        total += get_detection_efficiency(scanner, pair,
                                                          make_detection_bin(scanner, ta, expanded(ma, ea, na)),
                                                          make_detection_bin(scanner, tb, expanded(mb, eb, nb)))
                values.append(total)
            weights.append(np.mean(values) if values else 0.0)
    return np.array(weights)


@pytest.mark.parametrize("energy_bins", [None, [1]])
def test_sensitivity_weights_match_petsird(one_type, energy_bins):
    weights = reader.get_sensitivity_weights(one_type, energy_bins=energy_bins)
    assert weights.dtype == torch.float32 and len(weights) == 24 * 23 // 2
    assert np.allclose(weights.numpy(), brute_force_weights(one_type, energy_bins), rtol=1e-5)


def test_sensitivity_weights_two_module_types():
    scanner = scanner_with_random_efficiencies([module_def(), module_def(radius=150, num_modules_along_ring=4,
                                                                         number_of_event_energy_bins=1)], seed=3)
    assert np.allclose(reader.get_sensitivity_weights(scanner).numpy(), brute_force_weights(scanner), rtol=1e-5)


def test_pair_order_is_torch_combinations():
    n = 7
    pairs = torch.combinations(torch.arange(n), 2).numpy()
    assert (reader._pair_index(pairs[:, 0], pairs[:, 1], n) == np.arange(len(pairs))).all()


def test_wrong_petsird_version(monkeypatch):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.11.1")
    with pytest.raises(ImportError, match="petsird 0.9"):
        reader._petsird()


# --- the compiled reader against ETSI's own reader ---

def two_types():
    """Module type 0 has 144 elements x 2 energy bins, so its detection bins take two-byte varints."""
    scanner = scanner_with_random_efficiencies([
        module_def(num_crystals_per_module=(1, 4, 3), num_modules_along_ring=12),
        module_def(radius=150, num_modules_along_ring=4, number_of_event_energy_bins=1, number_of_tof_bins=3)])
    scanner.delayed_events_are_stored = True
    return scanner


def write_every_kind(path, scanner, seed=1, batches="one list"):
    """A file with every kind of time block. Event blocks hold singles, prompts, delayeds, triples and quadruples, and
    start at times that take varints of one to five bytes. ``batches``: how the stream of time blocks is written, as
    one list (one count), one block at a time (a count of 1 each), or two lists."""
    rng = np.random.default_rng(seed)
    n_types = scanner.scanner_geometry.number_of_module_types()
    n_bins = [get_num_detection_bins(scanner, t) for t in range(n_types)]
    n_modules = [len(r.transforms) for r in scanner.scanner_geometry.replicated_modules]
    n_tof = [[scanner.tof_bin_edges[a][b].number_of_bins() for b in range(n_types)] for a in range(n_types)]

    def coincidences(most):
        return [[[petsird.CoincidenceEvent(detection_bins=[int(rng.integers(n_bins[a])), int(rng.integers(n_bins[b]))],
                                           tof_idx=int(rng.integers(n_tof[a][b])))
                  for _ in range(int(rng.integers(most + 1)))] for b in range(n_types)] for a in range(n_types)]

    def triples(levels):
        if levels == 0:
            return [petsird.TripleEvent(detection_bins=[int(rng.integers(n_bins[0])) for _ in range(3)],
                                        tof_indices=[int(rng.integers(300)), 2]) for _ in range(int(rng.integers(3)))]
        return [triples(levels - 1) for _ in range(n_types)]

    def interval(start):
        return petsird.TimeInterval(start=start, stop=start + 1)

    def transform():
        return petsird.RigidTransformation(matrix=rng.normal(size=(3, 4)).astype(np.float32))

    blocks = []
    for k, start in enumerate([0, 7, 200, 2**14 + 3, 2**21 + 9, 2**28 + 1, 2**31 + 5]):
        blocks.append(petsird.TimeBlock.EventTimeBlock(petsird.EventTimeBlock(
            time_interval=interval(start),
            single_events=[[petsird.SingleEvent(detection_bin=int(rng.integers(n_bins[t])),
                                                time_offset_in_time_block=int(rng.integers(500))) for _ in range(k)]
                           for t in range(n_types)],
            prompt_events=coincidences(9), delayed_events=coincidences(4),
            triple_events=triples(3), quadruple_events=triples(4))))
        others = [
            petsird.TimeBlock.ExternalSignalTimeBlock(petsird.ExternalSignalTimeBlock(
                time_interval=interval(start), signal_id=300, signal_values=[0.5, -1.25, float(k)])),
            petsird.TimeBlock.BedMovementTimeBlock(petsird.BedMovementTimeBlock(time_interval=interval(start),
                                                                                transform=transform())),
            petsird.TimeBlock.GantryMovementTimeBlock(petsird.GantryMovementTimeBlock(time_interval=interval(start),
                                                                                      transform=transform())),
            petsird.TimeBlock.DeadTimeTimeBlock(petsird.DeadTimeTimeBlock(
                time_interval=interval(start), alive_time_fractions=petsird.AliveTimeFractions(
                    singles_alive_time_fractions=[rng.uniform(size=n_bins[t]).astype(np.float32)
                                                  for t in range(n_types)],
                    module_coincidence_alive_time_fractions=None if k % 2 else [
                        [rng.uniform(size=(n_modules[a], n_modules[b])).astype(np.float32) for b in range(n_types)]
                        for a in range(n_types)]))),
            petsird.TimeBlock.SinglesHistogramTimeBlock(petsird.SinglesHistogramTimeBlock(
                time_interval=interval(start),
                singles_histograms=[rng.integers(0, 2**40, size=n_bins[t]).astype(np.uint64)
                                    for t in range(n_types)])),
        ]
        blocks.append(others[k % len(others)])
    with petsird.BinaryPETSIRDWriter(str(path)) as writer:
        writer.write_header(petsird.Header(scanner=scanner))
        if batches == "one list":
            writer.write_time_blocks(blocks)
        elif batches == "one at a time":
            writer.write_time_blocks(iter(blocks))
        else:
            writer.write_time_blocks(blocks[:5])
            writer.write_time_blocks(blocks[5:])


def petsird_events(path, delayeds=False):
    """The coincidences of a file as ETSI's reader reads them. Columns: the two elements (PETSIRD's numbering), the TOF
    bin, the two energy bins, the start of the time block, and the number of the event block."""
    rows = []
    with petsird.BinaryPETSIRDReader(str(path)) as petsird_reader:
        scanner = petsird_reader.read_header().scanner
        offsets = reader.get_element_offsets(scanner)
        n_energy = [scanner.event_energy_bin_edges[t].number_of_bins() for t in range(len(offsets) - 1)]
        block = -1
        for time_block in petsird_reader.read_time_blocks():
            if not isinstance(time_block, petsird.TimeBlock.EventTimeBlock):
                continue
            block += 1
            events = time_block.value.delayed_events if delayeds else time_block.value.prompt_events
            for a, row in enumerate(events):
                for b, pair in enumerate(row):
                    for e in pair:
                        b0, b1 = e.detection_bins
                        rows.append((offsets[a] + b0 // n_energy[a], offsets[b] + b1 // n_energy[b], e.tof_idx,
                                     b0 % n_energy[a], b1 % n_energy[b], time_block.value.time_interval.start, block))
    return np.array(rows, dtype=np.int64).reshape(-1, 7)


@pytest.mark.parametrize("batches", ["one list", "one at a time", "two lists"])
def test_every_kind_of_time_block_is_read_as_petsird_reads_it(tmp_path, batches):
    path = tmp_path / "f.petsird"
    write_every_kind(path, two_types(), batches=batches)
    for delayeds in (False, True):
        expected = petsird_events(path, delayeds)
        assert len(expected) > 40 and expected[:, 0].max() >= 128  # some bins take two bytes
        got = reader.get_detector_ids(path, read_tof=False, read_energy=True, delayeds=delayeds)
        assert got.dtype == torch.int64
        assert got.tolist() == expected[:, [0, 1, 3, 4]].tolist()


def test_TOF_bins_and_the_choice_of_time_blocks_match_petsird(tmp_path):
    scanner = scanner_with_random_efficiencies([module_def(num_crystals_per_module=(1, 4, 3), num_modules_along_ring=12)])
    path = tmp_path / "f.petsird"
    write_every_kind(path, scanner, seed=2)
    expected = petsird_events(path)
    assert reader.get_detector_ids(path, read_energy=True).tolist() == expected[:, :5].tolist()  # TOF read by default
    starts = sorted(set(expected[:, 5].tolist()))
    t0, t1 = starts[1], starts[-1]
    inside = (expected[:, 5] >= t0) & (expected[:, 5] < t1)
    for window in [(t0, t1), (t0 - 0.5, t1 - 0.5), (float(t0), float(t1))]:
        got = reader.get_detector_ids(path, read_tof=False, time_interval=window)
        assert got.tolist() == expected[inside][:, :2].tolist()
    assert len(reader.get_detector_ids(path, time_interval=(2**40, math.inf))) == 0
    chosen = [1, 3, 6]
    got = reader.get_detector_ids(path, read_tof=False, time_block_ids=chosen)
    assert got.tolist() == expected[np.isin(expected[:, 6], chosen)][:, :2].tolist()
    assert len(reader.get_detector_ids(path, time_block_ids=[])) == 0
    per_block = np.bincount(expected[:, 6])
    max_events = per_block[0] + 1  # reached in the first block with events after block 0
    last = int(np.nonzero(np.cumsum(per_block) >= max_events)[0][0])
    got = reader.get_detector_ids(path, read_tof=False, max_events=max_events)
    assert got.tolist() == expected[expected[:, 6] <= last][:, :2].tolist()


def test_time_blocks_start_right_after_the_header(tmp_path, one_type):
    for scanner in (one_type, two_types()):
        path = tmp_path / "header_only.petsird"
        with petsird.BinaryPETSIRDWriter(str(path)) as writer:
            writer.write_header(petsird.Header(scanner=scanner))
            writer.write_time_blocks([])
        _, start = reader._read_header_and_stream_start(path)
        assert start == path.stat().st_size - 1  # an empty stream of time blocks is its end mark, one byte
        assert len(reader.get_detector_ids(path)) == 0


def test_a_given_header_is_not_read_again(tmp_path, monkeypatch):
    path = tmp_path / "f.petsird"
    write_every_kind(path, two_types())
    header = reader.read_header(path)
    expected = reader.get_detector_ids(path, read_tof=False)
    calls = []
    original = reader._read_header_and_stream_start
    monkeypatch.setattr(reader, "_read_header_and_stream_start", lambda f: calls.append(f) or original(f))
    assert reader.get_detector_ids(path, read_tof=False, header=header).tolist() == expected.tolist()
    assert calls == []
    status = path.stat()
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 10**9))  # the file changed: its header is read again
    reader.get_detector_ids(path, read_tof=False, header=header)
    assert len(calls) == 1


def test_incomplete_or_damaged_files_raise(tmp_path):
    path = tmp_path / "f.petsird"
    write_every_kind(path, two_types())
    _, start = reader._read_header_and_stream_start(path)
    data = path.read_bytes()
    for end in (len(data) - 1, (start + len(data)) // 2, start + 3):
        cut = tmp_path / f"cut_{end}.petsird"
        cut.write_bytes(data[:end])
        with pytest.raises(ValueError, match="incomplete"):
            reader.get_detector_ids(cut, read_tof=False)
    damaged = bytearray(data)
    damaged[start + 1] = 9  # the type of the first time block, after its one-byte count
    (tmp_path / "damaged.petsird").write_bytes(bytes(damaged))
    with pytest.raises(ValueError, match="does not have"):
        reader.get_detector_ids(tmp_path / "damaged.petsird", read_tof=False)


def test_detection_bins_outside_the_scanner_raise(one_type, tmp_path):
    write(tmp_path / "f.petsird", one_type, [(0, 1, {(0, 0): [(10**6, 0, 0)]})])
    with pytest.raises(ValueError, match="outside the scanner"):
        reader.get_detector_ids(tmp_path / "f.petsird")


# --- PyTomography's standard form ---

def ring_scanner(**kwargs):
    """Rings of one module type: 12 modules around, 2 along the axis 20 mm apart, each 4 x 3 crystals of 4 x 5 mm,
    20 mm deep, from radius 200."""
    values = dict(radius=200, crystal_length=(20, 4, 5), num_crystals_per_module=(1, 4, 3), num_modules_along_ring=12,
                  num_modules_along_axis=2, module_spacing_along_axis=20.)
    values.update(kwargs)
    return scanner_with_random_efficiencies([module_def(**values)])


def test_standard_form_describes_the_scanner():
    info = reader.get_detector_info(ring_scanner())
    expected = dict(NrRings=6, NrCrystalsPerRing=48, rsectorTransNr=12, crystalTransNr=4, crystalAxialNr=3,
                    moduleAxialNr=2, rsectorAxialNr=1, moduleTransNr=1, submoduleTransNr=1, submoduleAxialNr=1,
                    TOF=1, firstCrystalAxis=1, min_rsector_difference=0)
    assert {k: info[k] for k in expected} == expected
    for key, value in dict(radius=210, crystal_length=20, crystalTransSpacing=4, crystalAxialSpacing=5,
                           moduleAxialSpacing=20, energy_window_low=430, energy_window_high=650,
                           energy_resolution=0.1).items():
        assert math.isclose(info[key], value, abs_tol=1e-4), key
    # each module's crystals sit 2 mm (half a crystal) to one side of its centre, as the generator places boxes by a
    # corner: a turn of atan(2 / 210)
    assert math.isclose(abs(info["petsird_alignment_deg"]), math.degrees(math.atan(2 / 210)), abs_tol=1e-3)
    assert all(isinstance(v, (int, float, str)) for v in info.values())  # shared's caches need hashable values


@pytest.mark.parametrize("up, towards_bed, orientation", [("+x", "+z", "HFS"), ("-y", "+z", "FFP"),
                                                          ("+y", "-z", "HFP"), ("-x", "-z", "FFS")])
def test_standard_form_puts_every_crystal_on_its_element(up, towards_bed, orientation):
    scanner = ring_scanner()
    info = reader.get_detector_info(scanner, up=up, towards_bed=towards_bed, patient_orientation=orientation)
    frame = reader.get_frame_transform(info)
    expected = reader.get_scanner_LUT(scanner).double().numpy() @ frame[:, :3].T + frame[:, 3]
    ids = reader.get_detector_id_map(scanner, info).numpy()
    assert sorted(ids.tolist()) == list(range(len(ids)))
    assert np.abs(shared.get_scanner_LUT(info).double().numpy()[ids] - expected).max() < 0.1


def test_patient_coordinates_are_LPS():
    # STIR2PETSIRD's axes (x up, z towards the bed), head first supine: L = -y, P = -x, S = -z of PETSIRD
    assert reader._patient_matrix("+x", "+z", "HFS").tolist() == [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
    for orientation in ("HFS", "FFS", "HFP", "FFP"):
        for up in ("+x", "-x", "+y", "-y"):
            for towards_bed in ("+z", "-z"):
                rows = reader._patient_matrix(up, towards_bed, orientation)  # L, P, S in gantry coordinates
                bed, up_axis = np.array(reader._AXES[towards_bed]), np.array(reader._AXES[up])
                assert np.isclose(np.linalg.det(rows), 1)
                assert np.allclose(rows[2], bed if orientation.startswith("FF") else -bed)  # head first: S into
                assert np.allclose(rows[1], -up_axis if orientation.endswith("S") else up_axis)  # supine: P down


def test_detector_ids_and_weights_in_the_standard_form(tmp_path):
    scanner = ring_scanner(num_modules_along_ring=6)
    info = reader.get_detector_info(scanner)
    id_map = reader.get_detector_id_map(scanner, info).numpy()
    path = tmp_path / "f.petsird"
    write_every_kind(path, scanner, seed=3)
    raw = reader.get_detector_ids(path, read_energy=True)
    standard = reader.get_detector_ids(path, read_energy=True, info=info)
    assert standard[:, :2].tolist() == id_map[raw[:, :2].numpy()].tolist()
    assert standard[:, 2:].tolist() == raw[:, 2:].tolist()
    n = len(id_map)
    pairs = torch.combinations(torch.arange(n), 2).numpy()
    a, b = id_map[pairs[:, 0]], id_map[pairs[:, 1]]
    weights = reader.get_sensitivity_weights(scanner, info=info).numpy()
    assert np.array_equal(weights[reader._pair_index(np.minimum(a, b), np.maximum(a, b), n)],
                          reader.get_sensitivity_weights(scanner).numpy())
    assert reader.get_detector_info(path) == reader.get_detector_info(reader.read_header(path))  # from a file


def test_scanners_without_a_standard_form_raise(one_type):
    with pytest.raises(ValueError, match="one type of module"):
        reader.get_detector_info(scanner_with_random_efficiencies([module_def(), module_def(radius=150)]))
    with pytest.raises(ValueError, match="cylinders of rings"):  # two layers of crystals
        reader.get_detector_info(scanner_with_random_efficiencies([module_def(num_crystals_per_module=(2, 2, 2))]))
    with pytest.raises(ValueError, match="towards_bed"):
        reader.get_detector_info(one_type, towards_bed="+x")
    with pytest.raises(ValueError, match="different axes"):
        reader.get_detector_info(one_type, up="+z", towards_bed="+z")
