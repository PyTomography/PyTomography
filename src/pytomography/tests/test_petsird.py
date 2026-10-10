"""Tests of pytomography.io.PET.petsird on small PETSIRD files written here with ETSI's petsird package (CPU only).

The scanners come from petsird's own example generator, with random efficiencies, and every result is compared with
petsird's helpers (geometry, detection bins, detection efficiency)."""
import math

import numpy as np
import pytest
import torch

petsird = pytest.importorskip("petsird")
from petsird.helpers import generator, geometry, get_detection_efficiency, make_detection_bin  # noqa: E402

from pytomography.io.PET import petsird as reader  # noqa: E402


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
