"""PET in PyTomography's standard form: the scanner description (``scanner_info``, ``scanner_lut``) and the readers of
GATE, GE and the PET corrections, on small files written here (CPU only).

The GATE files are a macro and a ROOT file written with uproot, whose coincidences have known copy numbers, known
interaction points (``globalPos``, placed by GATE's own rules: the ring repeater turns rsector k by 360 k / N degrees
counter-clockwise, copy numbers run transaxially first) and known times, so the scanner lookup table, the frame and
the TOF bins can be checked against them."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")
uproot = pytest.importorskip("uproot")

from pytomography.io import PET  # noqa: E402
from pytomography.io.PET import gate, ge, shared  # noqa: E402
from pytomography.io.PET.scanner import (crystal_pair_index, frame_keys_after, keep_rsector_difference,  # noqa: E402
                                         rsector_difference, scanner_info, scanner_lut)
from pytomography.metadata import ObjectMeta  # noqa: E402
from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta  # noqa: E402

# a small GATE scanner: 12 rsectors around a ring, each 2 modules along z of 4 x 3 crystals (4 mm around, 5 mm along z,
# 10 mm deep), the rsector's centre 100 mm from the axis
MACRO = """# a small cylindrical PET system
/gate/world/daughters/name cylindricalPET
/gate/world/daughters/insert cylinder
/gate/cylindricalPET/daughters/name rsector
/gate/cylindricalPET/daughters/insert box
/gate/rsector/placement/setTranslation 10.0 0 0 cm
/gate/rsector/geometry/setXLength 10.0 mm
/gate/rsector/daughters/name module
/gate/rsector/daughters/insert box
/gate/module/daughters/name crystal
/gate/module/daughters/insert box
/gate/crystal/placement/setTranslation 0 0 0 mm
/gate/crystal/geometry/setXLength 10.0 mm   # the crystals' depth
/gate/crystal/repeaters/insert cubicArray
/gate/crystal/cubicArray/setRepeatNumberX 1
/gate/crystal/cubicArray/setRepeatNumberY 4
/gate/crystal/cubicArray/setRepeatNumberZ 3
/gate/crystal/cubicArray/setRepeatVector 0. 4.0 5.0 mm
/gate/module/repeaters/insert cubicArray
/gate/module/cubicArray/setRepeatNumberX 1
/gate/module/cubicArray/setRepeatNumberY 1
/gate/module/cubicArray/setRepeatNumberZ 2
/gate/module/cubicArray/setRepeatVector 0. 0. 16.0 mm
/gate/rsector/repeaters/insert ring
/gate/rsector/ring/setRepeatNumber 12
/gate/systems/cylindricalPET/rsector/attach rsector
"""
N_RSECTORS, N_MODULES, N_TRANS, N_AXIAL = 12, 2, 4, 3
DOI = 3.0


def gate_position(rsector, module, crystal, depth=DOI):
    """Where GATE puts an interaction ``depth`` mm inside the face of a crystal, in GATE's world (mm)."""
    trans, axial = crystal % N_TRANS, crystal // N_TRANS
    y = (trans - (N_TRANS - 1) / 2) * 4.0
    z = (module - (N_MODULES - 1) / 2) * 16.0 + (axial - (N_AXIAL - 1) / 2) * 5.0
    x = 100.0 - 5.0 + depth
    a = 2 * np.pi * rsector / N_RSECTORS
    return np.stack([x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a), z], axis=-1)


def write_gate(folder, n=400, seed=0, tof_positions=None, kinds=None, delays=0):
    """A macro and one ROOT file of ``n`` coincidences between random crystals. ``tof_positions``: (t1 - t2) c / 2 of
    each event in mm. ``kinds``: 'true', 'scatter' or 'random' for each event."""
    rng = np.random.default_rng(seed)
    (folder / "scanner.mac").write_text(MACRO)
    columns = {}
    trees = {"Coincidences": n, "delay": delays}
    path = folder / "run.root"
    with uproot.recreate(path) as f:
        for tree, count in trees.items():
            if count == 0:
                continue
            branches = {}
            for j in (1, 2):
                branches[f"rsectorID{j}"] = rng.integers(0, N_RSECTORS, count).astype(np.int32)
                branches[f"moduleID{j}"] = rng.integers(0, N_MODULES, count).astype(np.int32)
                branches[f"submoduleID{j}"] = np.zeros(count, np.int32)
                branches[f"crystalID{j}"] = rng.integers(0, N_TRANS * N_AXIAL, count).astype(np.int32)
                g = gate_position(branches[f"rsectorID{j}"], branches[f"moduleID{j}"], branches[f"crystalID{j}"])
                for k, axis in enumerate("XYZ"):
                    branches[f"globalPos{axis}{j}"] = g[:, k].astype(np.float32)
            source = rng.uniform(-30, 30, (count, 3))
            for j in (1, 2):
                for k, axis in enumerate("XYZ"):
                    branches[f"sourcePos{axis}{j}"] = source[:, k].astype(np.float32).copy()
                branches[f"comptonPhantom{j}"] = np.zeros(count, np.int32)
                branches[f"RayleighPhantom{j}"] = np.zeros(count, np.int32)
            if tree == "Coincidences" and kinds is not None:
                for i, kind in enumerate(kinds):
                    if kind == "random":
                        branches["sourcePosX2"][i] += 50.0
                    elif kind == "scatter":
                        branches["comptonPhantom2"][i] = 1
            position = np.zeros(count) if tof_positions is None or tree != "Coincidences" else np.asarray(tof_positions)
            branches["time2"] = np.full(count, 1.0)
            branches["time1"] = 1.0 + 2 * position / 0.3 * 1e-12   # (t1 - t2) c / 2 = position, c = 0.3 mm/ps
            f.mktree(tree, {k: v.dtype for k, v in branches.items()})
            f[tree].extend(branches)
            columns[tree] = branches
    return path, folder / "scanner.mac", columns


# ---------- the scanner description ----------

def test_scanner_info_checks_and_types():
    info = scanner_info(crystalTransNr=np.int64(4), crystalTransSpacing=np.float32(4.0), crystalAxialNr=3,
                        crystalAxialSpacing=5, moduleTransNr=1, moduleAxialNr=2, moduleAxialSpacing=16.0,
                        rsectorTransNr=12, radius=98.0, doi_mm=3)
    assert info["NrCrystalsPerRing"] == 48 and info["NrRings"] == 6
    assert all(type(v) in (int, float, str, type(None)) for v in info.values())
    assert (info["rsector0_angle_deg"], info["transaxial_direction"], info["axial_direction"]) == (0.0, 1, 1)
    assert info["energy_window_low_keV"] is None and info["doi_mm"] == 3.0
    for old in ("firstCrystalAxis", "energy_window_low", "mean_interaction_depth"):
        with pytest.raises(ValueError, match="replaced"):
            scanner_info(info, **{old: 1})
    with pytest.raises(ValueError, match="NrRings"):
        scanner_info(info, NrRings=7)
    with pytest.raises(ValueError, match="moduleAxialSpacing"):
        scanner_info({k: v for k, v in info.items() if k != "moduleAxialSpacing"})
    with pytest.raises(ValueError, match="1 or -1"):
        scanner_info(info, axial_direction=0)
    with pytest.raises(ValueError, match="upper limit"):
        scanner_info(info, energy_window_low_keV=500, energy_window_high_keV=400)
    with pytest.raises(ValueError, match="frame keys"):
        scanner_lut({k: v for k, v in info.items() if k != "axial_direction"})
    assert scanner_info(info, min_rsector_difference=2)["min_rsector_difference"] == 2


def test_the_table_scanner_is_upright():
    """The GE Discovery MI of PyTomography's table: crystal 0 at the top (anterior, -y), and the old geometry
    (firstCrystalAxis=0 and z negated) turned 180 degrees about the axis."""
    info = scanner_info("discovery_MI")
    assert (info["NrCrystalsPerRing"], info["NrRings"]) == (544, 36)
    lut = scanner_lut(info).double()
    assert lut[0, 1] < -370 and abs(lut[0, 0]) < 40
    old = scanner_lut(scanner_info(info, rsector0_angle_deg=90))           # PyTomography 3's GE geometry
    assert torch.allclose(lut, old * torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64), atol=1e-3)


@pytest.mark.parametrize("angle, transaxial, axial", [(0, 1, 1), (90, -1, -1), (30, -1, 1), (200, 1, -1)])
def test_scanner_lut_follows_the_frame_keys(angle, transaxial, axial):
    info = scanner_info(crystalTransNr=4, crystalTransSpacing=4.0, crystalAxialNr=3, crystalAxialSpacing=5.0,
                        moduleTransNr=1, moduleAxialNr=2, moduleAxialSpacing=16.0, rsectorTransNr=12, radius=98.0,
                        rsector0_angle_deg=angle, transaxial_direction=transaxial, axial_direction=axial)
    lut = scanner_lut(info).double().numpy()
    centre0 = lut[: 4].mean(axis=0)                         # the crystals of rsector 0 around its centre, in ring 0
    assert math.isclose(math.degrees(math.atan2(centre0[1], centre0[0])) % 360, angle % 360, abs_tol=1e-3)
    step = math.degrees(math.atan2(lut[4, 1], lut[4, 0]) - math.atan2(centre0[1], centre0[0])) % 360
    assert math.isclose(step, 30 if transaxial == 1 else 330, abs_tol=5)   # rsector 1 is 30 degrees one way
    tangent = np.array([-math.sin(math.radians(angle)), math.cos(math.radians(angle))])
    assert np.sign((lut[1, :2] - lut[0, :2]) @ tangent) == transaxial
    assert np.sign(lut[48, 2] - lut[0, 2]) == axial         # ring 1 against ring 0


def test_frame_keys_after_a_rotation_or_reflection():
    info = scanner_info(crystalTransNr=4, crystalTransSpacing=4.0, crystalAxialNr=3, crystalAxialSpacing=5.0,
                        moduleTransNr=1, moduleAxialNr=2, moduleAxialSpacing=16.0, rsectorTransNr=12, radius=98.0,
                        rsector0_angle_deg=20.0)
    lut = scanner_lut(info).double().numpy()
    for letters in ("LPS", "LPI", "RAS", "RAI", "PLS", "ALI", "PRS"):
        m = gate.world_matrix(letters)
        moved = scanner_lut(scanner_info(info, **frame_keys_after(m, 20.0, 1, 1))).double().numpy()
        assert np.abs(moved - lut @ m.T).max() < 1e-3, letters
    with pytest.raises(ValueError, match="head-foot"):
        gate.world_matrix("LSP")
    with pytest.raises(ValueError, match="three letters"):
        gate.world_matrix("LPX")


def test_crystal_pair_index_is_the_combinations_order_for_arrays_and_tensors():
    pairs = torch.combinations(torch.arange(9), 2)
    assert torch.equal(crystal_pair_index(pairs[:, 0], pairs[:, 1], 9), torch.arange(len(pairs)))
    assert (crystal_pair_index(pairs[:, 0].numpy(), pairs[:, 1].numpy(), 9) == np.arange(len(pairs))).all()


# ---------- GATE ----------

@pytest.mark.parametrize("world_axes", ["LPS", "LPI", "RAI", "PLS"])
def test_gate_lut_lands_on_the_interaction_points(tmp_path, world_axes):
    """Every event's two crystals, in the object frame and then in the patient, are where GATE recorded the
    interactions (globalPos), taken to the patient by world_axes and world_origin: within 0.1 mm."""
    root, macro, columns = write_gate(tmp_path)
    origin = (12.5, -40.0, 7.0)
    events, object_meta, proj_meta = PET.read_gate(root, macro, world_axes=world_axes, world_origin=origin,
                                                   doi_mm=DOI)
    assert events.dtype == torch.int32 and events.shape == (400, 2)
    assert object_meta.frame == "patient" and object_meta.modality == "PT" and object_meta.units == "counts"
    m = gate.world_matrix(world_axes)
    lut = proj_meta.scanner_lut.double().numpy()
    for j in (0, 1):
        c = columns["Coincidences"]
        g = np.stack([c[f"globalPos{a}{j + 1}"] for a in "XYZ"], axis=1).astype(np.float64)
        patient = g @ m.T + np.array(origin)
        found = lut[events[:, j].long().numpy()] + np.array(object_meta.centre)
        assert np.abs(found - patient).max() < 0.1
    info = proj_meta.info
    assert info["radius"] == 98.0 and info["doi_mm"] == DOI and info["crystal_length"] == 10.0


def test_gate_without_an_origin_is_in_the_scanner_frame(tmp_path):
    root, macro, _ = write_gate(tmp_path, n=10)
    _, object_meta, proj_meta = PET.read_gate(root, macro)
    assert object_meta.frame == "scanner" and object_meta.centre == (0.0, 0.0, 0.0)
    info = proj_meta.info
    assert (info["rsector0_angle_deg"], info["transaxial_direction"], info["axial_direction"]) == (0.0, 1, 1)
    # the default grid: half the crystal pitch, 80 % of the ring's diameter, every ring
    assert object_meta.dr == (2.0, 2.0, 2.5)
    assert object_meta.shape[0] == math.ceil(0.8 * 2 * 95 / 2.0)
    assert object_meta.shape[2] * 2.5 >= 16 + 3 * 5 - 1e-6


def test_gate_tof_bins_and_out_of_range_events(tmp_path):
    """Bin k of n holds (t1 - t2) c / 2 in [(k - n/2) w, (k + 1 - n/2) w]; events outside every bin are dropped."""
    positions = [-150, -50, 50, 150, 250, -250, -99, 99]
    root, macro, _ = write_gate(tmp_path, n=len(positions), tof_positions=positions)
    tof_meta = PETTOFMeta(num_bins=4, tof_range=400.0, fwhm=50.0)
    events, _, proj_meta = PET.read_gate(root, macro, tof_meta=tof_meta)
    assert events.dtype == torch.int32 and events.shape == (6, 3)
    assert events[:, 2].tolist() == [0, 1, 2, 3, 1, 2]
    assert proj_meta.tof_meta is tof_meta
    assert float(tof_meta.center_offset) == 0.0                            # centred for an even number of bins (#7)


def test_gate_event_kinds_and_delays(tmp_path):
    kinds = ["true", "scatter", "random", "true", "random", "scatter", "true"]
    root, macro, columns = write_gate(tmp_path, n=len(kinds), kinds=kinds, delays=5)
    every, _, _ = PET.read_gate(root, macro)
    rows = {k: [i for i, kind in enumerate(kinds) if kind == k] for k in ("true", "scatter", "random")}
    for events, wanted in [("trues", rows["true"]), ("scatters", rows["scatter"]), ("randoms", rows["random"]),
                           (("trues", "scatters"), sorted(rows["true"] + rows["scatter"]))]:
        got, _, _ = PET.read_gate(root, macro, events=events)
        assert got.tolist() == every[wanted].tolist(), events
    delayeds, _, _ = PET.read_gate(root, macro, events="delayeds")
    assert delayeds.shape == (5, 2)
    with pytest.raises(ValueError, match="prompts"):
        PET.read_gate(root, macro, events="everything")


def test_gate_min_rsector_difference_filters_events(tmp_path):
    """min_rsector_difference drops the events whose crystals are fewer rsectors apart; the sinogram is unchanged
    (#15)."""
    root, macro, _ = write_gate(tmp_path, n=300)
    every, _, proj_meta = PET.read_gate(root, macro)
    kept, _, kept_meta = PET.read_gate(root, macro, min_rsector_difference=3)
    difference = rsector_difference(every, proj_meta.info)
    assert kept.tolist() == every[difference >= 3].tolist() and len(kept) < len(every)
    assert kept_meta.info["min_rsector_difference"] == 3
    assert shared._sinogram_shape(kept_meta.info) == shared._sinogram_shape(proj_meta.info)
    for table, other in zip(shared.sinogram_coordinates(kept_meta.info), shared.sinogram_coordinates(proj_meta.info)):
        assert torch.equal(table, other)
    # every pair lands in the bin whose geometry holds its two crystals
    detector_coordinates, _ = shared.sinogram_to_spatial(kept_meta.info)
    lor, _ = shared.sinogram_coordinates(kept_meta.info)
    lut = scanner_lut(kept_meta.info)
    c = kept_meta.info["NrCrystalsPerRing"]
    a, b = kept[:, 0].long() % c, kept[:, 1].long() % c
    hi, lo = torch.maximum(a, b), torch.minimum(a, b)
    angle, radial = lor[hi, lo].T
    assert torch.allclose(detector_coordinates[angle, radial, 0], lut[hi, :2], atol=1e-4)
    assert torch.allclose(detector_coordinates[angle, radial, 1], lut[lo, :2], atol=1e-4)
    assert keep_rsector_difference(every, proj_meta.info) is every


def test_gate_normalization_weights_are_checked_and_nan_is_zero(tmp_path):
    root, macro, _ = write_gate(tmp_path, n=10)
    _, _, proj_meta = PET.read_gate(root, macro)
    n = 48 * 6
    weights = np.ones(n * (n - 1) // 2, dtype=np.float32)
    weights[[3, 7, 11]] = np.nan
    np.save(tmp_path / "weights.npy", weights)
    with pytest.warns(UserWarning, match="3 of"):
        _, _, with_weights = PET.read_gate(root, macro, normalization=tmp_path / "weights.npy")
    assert with_weights.weights_sensitivity.dtype == torch.float32
    assert int((with_weights.weights_sensitivity == 0).sum()) == 3 and not torch.isnan(with_weights.weights_sensitivity).any()
    with pytest.warns(UserWarning, match="NaN"):
        PET.read_gate(root, macro, normalization=torch.from_numpy(weights))
    assert np.isnan(weights).sum() == 3                                    # the caller's array is untouched
    with pytest.raises(ValueError, match="one weight per crystal pair"):
        PET.read_gate(root, macro, normalization=weights[:-1])


def test_gate_normalization_from_a_calibration_scan(tmp_path):
    """gate_normalization gives one weight per pair, without NaN: pairs outside the calibration cylinder get 0."""
    root, macro, _ = write_gate(tmp_path, n=2000)
    _, _, proj_meta = PET.read_gate(root, macro)
    with pytest.warns(UserWarning, match="NaN"):
        weights = PET.gate_normalization(root, proj_meta.info, cylinder_radius_mm=60.0)
    n = proj_meta.scanner_lut.shape[0]
    assert weights.shape == (n * (n - 1) // 2,) and not torch.isnan(weights).any()
    pairs = torch.combinations(torch.arange(n), 2)
    outside = gate.lor_radius(pairs, proj_meta.scanner_lut).abs() > 60.0
    assert bool((weights[outside] == 0).all())


def test_gate_placement_of_a_hosted_dataset_comes_from_the_registry(tmp_path, monkeypatch):
    """Decision 5: the files of a dataset the registry gives a GATE placement are read in that placement without the
    caller passing it; an explicit placement wins."""
    from pytomography.datasets import registry
    root, macro, _ = write_gate(tmp_path, n=20)
    entry = {"title": "test", "files": {f"runs/{root.name}": [root.stat().st_size, "0" * 64]},
             "geometry": {"gate": {"world_axes": "LPI", "world_origin": [1.0, 2.0, 3.0]}}}
    monkeypatch.setitem(registry.DATASETS, "PET/test-gate", entry)
    events, object_meta, proj_meta = PET.read_gate(root, macro)
    assert object_meta.frame == "patient" and object_meta.centre == (1.0, 2.0, 3.0)
    assert proj_meta.info["axial_direction"] == -1
    _, explicit, explicit_meta = PET.read_gate(root, macro, world_axes="LPS")
    assert explicit.frame == "scanner" and explicit_meta.info["axial_direction"] == 1


def test_hosted_gate_datasets_record_their_placement():
    from pytomography.datasets import registry
    placement = registry.DATASETS["PET/GATE-mMR-BrainInWater"]["geometry"]["gate"]
    assert placement["world_axes"] == "LPI"
    assert np.allclose(placement["world_origin"], (-0.35, 17.65, 22.35))


def test_gate_macros_the_reader_cannot_describe_raise(tmp_path):
    (tmp_path / "y.mac").write_text(MACRO.replace("setTranslation 10.0 0 0 cm", "setTranslation 0 100 0 mm"))
    with pytest.raises(NotImplementedError, match="along \\+x"):
        gate._macro_info(tmp_path / "y.mac")
    (tmp_path / "linear.mac").write_text(MACRO.replace("/gate/rsector/repeaters/insert ring", "/gate/rsector/repeaters/insert sphere"))
    with pytest.raises(NotImplementedError, match="sphere"):
        gate._macro_info(tmp_path / "linear.mac")


# ---------- events outside the grid (#16), randoms in TOF ----------

def _crosses_box(a: np.ndarray, b: np.ndarray, half: np.ndarray, samples: int = 4001) -> bool:
    t = np.linspace(0, 1, samples)[:, None]
    points = a + t * (b - a)
    return bool(np.all(np.abs(points) < half, axis=1).any())


def test_remove_events_outside_keeps_exactly_the_lors_that_cross_the_grid():
    info = scanner_info(crystalTransNr=4, crystalTransSpacing=4.0, crystalAxialNr=3, crystalAxialSpacing=5.0,
                        moduleTransNr=1, moduleAxialNr=4, moduleAxialSpacing=16.0, rsectorTransNr=12, radius=98.0)
    lut = scanner_lut(info).double().numpy()
    meta = ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(30, 20, 10))         # a box of 60 x 40 x 20 mm
    half = np.array(meta.shape) * np.array(meta.dr) / 2
    rng = np.random.default_rng(1)
    events = torch.from_numpy(rng.integers(0, lut.shape[0], (3000, 2)).astype(np.int32))
    events = events[events[:, 0] != events[:, 1]]
    expected = np.array([_crosses_box(lut[i], lut[j], half) for i, j in events.long().numpy()])
    got = PET.remove_events_outside(events, scanner_lut(info), meta)
    margin = np.array([_crosses_box(lut[i], lut[j], half - 0.2) != _crosses_box(lut[i], lut[j], half + 0.2)
                       for i, j in events.long().numpy()])     # LORs that graze the box's faces
    kept = set(map(tuple, got.tolist()))
    for row, inside, near in zip(events.tolist(), expected, margin):
        if not near:
            assert (tuple(row) in kept) == inside
    assert 0 < len(got) < len(events)
    assert torch.equal(PET.remove_events_outside(events, info, meta), got)


def test_randoms_to_tof_does_not_change_its_input():
    randoms = torch.rand(4, 5, 6)
    before = randoms.clone()
    tof = PET.randoms_sinogram_to_sinogramTOF(randoms, PETTOFMeta(5, 500.0, 60.0), coincidence_timing_width=4000.0)
    assert torch.equal(randoms, before)                                    # #10
    assert torch.allclose(tof, before * 100.0 / (4000.0 * 0.3))


# ---------- the scatter estimate's rings, whichever way they are numbered ----------

@pytest.mark.parametrize("axial_direction", [1, -1])
def test_sss_interpolates_the_rings_in_z_whichever_way_they_run(axial_direction):
    """The scatter estimate's axial interpolation is in z, so a sinogram linear in the two rings' z is interpolated
    exactly, with ring 0 at either end (the old code assumed ring 0 at +z)."""
    from pytomography.metadata.PET import PETSinogramPolygonProjMeta
    from pytomography.utils import sss
    info = scanner_info(crystalTransNr=4, crystalTransSpacing=4.0, crystalAxialNr=3, crystalAxialSpacing=5.0,
                        moduleTransNr=1, moduleAxialNr=4, moduleAxialSpacing=16.0, rsectorTransNr=12, radius=98.0,
                        axial_direction=axial_direction)
    proj_meta = PETSinogramPolygonProjMeta(info)
    idx_intraring, idx_ring, _ = sss.get_sample_detector_ids(proj_meta, 4, 4)
    _, ring_coordinates = shared.sinogram_to_spatial(info)
    shape = shared._sinogram_shape(info)
    plane_value = ring_coordinates[:, 0] + 2 * ring_coordinates[:, 1] + 200.0
    dense = plane_value.reshape(1, 1, -1).expand(shape).clone()
    interpolated = sss.interpolate_sparse_sinogram(dense, proj_meta, idx_intraring, idx_ring, lazy=False).cpu()
    lor, _ = shared.sinogram_coordinates(info)
    pairs = torch.combinations(torch.arange(info["NrCrystalsPerRing"]), 2)
    angle, radial = lor[pairs[:, 0], pairs[:, 1]].T
    assert torch.allclose(interpolated[angle, radial], dense[angle, radial], rtol=1e-3)


# ---------- GE ----------

def write_ge(folder, raw_tof):
    """A GE RDF list mode file with SystemGeometry, an energy window and events (ring, crystal, ring, crystal, TOF), and
    a corrections file."""
    n = len(raw_tof)
    rng = np.random.default_rng(2)
    events = np.stack([rng.integers(0, 36, n), rng.integers(0, 544, n), rng.integers(0, 36, n), rng.integers(0, 544, n),
                       np.asarray(raw_tof)], axis=1).astype(np.int16)
    path = folder / "LIST0000.BLF"
    with h5py.File(path, "w") as f:
        f["MiceList/TofCoinc"] = events
        geometry = dict(radialModulesPerSystem=34, radialBlocksPerModule=4, radialCrystalsPerBlock=4,
                        axialModulesPerSystem=1, axialBlocksPerModule=4, axialCrystalsPerBlock=9)
        for key, value in geometry.items():
            f[f"HeaderData/SystemGeometry/{key}"] = np.array([value])
        f["HeaderData/AcqParameters/EDCATParameters/lower_energy_limit"] = np.array([425])
        f["HeaderData/AcqParameters/EDCATParameters/upper_energy_limit"] = np.array([650])
        f["HeaderData/AcqParameters/RxScanParameters/acqTime"] = np.array([345])
    corrections = folder / "corrections.h5"
    pairs = np.stack([rng.integers(0, 544, 7), rng.integers(0, 36, 7), rng.integers(0, 544, 7), rng.integers(0, 36, 7)],
                     axis=1).astype(np.int16)                               # (crystal, ring, crystal, ring)
    with h5py.File(corrections, "w") as f:
        f["correction_lists/atten"] = rng.uniform(0.1, 1, n).astype(np.float32)
        f["correction_lists/sens"] = rng.uniform(0.5, 2, n).astype(np.float32)
        f["correction_lists/contam"] = rng.uniform(0, 1, n).astype(np.float32)
        f["all_xtals/xtal_ids"] = pairs
        f["all_xtals/atten"] = rng.uniform(0.1, 1, 7).astype(np.float32)
        f["all_xtals/sens"] = rng.uniform(0.5, 2, 7).astype(np.float32)
    return path, corrections, events, pairs


def test_ge_hdf5_events_tof_frame_and_energy_window(tmp_path):
    raw = [13 * k for k in range(-14, 15)] + [13 * 15, -13 * 15]          # 29 bins, and two outside them
    path, _, events, _ = write_ge(tmp_path, raw)
    got, object_meta, proj_meta = PET.read_ge_hdf5(path)
    assert got.dtype == torch.int32 and got.shape == (29, 3)
    expected = np.stack([544 * events[:29, 0] + events[:29, 1], 544 * events[:29, 2] + events[:29, 3],
                         14 - events[:29, 4] // 13], axis=1)
    assert got.tolist() == expected.tolist()
    info = proj_meta.info
    assert (info["energy_window_low_keV"], info["energy_window_high_keV"]) == (425.0, 650.0)
    assert proj_meta.tof_meta.num_bins == 29 and proj_meta.duration_s == 345.0
    assert object_meta.frame == "scanner" and object_meta.modality == "PT"
    assert proj_meta.scanner_lut[0, 1] < -370                              # crystal 0 at the top: the bed posterior


def test_ge_hdf5_of_another_scanner_raises(tmp_path):
    path, _, _, _ = write_ge(tmp_path, [0])
    with h5py.File(path, "r+") as f:
        f["HeaderData/SystemGeometry/radialModulesPerSystem"][0] = 30
    with pytest.raises(ValueError, match="crystals around"):
        PET.read_ge_hdf5(path)


def test_ge_corrections(tmp_path):
    path, corrections, events, pairs = write_ge(tmp_path, [0, 13, -13])
    event_weights, additive, pair_ids, pair_weights = PET.read_ge_corrections(corrections)
    with h5py.File(corrections, "r") as f:
        assert np.allclose(event_weights.numpy(), f["correction_lists/atten"][:] * f["correction_lists/sens"][:])
        assert np.allclose(additive.numpy(), f["correction_lists/contam"][:])
        assert np.allclose(pair_weights.numpy(), f["all_xtals/atten"][:] * f["all_xtals/sens"][:])
    assert pair_ids.dtype == torch.int32
    assert pair_ids.tolist() == np.stack([544 * pairs[:, 1] + pairs[:, 0], 544 * pairs[:, 3] + pairs[:, 2]], 1).tolist()
    assert PET.read_ge_corrections(corrections, pairs=False)[2] is None
