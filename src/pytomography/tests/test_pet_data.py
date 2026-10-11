"""The PET readers on the tutorial data: the checks of the I/O audit (D:/v4runs/io_audit, pet_verify_*) as tests. Each
reduces to a number with a threshold, and the wrong frame or sign scores worse.

They need the data (``PYTOMOGRAPHY_DATA``), and run on the CPU in a few GB of memory: one GATE run of 2.2 million
coincidences, the first 20 million GE events and one ring's crystal pairs, the PETSIRD header (3.6 GB while it is read)
and its first 3 million prompts."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from pytomography.io import PET

pytestmark = pytest.mark.data

GATE = "PET/GATE-mMR-BrainInWater"


def _file(data_dir, *parts):
    path = data_dir.joinpath(*parts)
    if not path.exists():
        pytest.skip(f"{path} is not in the data folder")
    return path


def _branches(path, names):
    uproot = pytest.importorskip("uproot")
    with uproot.open(path) as f:
        return f["Coincidences"].arrays(names, library="np")


def _gate_run(data_dir):
    return (_file(data_dir, GATE, "simple_phantom", "mMR_voxBrain_withSimplePhantom_1.root"),
            _file(data_dir, GATE, "mMR_Geometry.mac"))


def test_gate_lut_lands_on_gate_interaction_points(data_dir):
    """The hosted GATE data, read in the registry's placement ('LPI', the MR's centre): each event's crystals are where
    GATE recorded the interactions (median |dz| < 2 mm), and the annihilations lie on their LORs (median distance <
    3 mm). The rings numbered the other way along the axis miss both."""
    root, macro = _gate_run(data_dir)
    names = [f"{k}{a}{j}" for k in ("globalPos", "sourcePos") for a in "XYZ" for j in (1, 2)]
    names += [f"{k}{j}" for k in ("comptonPhantom", "RayleighPhantom") for j in (1, 2)]
    b = _branches(root, names)
    events, object_meta, proj_meta = PET.read_gate(root, macro, doi_mm=9)
    assert object_meta.frame == "patient" and proj_meta.info["axial_direction"] == -1
    m = PET.gate.world_matrix("LPI")
    centre = np.asarray(object_meta.centre)
    g1 = np.stack([b[f"globalPos{x}1"] for x in "XYZ"], 1).astype(np.float64) @ m.T + centre
    true = (np.all(np.stack([b[f"sourcePos{x}1"] == b[f"sourcePos{x}2"] for x in "XYZ"]), axis=0)
            & (b["comptonPhantom1"] == 0) & (b["comptonPhantom2"] == 0) & (b["RayleighPhantom1"] == 0)
            & (b["RayleighPhantom2"] == 0))
    source = np.stack([b[f"sourcePos{x}1"] for x in "XYZ"], 1)[true].astype(np.float64) @ m.T + centre
    scores = {}
    for name, info in (("reader", proj_meta.info), ("rings reversed", PET.scanner_info(proj_meta.info, axial_direction=1))):
        lut = PET.scanner_lut(info).double().numpy() + centre
        a, c = lut[events[:, 0].long().numpy()], lut[events[:, 1].long().numpy()]
        dz = np.median(np.abs(a[:, 2] - g1[:, 2]))
        u = (c - a)[true] / np.linalg.norm((c - a)[true], axis=1, keepdims=True)
        w = source - a[true]
        scores[name] = (dz, np.median(np.linalg.norm(w - (w * u).sum(1, keepdims=True) * u, axis=1)))
    assert scores["reader"][0] < 2 and scores["reader"][1] < 3, scores
    assert scores["rings reversed"][0] > 20 and scores["rings reversed"][1] > 10, scores


def test_gate_hosted_placement_is_where_v3_saved_the_images(data_dir):
    """Decision 5: for the hosted data, the registry's placement puts a reconstruction grid exactly where PyTomography 3
    put it with gate.get_patient_affine_from_nifti(MR): the MR centred on the scanner, its first two axes reversed."""
    nib = pytest.importorskip("nibabel")
    root, macro = _gate_run(data_dir)
    mr = nib.load(str(_file(data_dir, GATE, "fdg_pet_phantom_mri.nii.gz")))
    grid = PET.read_gate(root, macro, doi_mm=9, object_meta=__import__("pytomography").metadata.ObjectMeta(
        dr=(2.0, 2.0, 2.0), shape=(128, 128, 96)))[1]
    n = np.asarray(mr.shape[:3], float)
    d = np.asarray(mr.header["pixdim"][1:4], float) * [-1, -1, 1]
    m_nifti = np.diag([*d, 1.0])
    m_nifti[:3, 3] = -(n - 1) / 2 * d
    m_pet = np.diag([2.0, 2.0, 2.0, 1.0])
    m_pet[:3, 3] = -(np.array([128, 128, 96]) - 1) / 2 * 2.0
    v3 = np.diag([-1.0, -1.0, 1.0, 1.0]) @ mr.affine @ np.linalg.inv(m_nifti) @ m_pet
    assert np.abs(grid.affine - v3).max() < 1e-3


def test_gate_tof_bins_rise_with_the_true_position(data_dir):
    """True unscattered GATE events: the mean centre of the reader's TOF bin rises with the annihilation's true position
    along the LOR (from column 0's crystal towards column 1's)."""
    from pytomography.metadata.PET import PETTOFMeta
    root, macro = _gate_run(data_dir)
    tof_meta = PETTOFMeta(21, 300.0, 82.5)
    events, object_meta, proj_meta = PET.read_gate(root, macro, doi_mm=9, tof_meta=tof_meta)
    b = _branches(root, [f"sourcePos{x}{j}" for x in "XYZ" for j in (1, 2)] + ["time1", "time2"] +
                  [f"{k}{j}" for k in ("comptonPhantom", "RayleighPhantom") for j in (1, 2)])
    position = (b["time1"] - b["time2"]) * 1e12 * 0.15
    kept = (position >= -150) & (position < 150)                 # the events the reader keeps, in order
    assert int(kept.sum()) == events.shape[0]
    true = (np.all(np.stack([b[f"sourcePos{x}1"] == b[f"sourcePos{x}2"] for x in "XYZ"]), axis=0)
            & (b["comptonPhantom1"] == 0) & (b["comptonPhantom2"] == 0) & (b["RayleighPhantom1"] == 0)
            & (b["RayleighPhantom2"] == 0))[kept]
    lut = proj_meta.scanner_lut.double().numpy() + np.asarray(object_meta.centre)
    a, c = lut[events[true, 0].long().numpy()], lut[events[true, 1].long().numpy()]
    m = PET.gate.world_matrix("LPI")
    source = np.stack([b[f"sourcePos{x}1"] for x in "XYZ"], 1)[kept][true].astype(np.float64) @ m.T + np.asarray(object_meta.centre)
    u = (c - a) / np.linalg.norm(c - a, axis=1, keepdims=True)
    along = ((source - (a + c) / 2) * u).sum(1)
    centre = tof_meta.bin_positions.numpy()[events[true, 2].long().numpy()]
    means = [centre[(along >= lo) & (along < lo + 20)].mean() for lo in range(-100, 40, 20)]
    assert np.all(np.diff(means) > 0), means


def _ge(data_dir, name):
    return _file(data_dir, "PET/GE-DMI-NEMA", "dmi_nema_lm", name)


def test_ge_tof_sign_images_the_spheres(data_dir):
    """GE events placed at their TOF positions: the NEMA spheres show as spots with the reader's sign (the peak over the
    median of a slab through the spheres), and smear into rings with the opposite sign."""
    from scipy.ndimage import gaussian_filter
    events, _, proj_meta = PET.read_ge_hdf5(_ge(data_dir, "LIST0000.BLF"))
    events = events[:20_000_000]
    lut = proj_meta.scanner_lut.double()
    tof = proj_meta.tof_meta
    width = float(tof.bin_width)
    generator = torch.Generator().manual_seed(0)
    edges = np.arange(-200, 202, 3.0)
    slabs = np.arange(-90, 91, 20.0)
    images = {s: np.zeros((len(slabs), len(edges) - 1, len(edges) - 1)) for s in (1, -1)}
    for start in range(0, events.shape[0], 4_000_000):
        ev = events[start:start + 4_000_000].long()
        a, c = lut[ev[:, 0]], lut[ev[:, 1]]
        u = (c - a) / torch.linalg.norm(c - a, dim=1, keepdim=True)
        t = tof.bin_positions.double()[ev[:, 2]] + (torch.rand(ev.shape[0], generator=generator, dtype=torch.float64) - 0.5) * width
        for sign in (1, -1):
            p = ((a + c) / 2 + sign * t[:, None] * u).numpy()
            for k, z0 in enumerate(slabs):
                m = np.abs(p[:, 2] - z0) < 10
                images[sign][k] += np.histogram2d(p[m, 0], p[m, 1], bins=[edges, edges])[0]
    contrast = {}
    for sign in (1, -1):
        g = [gaussian_filter(images[sign][k], 1.5) for k in range(len(slabs))]
        contrast[sign] = np.array([x.max() / np.median(x[x > 0.3 * x.max()]) for x in g])
    assert (contrast[1] - contrast[-1]).max() > 0.5, contrast


def test_ge_bed_is_posterior(data_dir):
    """GE's attenuation factors of one direct plane (they include the bed), on the reader's lookup table: their
    attenuation-weighted centre lies posterior (+y) of the activity's centre, where the bed is under a supine
    phantom. Before PyTomography 4 it lay at -y: the images were upside down (#5)."""
    h5py = pytest.importorskip("h5py")
    events, _, proj_meta = PET.read_ge_hdf5(_ge(data_dir, "LIST0000.BLF"))
    lut = proj_meta.scanner_lut.double().numpy()
    c = proj_meta.info["NrCrystalsPerRing"]

    def least_squares_centre(a, b, w=None):
        d = b[:, :2] - a[:, :2]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        n = np.stack([-d[:, 1], d[:, 0]], 1)
        s = (a[:, :2] * n).sum(1)
        w = np.ones(len(s)) if w is None else w
        return np.linalg.solve((n * w[:, None]).T @ n, (n * w[:, None]).T @ s)

    ev = events[:5_000_000].long().numpy()
    activity = least_squares_centre(lut[ev[:, 0]], lut[ev[:, 1]])
    ring = 18
    ids, attenuation = [], []
    with h5py.File(_ge(data_dir, "corrections.h5"), "r") as f:
        pairs, factors = f["all_xtals/xtal_ids"], f["all_xtals/atten"]
        for start in range(0, pairs.shape[0], 8_000_000):
            x = pairs[start:start + 8_000_000]
            m = (x[:, 1] == ring) & (x[:, 3] == ring)
            if m.any():
                ids.append(x[m].astype(np.int64))
                attenuation.append(factors[start:start + 8_000_000][m])
            elif ids:
                break
    ids, attenuation = np.concatenate(ids), np.concatenate(attenuation)
    pair_ids = c * ids[:, [1, 3]] + ids[:, [0, 2]]                 # GE stores (crystal, ring, crystal, ring)
    weight = -np.log(np.clip(attenuation, 1e-6, None))
    attenuating = least_squares_centre(lut[pair_ids[:, 0]], lut[pair_ids[:, 1]], weight / weight.sum())
    assert attenuating[1] - activity[1] > 5, (activity, attenuating)


def _siemens_mu_map(path):
    """The Siemens mMR attenuation map (JSRecon Interfile), centred on the scanner, as (mu in 1/cm, voxel size)."""
    keys = {}
    for line in open(path).read().splitlines():
        if ":=" in line:
            k, v = line.split(":=", 1)
            keys[k.strip().lstrip("!").lower()] = v.strip()
    shape = [int(keys[f"matrix size[{k}]"]) for k in (1, 2, 3)]
    dr = [float(keys[f"scale factor (mm/pixel) [{k}]"]) for k in (1, 2, 3)]
    mu = np.fromfile(str(path)[:-4], dtype="<f4").reshape(shape[::-1]).transpose(2, 1, 0)
    return mu, np.array(dr)


def test_petsird_events_centre_on_the_siemens_attenuation_map(data_dir):
    """The PETSIRD mMR NEMA prompts in the patient's LPS (up '+x', towards the bed '+z', HFS), against the Siemens
    attenuation map centred on the scanner: the activity's x centre (least squares over the LORs) is the phantom's,
    within 2 mm; mirrored in x it would be 58 mm off."""
    from pytomography.io.PET import petsird
    path = _file(data_dir, "PET/PETSIRD-mMR-NEMA", "mMR_NEMA_norm.petsird")
    mu_path = _file(data_dir, "PET/PETSIRD-mMR-NEMA", "20170809_NEMA_MUMAP_UCL.v.hdr")
    header = petsird.read_header(path)
    info = petsird._detector_info(header, up="+x", towards_bed="+z", patient_position="HFS")
    prompts = petsird._detector_ids(path, info=info, header=header, max_events=3_000_000).long().numpy()
    del header
    lut = PET.scanner_lut(info).double().numpy()
    a, b = lut[prompts[:, 0]], lut[prompts[:, 1]]
    d = b[:, :2] - a[:, :2]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    activity = np.linalg.solve(n.T @ n, n.T @ (a[:, :2] * n).sum(1))
    mu, dr = _siemens_mu_map(mu_path)
    body = mu > 0.05
    x = (np.arange(mu.shape[0]) - (mu.shape[0] - 1) / 2) * dr[0]
    phantom_x = (x[:, None, None] * body).sum() / body.sum()
    assert abs(activity[0] - phantom_x) < 2, (activity, phantom_x)
    assert abs(phantom_x) > 20                                     # the phantom is off centre: a mirror would show
