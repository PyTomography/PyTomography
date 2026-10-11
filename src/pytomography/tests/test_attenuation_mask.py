"""``io.attenuation_map`` (one HU to mu module, mu map files in any units, the water check) and ``io.load_mask``
(RTSTRUCT, NIfTI label images), on files written here (CPU)."""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pydicom
import pytest
import torch

from pytomography import io
from pytomography.io import Image, _hu
from pytomography.metadata import ObjectMeta

WATER_208 = 0.013528774          # NIST water at 208 keV (1/mm), from PyTomography's fit


def _cylinder_ct(tmp_path, kvp=130, model="Symbia T16"):
    """A CT series: a water cylinder (0 HU, radius 30 mm) in air (-1000 HU), 1 mm voxels, centred at (5, -10, 40)."""
    shape = (80, 80, 40)
    x, y = np.meshgrid((np.arange(80) - 39.5), (np.arange(80) - 39.5), indexing="ij")
    hu = np.where(x**2 + y**2 < 30**2, 0.0, -1000.0)[:, :, None] * np.ones(shape)
    meta = ObjectMeta(dr=(1.0, 1.0, 1.0), shape=shape, centre=(5.0, -10.0, 40.0), frame="patient", units="HU",
                      modality="CT")
    files = io.save_dicom(torch.tensor(hu), tmp_path / "ct", meta)
    for f in files:
        ds = pydicom.dcmread(str(f))
        ds.KVP, ds.ManufacturerModelName = kvp, model
        ds.save_as(str(f))
    return [str(f) for f in files], meta


# ---------- HU to mu ----------

def test_the_table_gives_water_at_the_photon_energy():
    hu = np.array([-1000.0, 0.0, 1000.0])
    mu = _hu.hu_to_mu(hu, 208, kvp=130, model="Symbia T16")
    assert mu.dtype == np.float32 and mu[0] < 1e-4
    assert abs(mu[1] / WATER_208 - 1) < 0.03 and mu[2] > mu[1]
    with pytest.warns(UserWarning, match="not in PyTomography's table"):
        _hu.hu_to_mu(hu, 208, kvp=130, model="Unknown Scanner 9000")


def test_the_nist_curves_are_v3s_numbers():
    """The 'cortical_bone_fit' and 'nist' curves give v3's numbers (io.SPECT.attenuation_map.get_HU2mu_conversion,
    in 1/cm there) for a bone peak at 1400 HU, and for 120 kVp without one."""
    hu = np.array([-1000.0, -500.0, 0.0, 500.0, 1400.0])
    v3_bone = [0.00015092631, 0.06771933485, 0.13528774340, 0.17575445720, 0.24859454205]
    v3_no_peak = [0.00015092631, 0.06771933485, 0.13528774340, 0.18320956587, 0.26946884631]
    with_peak = _hu.bilinear(hu, *_hu.nist_coefficients(208, _hu.effective_ct_energy(1400.0)))
    assert np.allclose(with_peak, np.array(v3_bone) / 10, rtol=1e-5)
    nist = _hu.hu_to_mu(hu, 208, technique="nist", kvp=120)
    assert np.allclose(nist, np.array(v3_no_peak) / 10, rtol=1e-5)
    with pytest.warns(UserWarning, match="cortical bone peak"):
        assert np.allclose(_hu.hu_to_mu(hu, 208, technique="cortical_bone_fit", kvp=120), nist)


def test_the_table_has_no_511_kev_so_nist_is_used():
    with pytest.warns(UserWarning, match="no photon energy near 511"):
        mu = _hu.hu_to_mu(np.array([0.0]), 511, kvp=120, model="Symbia T16")
    assert abs(mu[0] / _hu.nist_mu("water", 511) - 1) < 1e-6
    with pytest.raises(ValueError, match="technique"):
        _hu.hu_to_mu(np.array([0.0]), 140, technique="from_table")


# ---------- attenuation_map ----------

def test_attenuation_map_of_a_ct_series(tmp_path):
    files, ct_meta = _cylinder_ct(tmp_path)
    grid = ObjectMeta(dr=(4.0, 4.0, 4.0), shape=(20, 20, 10), centre=ct_meta.centre, frame="patient", modality="NM")
    mu = io.attenuation_map(files, grid, 208)
    assert mu.dtype == torch.float32 and tuple(mu.shape) == (20, 20, 10)
    mu = mu.cpu()
    water = float(_hu.hu_to_mu(np.array([0.0]), 208, kvp=130, model="Symbia T16")[0])
    assert torch.allclose(mu[10, 10], torch.full((10,), water), rtol=1e-5)      # inside the cylinder
    assert float(mu[0, 0, 5]) < 1e-5                                             # air
    edge = mu[:, 10, 5].cpu().numpy()
    assert ((edge > 1e-5) & (edge < water * 0.999)).any()                         # averaged over the cylinder's edge
    same = io.attenuation_map(io.load_image(files), grid, 208).cpu()
    assert torch.equal(mu, same)
    with pytest.raises(ValueError, match="energy_keV"):
        io.attenuation_map(files, grid)


def _siemens_map(tmp_path, mu_per_cm):
    data = np.zeros((12, 10, 8), np.float32)
    data[2:10, 2:8, 1:7] = mu_per_cm
    lines = ['%SMS-MI header name space:=image subheader', 'image data byte order:=LITTLEENDIAN',
             'number format:=float', '!number of bytes per pixel:=4', 'matrix size[1]:=12', 'matrix size[2]:=10',
             'matrix size[3]:=8', 'scale factor (mm/pixel) [1]:=2.0', 'scale factor (mm/pixel) [2]:=2.0',
             'scale factor (mm/pixel) [3]:=3.0', 'quantification units:=1/cm', '!name of data file := mu.raw',
             '!END OF INTERFILE :=']
    (tmp_path / "mu.raw").write_bytes(np.ascontiguousarray(data.transpose(2, 1, 0)).tobytes())
    (tmp_path / "mu.v.hdr").write_text("!INTERFILE :=\n" + "\n".join(lines))
    return tmp_path / "mu.v.hdr"


def test_a_mu_map_in_1_per_cm_on_a_pet_grid(tmp_path):
    """A mu map's units are read and converted to 1/mm; a PET grid takes 511 keV, and water there passes the check."""
    path = _siemens_map(tmp_path, float(_hu.nist_mu("water", 511)) * 10)
    grid = ObjectMeta(dr=(2.0, 2.0, 3.0), shape=(12, 10, 8), modality="PT")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mu = io.attenuation_map(path, grid)
    assert torch.allclose(mu.max(), torch.tensor(float(_hu.nist_mu("water", 511)), device=mu.device), rtol=1e-5)
    with pytest.warns(UserWarning, match="almost no voxels"):          # ten times water: the units are wrong
        assert torch.allclose(io.attenuation_map(path, grid, units="1/mm").max(), mu.max() * 10, rtol=1e-6)


def test_the_water_check_catches_a_map_made_for_another_energy(tmp_path):
    """#2: a map of water at 208 keV used for 113 keV (or in the wrong units) warns."""
    path = _siemens_map(tmp_path, WATER_208 * 10)
    grid = ObjectMeta(dr=(2.0, 2.0, 3.0), shape=(12, 10, 8), modality="NM")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        io.attenuation_map(path, grid, 208)
    with pytest.warns(UserWarning, match="% from NIST"):
        io.attenuation_map(path, grid, 113)
    with pytest.warns(UserWarning, match="almost no voxels"):
        io.attenuation_map(path, grid, 208, units="1/m")
    with pytest.raises(ValueError, match="units of linear attenuation"):
        io.attenuation_map(path, grid, 208, units="Bq/mL")


def test_a_map_without_units_needs_them(tmp_path):
    image = Image(np.ones((4, 4, 4)), np.eye(4), frame="scanner")
    grid = ObjectMeta(dr=(1.0, 1.0, 1.0), shape=(4, 4, 4), modality="PT")
    with pytest.raises(ValueError, match="units="):
        io.attenuation_map(image, grid)
    ct = Image(np.zeros((4, 4, 4)), np.eye(4), modality="CT", frame="scanner")
    with pytest.raises(ValueError, match="units='HU'"):
        io.attenuation_map(ct, grid)


def test_gate_image_is_placed_by_the_world_axes(tmp_path):
    """A map in GATE's world goes to the patient frame through read_gate's world_axes and world_origin: with 'LPI', GATE's
    +z is the patient's inferior."""
    data = np.zeros((12, 10, 8), np.float32)
    data[3, 4, 6] = 1.0                                   # GATE (x, y, z) index (3, 4, 6) of a 2 x 2 x 3 mm grid
    image = Image(data, ObjectMeta(dr=(2.0, 2.0, 3.0), shape=data.shape).affine, units="1/cm", frame="scanner")
    proj_meta = SimpleNamespace(world_axes="LPI", world_origin=(10.0, -5.0, 20.0))
    placed = io.PET.gate_image(image, proj_meta)
    gate_position = (np.array([3, 4, 6]) - (np.array(data.shape) - 1) / 2) * np.array([2.0, 2.0, 3.0])
    expected = gate_position * np.array([1, 1, -1]) + np.array([10.0, -5.0, 20.0])
    assert np.allclose(placed.affine @ np.array([3, 4, 6, 1.0]), [*expected, 1])
    assert placed.frame == "patient" and placed.units == "1/cm"
    grid = ObjectMeta(dr=(2.0, 2.0, 3.0), shape=data.shape, centre=(10.0, -5.0, 20.0), frame="patient", modality="PT")
    on = placed.on(grid, mode="nearest").cpu().numpy()
    assert on[3, 4, 8 - 1 - 6] == 1.0                     # z reversed on the grid


# ---------- load_mask ----------

def test_load_mask_of_a_nifti_label_image(tmp_path):
    nib = pytest.importorskip("nibabel")
    labels = np.zeros((10, 12, 6), np.int16)
    labels[2:5, 3:6, 1:3] = 1
    labels[6:9, 3:6, 2:5] = 2
    affine_ras = np.diag([-2.0, -2.0, 2.0, 1.0])             # LPS axes, 2 mm
    nib.save(nib.Nifti1Image(labels, affine_ras), str(tmp_path / "labels.nii.gz"))
    mask = io.load_mask(tmp_path / "labels.nii.gz")
    assert mask.units == "label" and mask.labels == {"1": 1, "2": 2}
    grid = mask.object_meta
    assert torch.equal(mask.on(grid, label=2, mode="nearest").cpu().bool(), torch.from_numpy(labels == 2))
    assert torch.equal(mask.on(grid, label="1", threshold=0.5).cpu(), torch.from_numpy(labels == 1))
    with pytest.raises(ValueError, match="integer labels"):
        io.load_mask(Image(np.full((3, 3, 3), 0.5), np.eye(4)))


def test_load_mask_of_an_rtstruct_with_overlapping_rois(tmp_path):
    """An RTSTRUCT drawn on a CT: each ROI comes back on the CT's own voxels, and two ROIs that overlap both keep the
    voxels they share."""
    rt_utils = pytest.importorskip("rt_utils")
    files, ct_meta = _cylinder_ct(tmp_path)
    builder = rt_utils.RTStructBuilder.create_new(dicom_series_path=str(tmp_path / "ct"))
    rows, cols, n = 80, 80, 40                               # rt_utils' masks are (rows = y, columns = x, slices)
    a = np.zeros((rows, cols, n), bool)
    b = np.zeros((rows, cols, n), bool)
    a[20:50, 10:40, 5:25] = True
    b[40:70, 30:60, 15:35] = True
    builder.add_roi(mask=a, name="a")
    builder.add_roi(mask=b, name="b")
    builder.save(str(tmp_path / "rt.dcm"))
    mask = io.load_mask(tmp_path / "rt.dcm", reference=files)
    assert mask.labels["a"][0] == 1 and mask.labels["b"][0] == 2
    assert set(mask.labels["a"]) & set(mask.labels["b"])     # the shared voxels' value is in both
    for name, drawn in (("a", a), ("b", b)):
        got = mask.on(ct_meta, label=name, mode="nearest").cpu().numpy() > 0.5
        truth = drawn.transpose(1, 0, 2)                    # (x, y, z)
        dice = 2 * (got & truth).sum() / (got.sum() + truth.sum())
        assert dice > 0.95, (name, dice)
    only_a = io.load_mask(tmp_path / "rt.dcm", reference=str(tmp_path / "ct"), rois=["a"])
    assert only_a.labels == {"a": 1}
    with pytest.raises(ValueError, match="reference"):
        io.load_mask(tmp_path / "rt.dcm")


@pytest.mark.data
def test_masks_and_mu_from_one_ct_agree(data_dir):
    """Lu177-PSMA-GEDisc: the RTSTRUCT's organs drawn on their CT and the CT's mu at 208 keV, both put on a 4.4 mm grid
    around the CT: the lungs hold mu < 0.008/mm and the liver > 0.012/mm (a mask flipped or shifted against the CT would
    mix them)."""
    folder = data_dir / "SPECT" / "Lu177-PSMA-GEDisc"
    if not (folder / "segmentations.dcm").exists():
        pytest.skip("Lu177-PSMA-GEDisc is not in the data folder")
    files = sorted(str(f) for f in (folder / "CT").glob("*.dcm"))
    ct = io.load_image(files)
    lungs = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right",
             "lung_lower_lobe_right"]
    masks = io.load_mask(folder / "segmentations.dcm", reference=files, rois=lungs + ["liver"])
    extent = np.array(ct.shape) * np.array(ct.voxel_size)
    grid = ObjectMeta(dr=(4.4, 4.4, 4.4), shape=tuple(int(n) for n in extent // 4.4), centre=ct.centre,
                      frame="patient", modality="NM")
    mu = io.attenuation_map(ct, grid, 208).cpu()
    lung = torch.zeros(grid.shape, dtype=torch.bool)
    for name in lungs:
        lung |= masks.on(grid, label=name, threshold=0.9).cpu()
    liver = masks.on(grid, label="liver", threshold=0.9).cpu()
    assert lung.sum() > 1000 and liver.sum() > 1000
    assert float(mu[lung].mean()) < 0.008 and float(mu[liver].mean()) > 0.012, (float(mu[lung].mean()), float(mu[liver].mean()))
