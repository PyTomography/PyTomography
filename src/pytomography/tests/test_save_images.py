"""Saving images as NIfTI and DICOM (pytomography.io.shared.output): every voxel's value and place in the patient
survives a round trip, read back with nibabel, pydicom, and PyTomography's own DICOM reader."""
import numpy as np
import pydicom
import pytest
import torch

from pytomography.io.shared import open_multifile, _get_affine_multifile
from pytomography.io.shared.output import centred_affine, patient_affine, save_dicom, save_nifti
from pytomography.metadata import ObjectMeta
from pytomography.metadata.SPECT import SPECTObjectMeta


def _image(shape=(12, 10, 7), signed=False, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random(shape).astype(np.float32) * 50
    x[3, 4, 5] = 400.0                              # a marker voxel whose place we follow
    return x - 100 if signed else x


# affines with flips, unequal voxel sizes, and k running down as well as up
AFFINES = [
    np.array([[2.0, 0, 0, -11], [0, 3.0, 0, 20.5], [0, 0, 4.0, -300], [0, 0, 0, 1]]),
    np.array([[-1.5, 0, 0, 40], [0, 2.0, 0, -7], [0, 0, -5.0, 120], [0, 0, 0, 1]]),
    np.array([[0, 2.0, 0, 5], [2.5, 0, 0, -9], [0, 0, 3.0, 0], [0, 0, 0, 1]]),   # axes swapped (still orthogonal)
]


def _positions_from_dicom(files):
    """Each slice's pixel data, read back, keyed by its voxels' patient coordinates."""
    out = []
    for f in files:
        ds = pydicom.dcmread(str(f))
        row, col = np.array(ds.ImageOrientationPatient[:3], float), np.array(ds.ImageOrientationPatient[3:], float)
        dr, dc = (float(v) for v in ds.PixelSpacing)        # between rows, between columns
        frames = ds.pixel_array.reshape(-1, ds.Rows, ds.Columns) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        if "NumberOfFrames" in ds and int(ds.NumberOfFrames) > 1:
            ipp0 = np.array(ds.DetectorInformationSequence[0].ImagePositionPatient, float)
            normal = np.cross(row, col)
            ipps = [ipp0 + n * float(ds.SpacingBetweenSlices) * normal for n in range(len(frames))]
        else:
            ipps = [np.array(ds.ImagePositionPatient, float)]
        for frame, ipp in zip(frames, ipps):
            out.append((frame, ipp, row * dc, col * dr))
    return out


def _check_dicom(files, x, A, tol):
    for frame, ipp, step_c, step_r in _positions_from_dicom(files):
        # which slice of the original is at this position?
        ijk = np.linalg.solve(A[:3, :3], ipp - A[:3, 3])
        assert np.allclose(ijk[:2], 0, atol=1e-4), ijk
        k = int(round(ijk[2]))
        assert abs(ijk[2] - k) < 1e-4
        # pixel (r, c) is at ipp + c * step_c + r * step_r: voxel (i, j) = (c, r)
        assert np.allclose(step_c, A[:3, 0], atol=1e-5) and np.allclose(step_r, A[:3, 1], atol=1e-5)
        assert np.allclose(frame, x[:, :, k].T, atol=tol)


def test_centred_affine_and_spect_units():
    A = patient_affine(ObjectMeta(dr=(2.0, 2.0, 3.0), shape=(5, 5, 4)))
    assert np.allclose(A @ [2, 2, 1.5, 1], [0, 0, 0, 1])                     # the centre of the object is at 0
    S = patient_affine(SPECTObjectMeta(dr=(0.48, 0.48, 0.48), shape=(128, 128, 96)))
    assert np.isclose(S[0, 0], 4.8)                                          # cm to mm
    meta = SPECTObjectMeta(dr=(0.48, 0.48, 0.48), shape=(4, 4, 4))
    meta.affine_matrix = AFFINES[0]
    assert np.allclose(patient_affine(meta), AFFINES[0])                     # a reader's frame wins
    assert np.allclose(patient_affine(meta, affine=AFFINES[1]), AFFINES[1])  # and an explicit one wins over that
    assert np.allclose(centred_affine((1, 1, 1), (3, 3, 3)) @ [1, 1, 1, 1], [0, 0, 0, 1])


@pytest.mark.parametrize("A", AFFINES)
def test_nifti_round_trip(tmp_path, A):
    nib = pytest.importorskip("nibabel")
    x = _image()
    path = save_nifti(torch.from_numpy(x)[None], tmp_path / "recon", affine=A, units="counts")
    assert path.name == "recon.nii.gz"
    img = nib.load(str(path))
    assert np.array_equal(np.asarray(img.dataobj), x)
    assert np.allclose(img.affine, np.diag([-1, -1, 1, 1]) @ A)              # NIfTI is RAS
    assert int(img.header["sform_code"]) == 1 and b"counts" in bytes(img.header["descrip"])
    # the marker voxel is at the same point in RAS
    assert np.allclose(img.affine @ [3, 4, 5, 1], np.diag([-1, -1, 1, 1]) @ A @ [3, 4, 5, 1])


def test_nifti_centred_default_is_marked_aligned(tmp_path):
    nib = pytest.importorskip("nibabel")
    path = save_nifti(_image(), tmp_path / "a.nii", object_meta=ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(12, 10, 7)))
    assert int(nib.load(str(path)).header["sform_code"]) == 2


@pytest.mark.parametrize("modality", ["PT", "CT", "NM"])
@pytest.mark.parametrize("A", AFFINES)
def test_dicom_round_trip(tmp_path, modality, A):
    x = _image(signed=(modality == "CT"))
    if modality == "CT":
        x = np.round(x * 10)                                                  # HU are whole numbers
    files = save_dicom(x, tmp_path / modality, affine=A, modality=modality, units="Bq/mL")
    assert len(files) == (1 if modality == "NM" else x.shape[2])
    tol = 0 if modality == "CT" else float(np.abs(x).max()) / 65535 * 0.51 + 1e-6
    _check_dicom(files, x, A, tol)
    ds = pydicom.dcmread(str(files[0]))
    assert ds.Modality == modality and ds.SOPClassUID == ds.file_meta.MediaStorageSOPClassUID
    if modality == "PT":
        assert ds.Units == "BQML"


def test_pytomography_reads_its_own_series(tmp_path):
    """PyTomography's DICOM reader puts the saved CT back on the same grid, voxel for voxel."""
    A = np.array([[0.8, 0, 0, -50], [0, 0.8, 0, -60], [0, 0, 2.5, -400], [0, 0, 0, 1]])
    x = np.round(_image(signed=True) * 10)
    files = save_dicom(x, tmp_path / "ct", affine=A, modality="CT")
    back = open_multifile([str(f) for f in files]).cpu().numpy()
    A_back = _get_affine_multifile([str(f) for f in files])
    assert back.shape == x.shape and np.allclose(A_back, A, atol=1e-4)
    assert np.allclose(back, x)


def test_reference_joins_the_study(tmp_path):
    ref = pydicom.Dataset()
    ref.PatientName, ref.PatientID, ref.PatientSex = "Doe^Jane", "P-17", "F"
    ref.StudyInstanceUID, ref.FrameOfReferenceUID = pydicom.uid.generate_uid(), pydicom.uid.generate_uid()
    ref.StudyDate, ref.StudyTime, ref.StudyID = "20260101", "101500", "4"
    ref.SeriesInstanceUID = pydicom.uid.generate_uid()
    files = save_dicom(_image(), tmp_path / "pt", affine=AFFINES[0], modality="PT", reference=ref)
    uids = set()
    for f in files:
        ds = pydicom.dcmread(str(f))
        assert (ds.PatientName, ds.PatientID, ds.StudyInstanceUID, ds.FrameOfReferenceUID) == \
               ("Doe^Jane", "P-17", ref.StudyInstanceUID, ref.FrameOfReferenceUID)
        assert ds.SeriesInstanceUID != ref.SeriesInstanceUID
        uids.add(ds.SOPInstanceUID)
    assert len(uids) == len(files)                                            # every slice its own instance


def test_refuses_to_overwrite(tmp_path):
    save_dicom(_image(), tmp_path / "s", affine=AFFINES[0])
    with pytest.raises(FileExistsError):
        save_dicom(_image(), tmp_path / "s", affine=AFFINES[0])
    assert len(save_dicom(_image(), tmp_path / "s", affine=AFFINES[0], overwrite=True)) == 7


def test_sheared_affine_is_refused(tmp_path):
    A = AFFINES[0].copy()
    A[0, 2] = 1.0                                                             # a tilted slice axis
    with pytest.raises(ValueError, match="sheared"):
        save_dicom(_image(), tmp_path / "s", affine=A)


# ---------- the patient frame of data sources without one in object_meta (tutorial data) ----------

@pytest.mark.data
def test_starguide_frame_matches_ge_reconstruction(data_dir):
    """PyTomography's StarGuide grid, placed through the CT, is where GE's own reconstruction of the same grid is."""
    import os
    from pytomography.io.SPECT import dicom
    root = data_dir / "SPECT" / "Tc99m-NEMA-Starguide"
    files_CT = [str(root / "CT_files" / f) for f in os.listdir(root / "CT_files")]
    ge = pydicom.dcmread(str(root / "vendor_recon" / "i196884.NMDC.1"), stop_before_pixels=True)
    meta = SPECTObjectMeta(dr=(float(ge.PixelSpacing[0]) / 10,) * 3, shape=(196, 196, 112))   # get_starguide_metadata's grid
    A = dicom.get_starguide_patient_affine(files_CT, meta)
    det = ge.DetectorInformationSequence[0]
    row, col = np.array(det.ImageOrientationPatient[:3], float), np.array(det.ImageOrientationPatient[3:], float)
    A_ge = np.eye(4)
    A_ge[:3, 0] = row * float(ge.PixelSpacing[1])
    A_ge[:3, 1] = col * float(ge.PixelSpacing[0])
    A_ge[:3, 2] = np.cross(row, col) * float(ge.SpacingBetweenSlices)
    A_ge[:3, 3] = np.array(det.ImagePositionPatient, float)
    corners = np.array([[i, j, k, 1] for i in (0, 195) for j in (0, 195) for k in (0, 111)], float).T
    assert np.abs(A @ corners - A_ge @ corners).max() < 0.01


@pytest.mark.data
def test_gate_image_lands_on_its_phantom_nifti(tmp_path, data_dir):
    """A reconstruction-sized image saved with the phantom's frame overlays the phantom's own NIfTI: sampling the
    original MR at each saved voxel's position (through the two files' affines only) gives the saved value."""
    nib = pytest.importorskip("nibabel")
    from scipy.ndimage import map_coordinates
    from pytomography.io.PET import gate
    path = str(data_dir / "PET" / "GATE-mMR-Brain" / "fdg_pet_phantom_mri.nii.gz")
    meta = ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(100, 120, 110))
    on_grid = gate.get_attenuation_map_nifti(path, meta).cpu().numpy() * 10   # the MR as the PET grid sees it
    saved = nib.load(str(save_nifti(on_grid, tmp_path / "mr_on_pet_grid.nii.gz",
                                    affine=gate.get_patient_affine_from_nifti(path, meta))))
    mr = nib.load(path)
    rng = np.random.default_rng(0)
    ijk = np.argwhere(on_grid > 0.2 * on_grid.max())
    ijk = ijk[rng.choice(len(ijk), 2000, replace=False)]
    world = saved.affine @ np.c_[ijk, np.ones(len(ijk))].T
    src = np.linalg.inv(mr.affine) @ world
    sampled = map_coordinates(np.asarray(mr.dataobj, np.float32), src[:3], order=1)
    assert np.allclose(sampled, on_grid[tuple(ijk.T)], rtol=1e-3, atol=1e-3 * on_grid.max())


@pytest.mark.data
def test_save_dcm_scale_by_number_of_projections_reads_back(tmp_path, data_dir):
    """#230: with scale_by_number_projections the series reads back as the image itself, not N_proj times it."""
    from pytomography.io.SPECT import dicom
    file_NM = str(data_dir / "SPECT" / "Lu177-NEMA-SymT2" / "projection_data.dcm")
    object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=0)
    x = torch.rand(object_meta.shape) * 100
    before = x.clone()
    dicom.save_dcm(str(tmp_path / "n"), x, file_NM, scale_by_number_projections=True)
    assert torch.equal(x, before)                                   # the caller's image is untouched
    files = sorted(str(f) for f in (tmp_path / "n").glob("*.dcm"))
    back = open_multifile(files).cpu()
    assert torch.allclose(back, x, atol=0.51 / proj_meta.num_projections)
    with pytest.raises(ValueError, match="16 bits"):
        dicom.save_dcm(str(tmp_path / "big"), x * 1000, file_NM, scale_by_number_projections=True)


@pytest.mark.data
def test_projection_reader_applies_rescale(tmp_path, data_dir):
    """#232: projections stored with a rescale slope and intercept are read as their real values."""
    from pytomography.io.SPECT import dicom
    file_NM = str(data_dir / "SPECT" / "Lu177-NEMA-SymT2" / "projection_data.dcm")
    plain = dicom.parse_projection_dataset(pydicom.dcmread(file_NM))[0]
    ds = pydicom.dcmread(file_NM)
    ds.RescaleSlope, ds.RescaleIntercept = 2.5, 3.0
    scaled = dicom.parse_projection_dataset(ds)[0]
    assert torch.allclose(scaled, plain * 2.5 + 3.0)
