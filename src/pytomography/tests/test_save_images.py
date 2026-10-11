"""Saving images as NIfTI and DICOM (pytomography.io.shared.output): every voxel's value and place in the patient
survives a round trip, read back with nibabel, pydicom, and PyTomography's own DICOM reader."""
import numpy as np
import pydicom
import pytest
import torch

from pytomography.io import Image, load_image, save_dicom, save_nifti
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
        det = ds.DetectorInformationSequence[0] if "DetectorInformationSequence" in ds else None   # NM
        iop = det.ImageOrientationPatient if det is not None else ds.ImageOrientationPatient
        row, col = np.array(iop[:3], float), np.array(iop[3:], float)
        dr, dc = (float(v) for v in ds.PixelSpacing)        # between rows, between columns
        if "RealWorldValueMappingSequence" in ds:                # NM
            m = ds.RealWorldValueMappingSequence[0]
            slope, intercept = float(m.RealWorldValueSlope), float(m.RealWorldValueIntercept)
        else:
            slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
        frames = ds.pixel_array.reshape(-1, ds.Rows, ds.Columns) * slope + intercept
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


def test_object_meta_places_the_image(tmp_path):
    """A tensor saved with its object_meta lands where object_meta.affine says; SPECT voxel sizes (cm) become mm."""
    nib = pytest.importorskip("nibabel")
    meta = ObjectMeta(dr=(2.0, 2.0, 3.0), shape=(12, 10, 7), centre=(5.0, -20.0, 300.0), frame='patient',
                      units='Bq/mL', modality='PT')
    path = save_nifti(torch.from_numpy(_image()), tmp_path / "pet", meta)
    img = nib.load(str(path))
    assert np.allclose(img.affine, np.diag([-1, -1, 1, 1]) @ meta.affine)
    assert int(img.header["sform_code"]) == 1 and b"Bq/mL" in bytes(img.header["descrip"])
    files = save_dicom(_image(), tmp_path / "pt", meta)
    _check_dicom(files, _image(), meta.affine, tol=float(_image().max()) / 65535 * 0.51 + 1e-6)
    assert pydicom.dcmread(str(files[0])).Units == "BQML"
    S = SPECTObjectMeta(dr=(0.48, 0.48, 0.48), shape=(12, 10, 7))
    assert np.isclose(S.affine[0, 0], 4.8) and np.allclose(S.affine @ [5.5, 4.5, 3, 1], [0, 0, 0, 1])


@pytest.mark.parametrize("A", AFFINES)
def test_nifti_round_trip(tmp_path, A):
    nib = pytest.importorskip("nibabel")
    x = _image()
    path = save_nifti(Image(torch.from_numpy(x)[None], A), tmp_path / "recon", units="counts")
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
    files = save_dicom(Image(x, A), tmp_path / modality, modality=modality, units="Bq/mL")
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
    files = save_dicom(Image(x, A, modality="CT"), tmp_path / "ct")
    back = load_image([str(f) for f in files])
    assert back.shape == x.shape and np.allclose(back.affine, A, atol=1e-4)
    assert np.allclose(back.data, x) and back.modality == "CT" and back.units == "HU"


def test_reference_joins_the_study(tmp_path):
    ref = pydicom.Dataset()
    ref.PatientName, ref.PatientID, ref.PatientSex = "Doe^Jane", "P-17", "F"
    ref.StudyInstanceUID, ref.FrameOfReferenceUID = pydicom.uid.generate_uid(), pydicom.uid.generate_uid()
    ref.StudyDate, ref.StudyTime, ref.StudyID = "20260101", "101500", "4"
    ref.SeriesInstanceUID = pydicom.uid.generate_uid()
    files = save_dicom(Image(_image(), AFFINES[0]), tmp_path / "pt", modality="PT", reference=ref)
    uids = set()
    for f in files:
        ds = pydicom.dcmread(str(f))
        assert (ds.PatientName, ds.PatientID, ds.StudyInstanceUID, ds.FrameOfReferenceUID) == \
               ("Doe^Jane", "P-17", ref.StudyInstanceUID, ref.FrameOfReferenceUID)
        assert ds.SeriesInstanceUID != ref.SeriesInstanceUID
        uids.add(ds.SOPInstanceUID)
    assert len(uids) == len(files)                                            # every slice its own instance


def test_refuses_to_overwrite(tmp_path):
    image = Image(_image(), AFFINES[0], modality="PT")
    save_dicom(image, tmp_path / "s")
    with pytest.raises(FileExistsError):
        save_dicom(image, tmp_path / "s")
    assert len(save_dicom(image, tmp_path / "s", overwrite=True)) == 7


def test_sheared_affine_is_refused(tmp_path):
    A = AFFINES[0].copy()
    A[0, 2] = 1.0                                                             # a tilted slice axis
    with pytest.raises(ValueError, match="sheared"):
        save_dicom(Image(_image(), A), tmp_path / "s", modality="PT")


@pytest.mark.parametrize("signed", [False, True])
def test_nm_scale_uses_16_bits_and_is_in_the_headers(tmp_path, signed):
    """SPECT (NM): the stored integers use the full 16-bit range, and the slope is both in the Real World Value
    Mapping (with the units) and in Rescale Slope/Intercept, so readers using either get the values within half a step."""
    x = _image(signed=signed) * 1000
    [path] = save_dicom(Image(x, AFFINES[0]), tmp_path / "nm", modality="NM", units="Bq/mL")
    ds = pydicom.dcmread(str(path))
    m = ds.RealWorldValueMappingSequence[0]
    stored = ds.pixel_array
    assert int(np.abs(stored).max()) == (32767 if signed else 65535)
    assert m.MeasurementUnitsCodeSequence[0].CodeValue == "Bq/mL"
    assert int(m.RealWorldValueFirstValueMapped) == min(0, int(stored.min()))   # negative stored values are mapped too
    assert int(m.RealWorldValueLastValueMapped) == int(stored.max())
    step = float(m.RealWorldValueSlope)
    assert float(ds.RescaleSlope) == pytest.approx(step, rel=1e-9) and float(ds.RescaleIntercept) == 0
    for slope, intercept in [(step, float(m.RealWorldValueIntercept)), (float(ds.RescaleSlope), 0.0)]:
        real = np.transpose(stored * slope + intercept, (2, 1, 0))           # frames (k, j, i) to axes (i, j, k)
        assert np.abs(real - x).max() <= 0.5001 * step


def test_nm_is_the_default_for_spect(tmp_path):
    """SPECT images are saved as NM unless PET-style slices are asked for; NM without a reference has the required
    (empty) radiopharmaceutical sequence."""
    meta = SPECTObjectMeta(dr=(0.48, 0.48, 0.48), shape=(12, 10, 7))
    [path] = save_dicom(_image(), tmp_path / "spect", meta)
    ds = pydicom.dcmread(str(path))
    assert ds.Modality == "NM" and "RadiopharmaceuticalInformationSequence" in ds
    files = save_dicom(_image(), tmp_path / "spect_slices", meta, modality="PT")
    assert len(files) == 7 and pydicom.dcmread(str(files[0])).Modality == "PT"
    files = save_dicom(_image(), tmp_path / "pet", ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(12, 10, 7), modality="PT"))
    assert len(files) == 7 and pydicom.dcmread(str(files[0])).Modality == "PT"


def test_the_modality_comes_from_object_meta(tmp_path):
    """#41: a CT object_meta saves as CT without being told; an object_meta with no modality must be told one."""
    meta = ObjectMeta(dr=(1.0, 1.0, 2.0), shape=(12, 10, 7), centre=(0, 0, -50), frame="patient", units="HU",
                      modality="CT")
    files = save_dicom(np.round(_image(signed=True) * 10), tmp_path / "ct", meta)
    ds = pydicom.dcmread(str(files[0]))
    assert ds.Modality == "CT" and ds.RescaleType == "HU"
    with pytest.raises(ValueError, match="modality"):
        save_dicom(_image(), tmp_path / "unknown", ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(12, 10, 7)))


def test_slice_thickness_is_not_the_spacing(tmp_path):
    """#41: SliceThickness is written only when given; the spacing is in the slices' positions."""
    meta = ObjectMeta(dr=(1.0, 1.0, 1.0), shape=(12, 10, 7), modality="CT", units="HU")
    x = np.round(_image(signed=True) * 10)
    ds = pydicom.dcmread(str(save_dicom(x, tmp_path / "a", meta)[0]))
    assert ds.SliceThickness in ("", None)
    files = save_dicom(x, tmp_path / "b", meta, slice_thickness=1.25)
    z = [float(pydicom.dcmread(str(f)).ImagePositionPatient[2]) for f in files]
    assert float(pydicom.dcmread(str(files[0])).SliceThickness) == 1.25 and np.allclose(np.diff(z), 1.0)


def test_subsampled_grid_saves_where_it_is(tmp_path):
    """#27: after subsample the voxel size and the origin written are the coarse grid's, centred where the fine
    grid was."""
    fine = SPECTObjectMeta(dr=(0.48, 0.48, 0.48), shape=(24, 24, 14), centre=(3.0, -150.0, 400.0), frame="patient")
    coarse = fine.subsample(2)
    assert coarse.shape == (12, 12, 7) and np.allclose(coarse.dr, 0.96)
    [path] = save_dicom(_image((12, 12, 7)), tmp_path / "nm", coarse)
    ds = pydicom.dcmread(str(path))
    assert np.allclose([float(v) for v in ds.PixelSpacing], 9.6) and float(ds.SpacingBetweenSlices) == pytest.approx(9.6)
    ipp = np.array(ds.DetectorInformationSequence[0].ImagePositionPatient, float)
    assert np.allclose(ipp, np.array(fine.centre) - np.array([5.5, 5.5, 3.0]) * 9.6)


def test_scanner_frame_gets_its_own_frame_of_reference(tmp_path):
    """An image placed only relative to the scanner (frame 'scanner') joins the reference's study but not its frame of
    reference; NIfTI marks it as aligned (sform code 2)."""
    nib = pytest.importorskip("nibabel")
    ref = pydicom.Dataset()
    ref.PatientName, ref.PatientID = "Doe^Jane", "P-17"
    ref.StudyInstanceUID, ref.FrameOfReferenceUID = pydicom.uid.generate_uid(), pydicom.uid.generate_uid()
    for frame, same in (("scanner", False), ("patient", True)):
        meta = ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(12, 10, 7), frame=frame, modality="PT")
        ds = pydicom.dcmread(str(save_dicom(_image(), tmp_path / frame, meta, reference=ref)[0]))
        assert ds.StudyInstanceUID == ref.StudyInstanceUID
        assert (ds.FrameOfReferenceUID == ref.FrameOfReferenceUID) == same
        code = int(nib.load(str(save_nifti(_image(), tmp_path / f"{frame}.nii", meta))).header["sform_code"])
        assert code == (1 if same else 2)


def test_an_image_and_its_grid(tmp_path):
    """An Image's object_meta is its own grid; an Image carries its own affine, so the writers refuse an object_meta
    with it, and they refuse a tensor whose shape is not object_meta's."""
    meta = ObjectMeta(dr=(2.0, 3.0, 4.0), shape=(12, 10, 7), centre=(1, 2, 3), frame="patient", modality="PT")
    image = Image(_image(), meta.affine, modality="PT", units="counts")
    assert np.allclose(image.centre, (1, 2, 3)) and image.voxel_size == (2.0, 3.0, 4.0)
    assert np.allclose(image.object_meta.affine, meta.affine) and image.object_meta.modality == "PT"
    with pytest.raises(ValueError, match="LPS-aligned"):
        Image(_image(), AFFINES[1]).object_meta
    with pytest.raises(TypeError):
        save_dicom(image, tmp_path / "a", meta)
    with pytest.raises(ValueError, match="shape"):
        save_dicom(_image((6, 6, 6)), tmp_path / "b", meta)


# ---------- the patient frame of data sources without one in object_meta (tutorial data) ----------

def _tutorial_data(data_dir, dataset, *path):
    """A file of one tutorial dataset; the test is skipped when that dataset isn't in the data folder (CI's data job
    fetches only SPECT/Lu177-PSMA-GEDisc)."""
    target = data_dir / dataset
    for part in path:
        target = target / part
    if not target.exists():
        pytest.skip(f"{dataset} is not in the tutorial data folder")
    return target


@pytest.mark.data
def test_starguide_frame_matches_ge_reconstruction(data_dir):
    """PyTomography's StarGuide grid, placed through the CT, is where GE's own reconstruction of the same grid is."""
    import os
    from pytomography.io.SPECT import dicom
    root = _tutorial_data(data_dir, "SPECT/Tc99m-NEMA-Starguide")
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
    path = str(_tutorial_data(data_dir, "PET/GATE-mMR-Brain", "fdg_pet_phantom_mri.nii.gz"))
    meta = ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(100, 120, 110))
    on_grid = gate.get_attenuation_map_nifti(path, meta).cpu().numpy() * 10   # the MR as the PET grid sees it
    saved = nib.load(str(save_nifti(Image(on_grid, gate.get_patient_affine_from_nifti(path, meta)),
                                    tmp_path / "mr_on_pet_grid.nii.gz")))
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
    file_NM = str(_tutorial_data(data_dir, "SPECT/Lu177-PSMA-GEDisc", "bed1_projections.dcm"))
    object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=0)
    x = torch.rand(object_meta.shape) * 100
    before = x.clone()
    dicom.save_dcm(str(tmp_path / "n"), x, file_NM, scale_by_number_projections=True)
    assert torch.equal(x, before)                                   # the caller's image is untouched
    files = sorted(str(f) for f in (tmp_path / "n").glob("*.dcm"))
    back = torch.from_numpy(load_image(files).data)
    assert torch.allclose(back, x, atol=0.51 / proj_meta.num_projections)
    with pytest.raises(ValueError, match="16 bits"):
        dicom.save_dcm(str(tmp_path / "big"), x * 1000, file_NM, scale_by_number_projections=True)


@pytest.mark.data
def test_projection_reader_applies_rescale(tmp_path, data_dir):
    """#232: projections stored with a rescale slope and intercept are read as their real values."""
    from pytomography.io.SPECT import dicom
    file_NM = str(_tutorial_data(data_dir, "SPECT/Lu177-PSMA-GEDisc", "bed1_projections.dcm"))
    ds = pydicom.dcmread(file_NM)
    ds.RescaleSlope, ds.RescaleIntercept = 1, 0                    # the stored values, whatever the file says
    plain = dicom.parse_projection_dataset(ds)[0]
    ds.RescaleSlope, ds.RescaleIntercept = 2.5, 3.0
    scaled = dicom.parse_projection_dataset(ds)[0]
    assert torch.allclose(scaled, plain * 2.5 + 3.0)


def test_required_attributes_without_a_reference(tmp_path):
    """Without a reference (simulated PET, a micro-CT sample), PET has the required, empty Radiopharmaceutical
    Information Sequence and CT the required, empty Patient Position, as the standard's validator asks."""
    pt = pydicom.dcmread(str(save_dicom(Image(_image(), AFFINES[0]), tmp_path / "pt", modality="PT")[0]))
    assert "RadiopharmaceuticalInformationSequence" in pt and len(pt.RadiopharmaceuticalInformationSequence) == 0
    ct = pydicom.dcmread(str(save_dicom(Image(_image(), AFFINES[0]), tmp_path / "ct", modality="CT", units="1/mm")[0]))
    assert "PatientPosition" in ct and ct.PatientPosition == ""


def test_reference_values_dicom_does_not_allow_are_left_empty(tmp_path):
    """An anonymised reference with "0" for the birth date, sex and age: the series gets them empty (the reference is
    untouched), and the retired Other Patient IDs and Ethnic Group are not copied."""
    ref = pydicom.Dataset()
    ref.PatientName, ref.PatientID = "Anon", "0"
    ref.PatientBirthDate, ref.PatientSex, ref.PatientAge = "0", "0", "0"
    ref.OtherPatientIDs, ref.EthnicGroup = "X-1", "unknown"
    ref.StudyInstanceUID, ref.FrameOfReferenceUID = pydicom.uid.generate_uid(), pydicom.uid.generate_uid()
    [path] = save_dicom(Image(_image(), AFFINES[0]), tmp_path / "nm", modality="NM", reference=ref)
    ds = pydicom.dcmread(str(path))
    assert (ds.PatientBirthDate, ds.PatientSex, ds.PatientAge) == ("", "", "")
    assert "OtherPatientIDs" not in ds and "EthnicGroup" not in ds
    assert (ds.PatientName, ds.StudyInstanceUID) == ("Anon", ref.StudyInstanceUID)
    assert ref.PatientBirthDate == "0"
    ref.PatientBirthDate, ref.PatientSex, ref.PatientAge = "19700101", "F", "055Y"
    [path] = save_dicom(Image(_image(), AFFINES[0]), tmp_path / "nm", modality="NM", reference=ref, overwrite=True)
    ds = pydicom.dcmread(str(path))
    assert (ds.PatientBirthDate, ds.PatientSex, ds.PatientAge) == ("19700101", "F", "055Y")


def test_nm_units_in_megabecquerel_seconds(tmp_path):
    [path] = save_dicom(Image(_image(), AFFINES[0]), tmp_path / "nm", modality="NM", units="MBq.s")
    assert pydicom.dcmread(str(path)).RealWorldValueMappingSequence[0].MeasurementUnitsCodeSequence[0].CodeValue == "MBq.s"


def test_ct_in_other_units_keeps_fractions(tmp_path):
    """CT in HU is rounded to whole HU; attenuation per mm (micro-CT) keeps its fractional values."""
    mu = _image() / 2000                                                        # about 0 to 0.2 per mm
    files = save_dicom(Image(mu, AFFINES[0]), tmp_path / "mu", modality="CT", units="1/mm")
    _check_dicom(files, mu, AFFINES[0], tol=float(mu.max()) / 65535 * 0.51 + 1e-9)
    assert pydicom.dcmread(str(files[0])).RescaleType == "US"


def test_ct_keeps_whole_hu_and_clips_wild_voxels(tmp_path):
    """CT in HU stays in 1 HU steps even when a few voxels are far outside the 16-bit range; those are clipped."""
    x = np.round(_image(signed=True) * 10)
    x[0, 0, 0] = 6e5                                                            # a wild voxel at the edge of coverage
    with pytest.warns(UserWarning, match="1 voxels"):
        files = save_dicom(Image(x, AFFINES[0]), tmp_path / "ct", modality="CT")
    clipped = np.clip(x, -32768, 32767)
    _check_dicom(files, clipped, AFFINES[0], tol=0)
