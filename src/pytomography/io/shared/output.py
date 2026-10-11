"""Saving 3D images (reconstructions, attenuation maps, masks...) as NIfTI or DICOM, each voxel in its place in the patient.

Every writer takes the image in one of two forms:

* a torch tensor or numpy array with axes (x, y, z) (a leading axis of length 1 is dropped), and its ``object_meta``.
  The voxels are where ``object_meta.affine`` says (LPS, mm), and ``object_meta.units``, ``object_meta.modality`` and
  ``object_meta.frame`` say what the values are and whether the image is in the patient's frame;
* an :class:`~pytomography.io.Image`, which carries its own affine (any orientation), units, modality and frame.

``save_nifti`` writes one NIfTI-1 file in float32, with the affine (converted to NIfTI's RAS) as its sform and qform:
code 1 (scanner anatomy) for an image in the patient's frame, 2 (aligned) for one placed only relative to the scanner.
``save_dicom`` writes a DICOM series: one file per slice for PET (PT) and CT, or one multi-frame file for SPECT (NM).
Values are stored as 16-bit integers over the full range (one slope, from the image's largest value) and read back
within half a step; CT in HU is stored in whole HU. The slope is in the rescale tags, and for NM also in a Real World
Value Mapping with the units. Given a ``reference`` (any DICOM file of the same acquisition: the projections, a CT slice,
raw CT projections), the series joins that patient and study, and, for an image in the patient's frame, its frame of
reference, so clinical viewers fuse it with the other images of the study.
"""
from __future__ import annotations

import copy
import re
import warnings
import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import ExplicitVRLittleEndian, PYDICOM_IMPLEMENTATION_UID, generate_uid

import pytomography

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
SOP_CLASS = {
    "CT": "1.2.840.10008.5.1.4.1.1.2",      # CT Image Storage
    "PT": "1.2.840.10008.5.1.4.1.1.128",    # Positron Emission Tomography Image Storage
    "NM": "1.2.840.10008.5.1.4.1.1.20",     # Nuclear Medicine Image Storage (multi-frame)
}
# Units of a PET series, (0054,1001)
PET_UNITS = {"counts": "CNTS", "cnts": "CNTS", "bq/ml": "BQML", "bqml": "BQML", "kbq/ml": "BQML", "suv": "GML",
             "gml": "GML", "propcps": "PROPCPS", "a.u.": "CNTS"}
# The image's units as UCUM codes (NM images give their real values through a Real World Value Mapping)
UCUM = {"counts": ("{counts}", "counts"), "cnts": ("{counts}", "counts"), "bq/ml": ("Bq/mL", "becquerels/milliliter"),
        "bqml": ("Bq/mL", "becquerels/milliliter"), "kbq/ml": ("kBq/mL", "kilobecquerels/milliliter"),
        "counts/s": ("{counts}/s", "counts per second"), "mbq/ml": ("MBq/mL", "megabecquerels/milliliter"),
        # activity times time, e.g. a system matrix scaled by a sensitivity in counts per second per MBq
        "mbq.s": ("MBq.s", "megabecquerel seconds"), "mbq·s": ("MBq.s", "megabecquerel seconds")}
# Copied from the reference, when it has them, so the series joins its patient, study and frame of reference (not
# Other Patient IDs or Ethnic Group, which the standard retired in 2017 and 2025)
PATIENT_STUDY = ["PatientName", "PatientID", "PatientBirthDate", "PatientSex", "PatientAge", "PatientSize", "PatientWeight",
                 "IssuerOfPatientID", "StudyInstanceUID", "StudyDate", "StudyTime",
                 "StudyID", "StudyDescription", "AccessionNumber", "ReferringPhysicianName", "FrameOfReferenceUID",
                 "PositionReferenceIndicator", "PatientPosition", "BodyPartExamined"]
# The form these must have; some anonymisers write values DICOM doesn't allow (e.g. "0" for a date), which are copied
# as empty instead
VALID_FORM = {"PatientBirthDate": r"\d{8}", "PatientAge": r"\d{3}[DWMY]", "PatientSex": r"[MFO]"}


# ---------- where the voxels are ----------

def _source(image, object_meta) -> tuple:
    """The image as a 3D numpy array (x, y, z), with its affine (voxel index to LPS mm), units, modality and frame,
    from an :class:`~pytomography.io.Image` or from an array and its ``object_meta``."""
    from .._image import Image
    if isinstance(image, Image):
        if object_meta is not None:
            raise TypeError("an Image carries its own affine: pass it without object_meta")
        return image.data, image.affine, image.units, image.modality, image.frame
    if object_meta is None:
        raise ValueError("give the image's object_meta (its affine, units and modality), or an io.Image")
    arr = _volume(image)
    if tuple(arr.shape) != tuple(object_meta.shape):
        raise ValueError(f"the image's shape {tuple(arr.shape)} is not object_meta's {tuple(object_meta.shape)}")
    return arr, object_meta.affine, object_meta.units, object_meta.modality, object_meta.frame


def centred_affine(dr: Sequence[float], shape: Sequence[int]) -> np.ndarray:
    """The grid centred on the scanner's centre: voxel (i, j, k) at ((i - (Lx - 1) / 2) dx, ...), in mm; the affine of
    ``ObjectMeta(dr, shape)``."""
    d = np.asarray(dr, float)
    A = np.diag([*d, 1.0])
    A[:3, 3] = [-(n - 1) / 2 * dk for n, dk in zip(shape, d)]
    return A


def _numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _volume(image) -> np.ndarray:
    arr = _numpy(image)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D image (x, y, z), got shape {arr.shape}")
    return arr


# ---------- NIfTI ----------

def save_nifti(image, path, object_meta=None, *, units: str | None = None, description: str | None = None):
    """Save a 3D image as a NIfTI-1 file (float32), with its place in the patient.

    Args:
        image (torch.Tensor | np.ndarray | Image): the image, axes (x, y, z), or an :class:`~pytomography.io.Image`.
        path (str | Path): the file to write; ``.nii`` or ``.nii.gz`` (added when the name has neither).
        object_meta (ObjectMeta, optional): the image's grid and where it is (needed for a tensor or an array).
        units (str, optional): the image's units, kept in the header's description (e.g. ``"counts"``, ``"Bq/mL"``).
            Defaults to the units of ``object_meta`` or of the Image.
        description (str, optional): the rest of that description (80 characters in all).

    Returns:
        Path: the file written.
    """
    import nibabel as nib
    arr, A_lps, own_units, _, frame = _source(image, object_meta)
    units = units if units is not None else own_units
    arr = np.asarray(arr, dtype=np.float32)
    A = LPS_TO_RAS @ A_lps
    path = Path(path)
    if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        path = path.with_name(path.name + ".nii.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr, A)
    # 1 = scanner-based anatomical coordinates (the patient's frame); 2 = aligned (placed only relative to the scanner)
    code = 1 if frame == "patient" else 2
    img.set_sform(A, code=code)
    img.set_qform(A, code=code)
    img.header.set_xyzt_units(xyz="mm")
    text = " ".join(t for t in [description or f"PyTomography {pytomography.__version__}", units and f"[{units}]"] if t)
    img.header["descrip"] = text.encode("ascii", "replace")[:79]
    nib.save(img, str(path))
    return path


# ---------- DICOM ----------

def _geometry(A: np.ndarray, shape) -> dict:
    """The DICOM geometry of an image whose voxel (i, j, k) is at A @ (i, j, k, 1): slices along k, rows along j and
    columns along i. Slices must be perpendicular to the slice plane (no gantry tilt)."""
    u, v, w = A[:3, 0], A[:3, 1], A[:3, 2]
    dx, dy, dz = (float(np.linalg.norm(a)) for a in (u, v, w))
    if min(dx, dy, dz) <= 0:
        raise ValueError("the affine has a zero voxel size")
    row, col = u / dx, v / dy
    normal = np.cross(row, col)
    if abs(float(row @ col)) > 1e-4 or abs(abs(float(normal @ (w / dz))) - 1) > 1e-4:
        raise ValueError("DICOM needs the slice axis perpendicular to the slices; this affine is sheared")
    positions = [A @ np.array([0.0, 0.0, k, 1.0]) for k in range(shape[2])]
    return {"dx": dx, "dy": dy, "dz": dz, "row": row, "col": col, "normal": normal,
            "ipp": [p[:3] for p in positions], "ascending": float(normal @ w) > 0}


def _is_hu(modality: str, units: str | None) -> bool:
    """A CT image in Hounsfield units (the default for CT). Other CT images, such as attenuation per mm, keep their
    fractional values."""
    return modality == "CT" and (units is None or units.strip().upper() == "HU")


def _scaling(arr: np.ndarray, modality: str, units: str | None = None):
    """Integers for the pixel data and the rescale slope: whole HU for CT in HU (int16, slope 1); otherwise unsigned
    16 bits for images without negative values and signed 16 bits for the rest, scaled to their maximum."""
    finite = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(finite.min()), float(finite.max())
    if _is_hu(modality, units):
        # 1 HU steps always: a few wild voxels (e.g. at the edge of a reconstruction's coverage) would otherwise force a
        # coarse scale on the whole image, so values beyond 16 bits are clipped, with a warning
        outside = int(np.count_nonzero((finite < -32768) | (finite > 32767)))
        if outside:
            warnings.warn(f"save_dicom: {outside} voxels beyond -32768..32767 HU were clipped to that range")
        return np.clip(np.round(finite), -32768, 32767).astype(np.int16), 1.0, True
    if lo >= 0:
        integral = bool(np.all(finite == np.round(finite))) and hi <= 65535
        slope = 1.0 if integral else (hi / 65535 if hi > 0 else 1.0)
        return np.round(finite / slope).astype(np.uint16), slope, False
    integral = bool(np.all(finite == np.round(finite))) and lo >= -32768 and hi <= 32767
    slope = 1.0 if integral else max(abs(lo), abs(hi)) / 32767
    return np.round(finite / slope).astype(np.int16), slope, True


def _read_reference(reference) -> Dataset | None:
    if reference is None:
        return None
    if isinstance(reference, Dataset):
        return reference
    if isinstance(reference, (list, tuple)):
        reference = reference[0]
    return pydicom.dcmread(str(reference), stop_before_pixels=True)


def _number(x: float) -> str:
    """A DICOM decimal string (at most 16 characters)."""
    s = f"{x:.10g}"
    return s if len(s) <= 16 else f"{x:.6e}"


def _base(modality: str, ref: Dataset | None, series_uid: str, frame_uid: str, series_description: str,
          series_number: int, now: datetime.datetime) -> Dataset:
    ds = Dataset()
    ds.SpecificCharacterSet = "ISO_IR 100"
    for key in PATIENT_STUDY:
        if ref is not None and key in ref:
            elem = ref[key]
            value = str(elem.value) if elem.value is not None else ""
            if key in VALID_FORM and value and not re.fullmatch(VALID_FORM[key], value):
                elem = pydicom.DataElement(elem.tag, elem.VR, "")
            ds[key] = copy.deepcopy(elem)                               # a copy: the reference stays as it was
    if ref is None or "StudyInstanceUID" not in ref:
        ds.PatientName = getattr(ref, "PatientName", "") if ref is not None else "PyTomography"
        ds.PatientID = getattr(ref, "PatientID", "") if ref is not None else "PYTOMOGRAPHY"
        ds.StudyInstanceUID = generate_uid()
        ds.StudyDate = now.strftime("%Y%m%d")
        ds.StudyTime = now.strftime("%H%M%S")
        ds.StudyID = "1"
    for key in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "ReferringPhysicianName",
                "AccessionNumber", "StudyID"):           # type 2: present, possibly empty
        if key not in ds:
            setattr(ds, key, "")
    ds.FrameOfReferenceUID = frame_uid
    if "PositionReferenceIndicator" not in ds:
        ds.PositionReferenceIndicator = ""
    ds.Modality = modality
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = int(series_number)
    ds.SeriesDescription = series_description[:64]
    ds.SeriesDate = ds.ContentDate = now.strftime("%Y%m%d")
    ds.SeriesTime = ds.ContentTime = now.strftime("%H%M%S")
    ds.Manufacturer = "PyTomography"
    ds.ManufacturerModelName = f"PyTomography {pytomography.__version__}"[:64]
    ds.SoftwareVersions = str(pytomography.__version__)
    ds.SOPClassUID = SOP_CLASS[modality]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    return ds


def _patient_orientation(ds: Dataset, ref: Dataset | None) -> None:
    """The NM/PET Patient Orientation module (type 2): the reference's codes, or empty when there are none."""
    for key in ("PatientOrientationCodeSequence", "PatientGantryRelationshipCodeSequence"):
        ds[key] = ref[key] if ref is not None and key in ref else pydicom.DataElement(key, "SQ", DicomSequence([]))


def _file(ds: Dataset, path: Path) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = ds.SOPClassUID
    meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    fds = FileDataset(str(path), ds, file_meta=meta, preamble=b"\0" * 128)
    fds.save_as(str(path), enforce_file_format=True)


def save_dicom(image, folder, object_meta=None, *, reference=None, units: str | None = None,
               modality: str | None = None, slice_thickness: float | None = None,
               series_description: str = "PyTomography reconstruction", series_number: int = 1000,
               overwrite: bool = False) -> list:
    """Save a 3D image as a DICOM series, each voxel in its place in the patient.

    Args:
        image (torch.Tensor | np.ndarray | Image): the image, axes (x, y, z), or an :class:`~pytomography.io.Image`.
        folder (str | Path): the folder to write into (created if needed).
        object_meta (ObjectMeta, optional): the image's grid and where it is (needed for a tensor or an array); its
            units and modality are used unless given here.
        reference (str | Path | pydicom.Dataset | list, optional): a DICOM file of the same acquisition (projections,
            a CT slice, raw CT projections). The series takes its patient and study, and, for an image in the
            patient's frame (``frame='patient'``), its frame of reference, so it lines up with that study's other
            images. Without one, or for an image placed only relative to the scanner (``frame='scanner'``), the
            series gets a new frame of reference.
        units (str, optional): the image's units, e.g. ``"counts"`` or ``"Bq/mL"`` (PET: written as DICOM's code,
            CNTS or BQML). CT is in HU, rounded to whole numbers, unless the units say otherwise (e.g. ``"1/mm"`` for
            attenuation), which keeps fractional values. Defaults to the units of ``object_meta`` or of the Image.
        modality (str, optional): ``"NM"`` (SPECT; one multi-frame file), ``"PT"`` (PET; one file per slice) or
            ``"CT"`` (one file per slice). Defaults to the modality of ``object_meta`` or of the Image; ``"PT"`` also
            saves a SPECT image as PET-style slices.
        slice_thickness (float, optional): the nominal thickness of a slice in mm, written as SliceThickness. This is
            not the distance between slices (which the slices' positions give), and is left empty when not given.
        series_description (str): shown in DICOM viewers' series lists.
        series_number (int): the series number.
        overwrite (bool): replace DICOM files already in ``folder``. Defaults to False, which refuses to.

    Returns:
        list[Path]: the files written, in slice order.
    """
    arr, A, own_units, own_modality, frame = _source(image, object_meta)
    units = units if units is not None else own_units
    modality = modality if modality is not None else own_modality
    if modality is None:
        raise ValueError("the image's modality is unknown: pass modality='NM', 'PT' or 'CT' (or set object_meta.modality)")
    modality = str(modality).upper()
    if modality not in SOP_CLASS:
        raise ValueError(f"modality must be one of {sorted(SOP_CLASS)}, not {modality!r}")
    geo = _geometry(A, arr.shape)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    old = sorted(folder.glob("*.dcm"))
    if old and not overwrite:
        raise FileExistsError(f"{folder} already has {len(old)} DICOM files; pass overwrite=True to replace them")
    for f in old:
        f.unlink()
    ref = _read_reference(reference)
    now = datetime.datetime.now()
    series_uid = generate_uid()
    # an image placed only relative to the scanner is not in the reference's frame of reference
    in_frame = frame == "patient" and ref is not None and "FrameOfReferenceUID" in ref
    frame_uid = ref.FrameOfReferenceUID if in_frame else generate_uid()
    pixels, slope, signed = _scaling(arr, modality, units)
    base = _base(modality, ref, series_uid, frame_uid, series_description, series_number, now)
    base.PixelRepresentation = 1 if signed else 0
    base.Rows, base.Columns = int(arr.shape[1]), int(arr.shape[0])       # rows run along y (j), columns along x (i)
    base.PixelSpacing = [_number(geo["dy"]), _number(geo["dx"])]         # between rows, between columns
    base.SliceThickness = _number(float(slice_thickness)) if slice_thickness is not None else ""
    base.ImageOrientationPatient = [_number(c) for c in (*geo["row"], *geo["col"])]
    if modality == "NM":                       # NM gives real values through a Real World Value Mapping instead
        return [_write_nm(base, pixels, slope, geo, folder, ref, units)]
    base.RescaleSlope, base.RescaleIntercept = _number(slope), "0"
    if modality == "CT":
        base.ImageType = ["DERIVED", "PRIMARY", "AXIAL"]
        base.RescaleType = "HU" if _is_hu(modality, units) else "US"   # US: unspecified (e.g. attenuation per mm)
        base.KVP = getattr(ref, "KVP", "") if ref is not None else ""
        base.AcquisitionNumber = 1
        if "PatientPosition" not in base:      # required for CT (type 2C), empty when unknown (e.g. a micro-CT sample)
            base.PatientPosition = ""
    else:
        base.ImageType = ["DERIVED", "PRIMARY"]
        base.Units = PET_UNITS.get((units or "counts").lower(), "CNTS")
        base.SeriesType = ["STATIC", "IMAGE"]
        base.CountsSource = "EMISSION"
        base.DecayCorrection = "NONE"
        base.CorrectedImage = []
        base.NumberOfSlices = int(arr.shape[2])
        base.FrameReferenceTime = "0"
        base.AcquisitionDate = getattr(ref, "AcquisitionDate", base.ContentDate) if ref is not None else base.ContentDate
        base.AcquisitionTime = getattr(ref, "AcquisitionTime", base.ContentTime) if ref is not None else base.ContentTime
        base.ActualFrameDuration = int(getattr(ref, "ActualFrameDuration", 0) or 0) if ref is not None else 0   # ms
        base.CollimatorType = "NONE"                                       # PET: no collimator
        # PET Isotope module: required, and empty when unknown (simulated data, or a reference without one)
        base.RadiopharmaceuticalInformationSequence = (ref.RadiopharmaceuticalInformationSequence if ref is not None
                                                       and "RadiopharmaceuticalInformationSequence" in ref
                                                       else DicomSequence([]))
        _patient_orientation(base, ref)
    files = []
    for k in range(arr.shape[2]):
        ds = copy.deepcopy(base)
        ds.SOPInstanceUID = generate_uid()
        ds.InstanceNumber = k + 1
        if modality == "PT":
            ds.ImageIndex = k + 1
        ds.ImagePositionPatient = [_number(c) for c in geo["ipp"][k]]
        ds.SliceLocation = _number(float(geo["ipp"][k] @ geo["normal"]))
        ds.PixelData = np.ascontiguousarray(pixels[:, :, k].T).tobytes()    # (rows, columns) = (y, x)
        path = folder / f"{modality}_{k + 1:04d}.dcm"
        _file(ds, path)
        files.append(path)
    return files


def _write_nm(base: Dataset, pixels: np.ndarray, slope: float, geo: dict, folder: Path, ref: Dataset | None,
              units) -> Path:
    """SPECT as one NM multi-frame file of reconstructed slices (RECON TOMO). Frames run along the slice normal (the
    cross product of the row and column directions), as DICOM requires, so slices are reversed when k runs against it.
    Where the slices are is in the Detector Information Sequence. The stored integers become real values through a
    Real World Value Mapping with the units, NM's own way, and through the same slope as Rescale Slope and Intercept:
    the NM IOD doesn't define those, but GDCM (3D Slicer, SimpleITK) and pydicom read only them, and show the stored
    integers without them. The standard allows extra standard attributes (a Standard Extended SOP Class)."""
    order = list(range(pixels.shape[2])) if geo["ascending"] else list(range(pixels.shape[2]))[::-1]
    ds = base
    ds.SOPInstanceUID = generate_uid()
    ds.InstanceNumber = 1
    ds.ImageType = ["DERIVED", "PRIMARY", "RECON TOMO", "EMISSION"]
    ds.NumberOfFrames = len(order)
    ds.FrameIncrementPointer = pydicom.tag.Tag(0x0054, 0x0080)          # SliceVector
    ds.SliceVector = list(range(1, len(order) + 1))
    ds.NumberOfSlices = len(order)
    ds.SpacingBetweenSlices = _number(geo["dz"])
    detector = Dataset()
    detector.ImagePositionPatient = [_number(c) for c in geo["ipp"][order[0]]]
    detector.ImageOrientationPatient = base.ImageOrientationPatient
    ref_det = ref.DetectorInformationSequence[0] if ref is not None and ref.get("DetectorInformationSequence") else None
    detector.CollimatorType = getattr(ref_det, "CollimatorType", "") if ref_det is not None else ""
    detector.FocalDistance = getattr(ref_det, "FocalDistance", "") if ref_det is not None else ""
    ds.DetectorInformationSequence = DicomSequence([detector])
    del ds.ImageOrientationPatient                                     # it lives in the Detector Information Sequence
    ds.NumberOfDetectors = 1
    ds.NumberOfEnergyWindows = 1
    ds.CountsAccumulated = ""
    # one energy window: the reference's, when it has exactly one (a reconstruction doesn't say which of several it used)
    windows = ref.get("EnergyWindowInformationSequence") if ref is not None else None
    ds.EnergyWindowInformationSequence = DicomSequence(list(windows) if windows is not None and len(windows) == 1 else [])
    for key in ("RadiopharmaceuticalInformationSequence", "RotationInformationSequence"):
        if ref is not None and key in ref:
            ds[key] = ref[key]
    if "RadiopharmaceuticalInformationSequence" not in ds:            # required, empty when unknown (NM Isotope module)
        ds.RadiopharmaceuticalInformationSequence = DicomSequence([])
    _patient_orientation(ds, ref)
    code, meaning = UCUM.get((units or "counts").strip().lower(), ("1", "no units"))
    unit = Dataset()
    unit.CodeValue, unit.CodingSchemeDesignator, unit.CodeMeaning = code, "UCUM", meaning
    rwvm = Dataset()
    rwvm.LUTExplanation = "PyTomography reconstruction"
    rwvm.LUTLabel = "PYTOMOGRAPHY"
    rwvm.MeasurementUnitsCodeSequence = DicomSequence([unit])
    rwvm.RealWorldValueFirstValueMapped = min(0, int(pixels.min()))     # negative stored values too, when signed
    rwvm.RealWorldValueLastValueMapped = int(pixels.max())
    rwvm.RealWorldValueIntercept = 0.0
    rwvm.RealWorldValueSlope = float(slope)
    ds.RealWorldValueMappingSequence = DicomSequence([rwvm])
    ds.RescaleSlope, ds.RescaleIntercept = _number(slope), "0"
    ds.PixelData = np.ascontiguousarray(np.stack([pixels[:, :, k].T for k in order])).tobytes()
    path = folder / "NM_0001.dcm"
    _file(ds, path)
    return path
