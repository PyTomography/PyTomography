from __future__ import annotations
import datetime
import pytomography
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ImplicitVRLittleEndian
from pydicom.uid import PYDICOM_IMPLEMENTATION_UID

def get_file_meta(SOP_instance_UID: str, SOP_class_UID: str) -> FileMetaDataset:
    """Creates DICOM file metadata given an SOP instance and class UID.

    Args:
        SOP_instance_UID (str): Identifier unique to each DICOM file
        SOP_class_UID (str): Identifier specifying imaging modality

    Returns:
        FileMetaDataset: Metadata for DICOM file
    """
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 202
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    #file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.128'
    file_meta.MediaStorageSOPClassUID = SOP_class_UID
    file_meta.MediaStorageSOPInstanceUID = (SOP_instance_UID)  
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    return file_meta

def generate_base_dataset(SOP_instance_UID: str, SOP_class_UID: str) -> FileDataset:
    """Generates a base dataset with the minimal number of required parameters

    Args:
        SOP_instance_UID (str): Identifier unique to each DICOM file
        SOP_class_UID (str): Identifier specifying imaging modality

    Returns:
        FileDataset: DICOM dataset
    """
    file_name = "pydicom-reconstruction"
    file_meta = get_file_meta(SOP_instance_UID, SOP_class_UID)
    ds = FileDataset(file_name, {}, file_meta=file_meta, preamble=b"\0" * 128)
    add_required_elements_to_ds(ds)
    return ds

def add_required_elements_to_ds(ds: FileDataset) -> None:
    """Adds elements to dataset including timing and manufacturer details

    Args:
        ds (FileDataset): DICOM dataset that will be updated
    """
    dt = datetime.datetime.now()
    # Append data elements required by the DICOM standarad
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.InstanceCreationDate = dt.strftime("%Y%m%d")
    ds.InstanceCreationTime = dt.strftime("%H%M%S.%f")
    ds.Manufacturer = "Qurit"
    ds.ManufacturerModelName = f"PyTomography {pytomography.__version__}"
    ds.InstitutionName = "Qurit"
    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    # Set values already defined in the file meta
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.ApprovalStatus = "UNAPPROVED"
    
def add_study_and_series_information(ds: FileDataset, reference_ds: FileDataset) -> None:
    """Adds study and series information to dataset based on reference dataset

    Args:
        ds (FileDataset): Dataset for which to add headers
        reference_ds (FileDataset): Dataset from which to copy headers
    """
    ds.StudyDate = reference_ds.StudyDate
    ds.SeriesDate = getattr(reference_ds, "SeriesDate", "")
    ds.StudyTime = reference_ds.StudyTime
    ds.SeriesTime = getattr(reference_ds, "SeriesTime", "")
    ds.StudyDescription = getattr(reference_ds, "StudyDescription", "")
    ds.SeriesDescription = getattr(reference_ds, "SeriesDescription", "")
    ds.StudyInstanceUID = reference_ds.StudyInstanceUID
    ds.SeriesInstanceUID = generate_uid()  # TODO: find out if random generation is ok
    ds.StudyID = reference_ds.StudyID
    ds.SeriesNumber = "1"  # TODO: find out if we can just use 1 (Should be fine since its a new series)
    ds.FrameOfReferenceUID =  getattr(reference_ds, 'FrameOfReferenceUID', generate_uid())
    
def add_patient_information(ds: FileDataset, reference_ds):
    """Adds patient information to dataset based on reference dataset

    Args:
        ds (FileDataset): Dataset for which to add headers
        reference_ds (FileDataset): Dataset from which to copy headers
    """
    ds.PatientName = getattr(reference_ds, "PatientName", "")
    ds.PatientID = getattr(reference_ds, "PatientID", "")
    ds.PatientBirthDate = getattr(reference_ds, "PatientBirthDate", "")
    ds.PatientSex = getattr(reference_ds, "PatientSex", "")
    ds.PatientAge = getattr(reference_ds, "PatientAge", "")
    ds.PatientSize = getattr(reference_ds, "PatientSize", "")
    ds.PatientWeight = getattr(reference_ds, "PatientWeight", "")
    
def create_ds(reference_ds: FileDataset, SOP_instance_UID: str, SOP_class_UID: str, modality: str, imagetype: str):
    """Creates a new DICOM dataset based on a reference dataset with all required headers. Because this is potentially used to save images corresponding to different modalities, the UIDs must be input arguments to this function. In addition, since some modalities require saving multiple slices whereby ``SOP_instance_UIDs`` may use some convention to specify slice number, these are also input arguments.

    Args:
        reference_ds (FileDataset): Dataset from which to copy all important headers such as patient information and study UID. 
        SOP_instance_UID (str): Unique identifier for the particular instance (this is different for every DICOM file created)
        SOP_class_UID (str): Unique identifier for the imaging modality
        modality (str): String specifying imaging modality
        imagetype (str): String specifying image type

    Returns:
        _type_: _description_
    """
    ds = generate_base_dataset(SOP_instance_UID, SOP_class_UID)
    add_study_and_series_information(ds, reference_ds)
    add_patient_information(ds, reference_ds)
    ds.Modality = modality
    ds.ImageType = imagetype
    return ds