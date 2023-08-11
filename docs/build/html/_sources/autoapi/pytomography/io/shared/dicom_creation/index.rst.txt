:py:mod:`pytomography.io.shared.dicom_creation`
===============================================

.. py:module:: pytomography.io.shared.dicom_creation


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.shared.dicom_creation.get_file_meta
   pytomography.io.shared.dicom_creation.generate_base_dataset
   pytomography.io.shared.dicom_creation.add_required_elements_to_ds
   pytomography.io.shared.dicom_creation.add_study_and_series_information
   pytomography.io.shared.dicom_creation.add_patient_information
   pytomography.io.shared.dicom_creation.create_ds



.. py:function:: get_file_meta(SOP_instance_UID, SOP_class_UID)

   Creates DICOM file metadata given an SOP instance and class UID.

   :param SOP_instance_UID: Identifier unique to each DICOM file
   :type SOP_instance_UID: str
   :param SOP_class_UID: Identifier specifying imaging modality
   :type SOP_class_UID: str

   :returns: Metadata for DICOM file
   :rtype: FileMetaDataset


.. py:function:: generate_base_dataset(SOP_instance_UID, SOP_class_UID)

   Generates a base dataset with the minimal number of required parameters

   :param SOP_instance_UID: Identifier unique to each DICOM file
   :type SOP_instance_UID: str
   :param SOP_class_UID: Identifier specifying imaging modality
   :type SOP_class_UID: str

   :returns: DICOM dataset
   :rtype: FileDataset


.. py:function:: add_required_elements_to_ds(ds)

   Adds elements to dataset including timing and manufacturer details

   :param ds: DICOM dataset that will be updated
   :type ds: FileDataset


.. py:function:: add_study_and_series_information(ds, reference_ds)

   Adds study and series information to dataset based on reference dataset

   :param ds: Dataset for which to add headers
   :type ds: FileDataset
   :param reference_ds: Dataset from which to copy headers
   :type reference_ds: FileDataset


.. py:function:: add_patient_information(ds, reference_ds)

   Adds patient information to dataset based on reference dataset

   :param ds: Dataset for which to add headers
   :type ds: FileDataset
   :param reference_ds: Dataset from which to copy headers
   :type reference_ds: FileDataset


.. py:function:: create_ds(reference_ds, SOP_instance_UID, SOP_class_UID, modality)

   Creates a new DICOM dataset based on a reference dataset with all required headers. Because this is potentially used to save images corresponding to different modalities, the UIDs must be input arguments to this function. In addition, since some modalities require saving multiple slices whereby ``SOP_instance_UIDs`` may use some convention to specify slice number, these are also input arguments.

   :param reference_ds: Dataset from which to copy all important headers such as patient information and study UID.
   :type reference_ds: FileDataset
   :param SOP_instance_UID: Unique identifier for the particular instance (this is different for every DICOM file created)
   :type SOP_instance_UID: str
   :param SOP_class_UID: Unique identifier for the imaging modality
   :type SOP_class_UID: str
   :param modality: String specifying imaging modality
   :type modality: str

   :returns: _description_
   :rtype: _type_


