:py:mod:`pytomography.io.shared`
================================

.. py:module:: pytomography.io.shared

.. autoapi-nested-parse::

   Shared functionality between different imaging modalities.



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   dicom_creation/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.shared.create_ds



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


