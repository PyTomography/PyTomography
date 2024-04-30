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

   dicom/index.rst
   dicom_creation/index.rst
   interfile/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.shared.create_ds
   pytomography.io.shared.get_header_value
   pytomography.io.shared.get_attenuation_map_interfile
   pytomography.io.shared.open_multifile
   pytomography.io.shared.align_images_affine
   pytomography.io.shared._get_affine_multifile



.. py:function:: create_ds(reference_ds, SOP_instance_UID, SOP_class_UID, modality, imagetype)

   Creates a new DICOM dataset based on a reference dataset with all required headers. Because this is potentially used to save images corresponding to different modalities, the UIDs must be input arguments to this function. In addition, since some modalities require saving multiple slices whereby ``SOP_instance_UIDs`` may use some convention to specify slice number, these are also input arguments.

   :param reference_ds: Dataset from which to copy all important headers such as patient information and study UID.
   :type reference_ds: FileDataset
   :param SOP_instance_UID: Unique identifier for the particular instance (this is different for every DICOM file created)
   :type SOP_instance_UID: str
   :param SOP_class_UID: Unique identifier for the imaging modality
   :type SOP_class_UID: str
   :param modality: String specifying imaging modality
   :type modality: str
   :param imagetype: String specifying image type
   :type imagetype: str

   :returns: _description_
   :rtype: _type_


.. py:function:: get_header_value(list_of_attributes, header, dtype = np.float32, split_substr=':=', split_idx=-1, return_all=False)

   Finds the first entry in an Interfile with the string ``header``

   :param list_of_attributes: Simind data file, as a list of lines.
   :type list_of_attributes: list[str]
   :param header: The header looked for
   :type header: str
   :param dtype: The data type to be returned corresponding to the value of the header. Defaults to np.float32.
   :type dtype: type, optional

   :returns: The value corresponding to the header (header).
   :rtype: float|str|int


.. py:function:: get_attenuation_map_interfile(headerfile)

   Opens attenuation data from SIMIND output

   :param headerfile: Path to header file
   :type headerfile: str

   :returns: Tensor containing attenuation map required for attenuation correction in SPECT/PET imaging.
   :rtype: torch.Tensor[batch_size, Lx, Ly, Lz]


.. py:function:: open_multifile(files)

   Given a list of seperate DICOM files, opens them up and stacks them together into a single CT image.

   :param files: List of CT DICOM filepaths corresponding to different z slices of the same scan.
   :type files: Sequence[str]

   :returns: CT scan in units of Hounsfield Units at the effective CT energy.
   :rtype: np.array


.. py:function:: align_images_affine(im_fixed, im_moving, affine_fixed, affine_moving, cval=0)


.. py:function:: _get_affine_multifile(files)

   Computes an affine matrix corresponding the coordinate system of a CT DICOM file. Note that since CT scans consist of many independent DICOM files, ds corresponds to an individual one of these files. This is why the maximum z value is also required (across all seperate independent DICOM files).

   :param ds: DICOM dataset of CT data
   :type ds: Dataset
   :param max_z: Maximum value of z across all axial slices that make up the CT scan
   :type max_z: float

   :returns: Affine matrix corresponding to CT scan.
   :rtype: np.array


