:py:mod:`pytomography.io`
=========================

.. py:module:: pytomography.io


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   dicom/index.rst
   simind/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.simind_CT_to_data
   pytomography.io.simind_projections_to_data
   pytomography.io.simind_MEW_to_data
   pytomography.io.get_SPECT_recon_algorithm_simind
   pytomography.io.dicom_projections_to_data
   pytomography.io.dicom_CT_to_data
   pytomography.io.dicom_MEW_to_data
   pytomography.io.get_SPECT_recon_algorithm_dicom



.. py:function:: simind_CT_to_data(headerfile)

   Opens attenuation data from SIMIND output

   :param headerfile: Path to header file
   :type headerfile: str

   :returns: Tensor containing CT data.
   :rtype: torch.tensor[Lx,Ly,Lz]


.. py:function:: simind_projections_to_data(headerfile, distance = 'cm')

   Obtains ObjectMeta, ImageMeta, and projections from a SIMIND header file.

   :param headerfile: Path to the header file
   :type headerfile: str
   :param distance: The units of measurements in the SIMIND file (this is required as input, since SIMIND uses mm/cm but doesn't specify). Defaults to 'cm'.
   :type distance: str, optional

   :returns: Required information for reconstruction in PyTomography.
   :rtype: (ObjectMeta, ImageMeta, torch.Tensor[1, Ltheta, Lr, Lz])


.. py:function:: simind_MEW_to_data(headerfiles, distance = 'cm')

   Opens multiple projection files corresponding to the primary, lower scatter, and upper scatter windows

   :param headerfiles: List of file paths to required files. Must be in order of: 1. Primary, 2. Lower Scatter, 3. Upper scatter
   :type headerfiles: list[str]
   :param distance: The units of measurements in the SIMIND file (this is required as input, since SIMIND uses mm/cm but doesn't specify). Defaults to 'cm'.
   :type distance: str, optional

   :returns: Required information for reconstruction in PyTomography. First returned tensor contains primary data, and second returned tensor returns estimated scatter using the triple energy window method.
   :rtype: (ObjectMeta, ImageMeta, torch.Tensor[1, Ltheta, Lr, Lz], torch.Tensor[1, Ltheta, Lr, Lz])


.. py:function:: get_SPECT_recon_algorithm_simind(projections_header, scatter_headers = None, CT_header = None, psf_meta = None, prior = None, object_initial = None, recon_algorithm_class = OSEMOSL)


.. py:function:: dicom_projections_to_data(file)

   Obtains ObjectMeta, ImageMeta, and projections from a .dcm file.

   :param file: Path to the .dcm file
   :type file: str

   :returns: Required information for reconstruction in PyTomography.
   :rtype: (ObjectMeta, ImageMeta, torch.Tensor[1, Ltheta, Lr, Lz])


.. py:function:: dicom_CT_to_data(files_CT, file_NM, photopeak_window_index = 0)

   Converts a sequence of DICOM CT files (corresponding to a single scan) into a torch.Tensor object usable as an attenuation map in PyTomography. This is primarily intended for opening pre-reconstructed CT data such that it can be used as an attenuation map during PET/SPECT reconstruction.

   :param files_CT: List of all files corresponding to an individual CT scan
   :type files_CT: Sequence[str]
   :param file_NM: File corresponding to raw PET/SPECT data (required to align CT with projections)
   :type file_NM: str
   :param photopeak_window_index: Index corresponding to photopeak in projection data. Defaults to 0.
   :type photopeak_window_index: int, optional

   :returns: Tensor of shape [Lx, Ly, Lz] corresponding to attenuation map.
   :rtype: torch.Tensor


.. py:function:: dicom_MEW_to_data(file, type='DEW')


.. py:function:: get_SPECT_recon_algorithm_dicom(projections_file, atteunation_files = None, use_psf = False, scatter_type = None, prior = None, recon_algorithm_class = OSEMOSL, object_initial = None)

   Helper function to quickly create reconstruction algorithm given SPECT DICOM files and CT dicom files.

   :param projections_file: DICOM filepath corresponding to SPECT data.
   :type projections_file: str
   :param atteunation_files: DICOM filepaths corresponding to CT data. If None, then atteunation correction is not used. Defaults to None.
   :type atteunation_files: Sequence[str], optional
   :param use_psf: Whether or not to use PSF modeling. Defaults to False.
   :type use_psf: bool, optional
   :param scatter_type: Type of scatter correction used in reconstruction. Defaults to None.
   :type scatter_type: str | None, optional
   :param prior: Bayesian Prior used in reconstruction algorithm. Defaults to None.
   :type prior: Prior, optional
   :param recon_algorithm_class: Type of reconstruction algorithm used. Defaults to OSEMOSL.
   :type recon_algorithm_class: nn.Module, optional
   :param object_initial: Initial object used in reconstruction. If None, defaults to all ones. Defaults to None.
   :type object_initial: torch.Tensor | None, optional

   :raises Exception: If not able to compute relevant PSF parameters from DICOM data and corresponding data tables.

   :returns: Reconstruction algorithm used.
   :rtype: OSML


