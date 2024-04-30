:py:mod:`pytomography.io.SPECT.dicom`
=====================================

.. py:module:: pytomography.io.SPECT.dicom


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.SPECT.dicom.parse_projection_dataset
   pytomography.io.SPECT.dicom.get_metadata
   pytomography.io.SPECT.dicom.get_projections
   pytomography.io.SPECT.dicom.get_window_width
   pytomography.io.SPECT.dicom.get_energy_window_scatter_estimate
   pytomography.io.SPECT.dicom.get_energy_window_scatter_estimate_projections
   pytomography.io.SPECT.dicom.get_attenuation_map_from_file
   pytomography.io.SPECT.dicom.get_psfmeta_from_scanner_params
   pytomography.io.SPECT.dicom.CT_to_mumap
   pytomography.io.SPECT.dicom.get_attenuation_map_from_CT_slices
   pytomography.io.SPECT.dicom._get_affine_spect_projections
   pytomography.io.SPECT.dicom.load_multibed_projections
   pytomography.io.SPECT.dicom.load_multibed_projections
   pytomography.io.SPECT.dicom.stitch_multibed
   pytomography.io.SPECT.dicom.get_aligned_rtstruct
   pytomography.io.SPECT.dicom.get_aligned_nifti_mask
   pytomography.io.SPECT.dicom.save_dcm



.. py:function:: parse_projection_dataset(ds)

   Gets projections with corresponding radii and angles corresponding to projection data from a DICOM file.

   :param ds: pydicom dataset object.
   :type ds: Dataset

   :returns: Returns (i) projection data (ii) angles (iii) radii and (iv) flags for whether or not multiple energy windows/time slots were detected.
   :rtype: (torch.tensor[EWindows, TimeWindows, Ltheta, Lr, Lz], np.array, np.array)


.. py:function:: get_metadata(file, index_peak = 0)

   Gets PyTomography metadata from a .dcm file.

   :param file: Path to the .dcm file of SPECT projection data.
   :type file: str
   :param index_peak: EnergyWindowInformationSequence index corresponding to the photopeak. Defaults to 0.
   :type index_peak: int

   :returns: Required metadata information for reconstruction in PyTomography.
   :rtype: (ObjectMeta, ProjMeta)


.. py:function:: get_projections(file, index_peak = None, index_time = None)

   Gets projections from a .dcm file.

   :param file: Path to the .dcm file of SPECT projection data.
   :type file: str
   :param index_peak: If not none, then the returned projections correspond to the index of this energy window. Otherwise returns all energy windows. Defaults to None.
   :type index_peak: int
   :param index_time: If not none, then the returned projections correspond to the index of the time slot in gated SPECT. Otherwise returns all time slots. Defaults to None
   :type index_time: int

   :returns: (SPECTObjectMeta, SPECTProjMeta, torch.Tensor[..., Ltheta, Lr, Lz]) where ... depends on if time slots are considered.


.. py:function:: get_window_width(ds, index)

   Computes the width of an energy window corresponding to a particular index in the DetectorInformationSequence DICOM attribute.

   :param ds: DICOM dataset.
   :type ds: Dataset
   :param index: Energy window index corresponding to the DICOM dataset.
   :type index: int

   :returns: Range of the energy window in keV
   :rtype: float


.. py:function:: get_energy_window_scatter_estimate(file, index_peak, index_lower, index_upper = None, weighting_lower = 0.5, weighting_upper = 0.5, return_scatter_variance_estimate = False)

   Gets an estimate of scatter projection data from a DICOM file using either the dual energy window (`index_upper=None`) or triple energy window method.

   :param file: Filepath of the DICOM file
   :type file: str
   :param index_peak: Index of the ``EnergyWindowInformationSequence`` DICOM attribute corresponding to the photopeak.
   :type index_peak: int
   :param index_lower: Index of the ``EnergyWindowInformationSequence`` DICOM attribute corresponding to lower scatter window.
   :type index_lower: int
   :param index_upper: Index of the ``EnergyWindowInformationSequence`` DICOM attribute corresponding to upper scatter window. Defaults to None (dual energy window).
   :type index_upper: int
   :param weighting_lower: Weighting of the lower scatter window. Defaults to 0.5.
   :type weighting_lower: float
   :param weighting_upper: Weighting of the upper scatter window. Defaults to 0.5.
   :type weighting_upper: float
   :param return_scatter_variance_estimate: If true, then also return the variance estimate of the scatter. Defaults to False.
   :type return_scatter_variance_estimate: bool

   :returns: Tensor corresponding to the scatter estimate.
   :rtype: torch.Tensor[Ltheta,Lr,Lz]


.. py:function:: get_energy_window_scatter_estimate_projections(file, projections, index_peak, index_lower, index_upper = None, weighting_lower = 0.5, weighting_upper = 0.5, return_scatter_variance_estimate = False)

   Gets an estimate of scatter projection data from a DICOM file using either the dual energy window (`index_upper=None`) or triple energy window method. This is seperate from ``get_energy_window_scatter_estimate`` as it allows a user to input projecitons that are already loaded/modified. This is useful for when projection data gets mixed for reconstructing multiple bed positions.

   :param file: Filepath of the DICOM file
   :type file: str
   :param projections: Loaded projection data
   :type projections: torch.Tensor
   :param index_peak: Index of the ``EnergyWindowInformationSequence`` DICOM attribute corresponding to the photopeak.
   :type index_peak: int
   :param index_lower: Index of the ``EnergyWindowInformationSequence`` DICOM attribute corresponding to lower scatter window.
   :type index_lower: int
   :param index_upper: Index of the ``EnergyWindowInformationSequence`` DICOM attribute corresponding to upper scatter window.
   :type index_upper: int
   :param weighting_lower: Weighting of the lower scatter window. Defaults to 0.5.
   :type weighting_lower: float
   :param weighting_upper: Weighting of the upper scatter window. Defaults to 0.5.
   :type weighting_upper: float
   :param return_scatter_variance_estimate: If true, then also return the variance estimate of the scatter. Defaults to False.
   :type return_scatter_variance_estimate: bool

   :returns: Tensor corresponding to the scatter estimate.
   :rtype: torch.Tensor[Ltheta,Lr,Lz]


.. py:function:: get_attenuation_map_from_file(file_AM)

   Gets an attenuation map from a DICOM file. This data is usually provided by the manufacturer of the SPECT scanner.

   :param file_AM: File name of attenuation map
   :type file_AM: str

   :returns: Tensor of shape [batch_size, Lx, Ly, Lz] corresponding to the atteunation map in units of cm:math:`^{-1}`
   :rtype: torch.Tensor


.. py:function:: get_psfmeta_from_scanner_params(collimator_name, energy_keV, min_sigmas = 3, material = 'lead', intrinsic_resolution = 0)

   Obtains SPECT PSF metadata given a unique collimator code and photopeak energy of radionuclide. For more information on collimator codes, see the "external data" section of the readthedocs page.

   :param collimator_name: Code for the collimator used.
   :type collimator_name: str
   :param energy_keV: Energy of the photopeak
   :type energy_keV: float
   :param min_sigmas: Minimum size of the blurring kernel used. Fixes the convolutional kernel size so that all locations have at least ``min_sigmas`` in dimensions (some will be greater)
   :type min_sigmas: float
   :param material: Material of the collimator.
   :type material: str
   :param intrinsic_resolution: Intrinsic resolution (FWHM) of the scintillator crystals. Defaults to 0.
   :type intrinsic_resolution: float

   :returns: PSF metadata.
   :rtype: SPECTPSFMeta


.. py:function:: CT_to_mumap(CT, files_CT, file_NM, index_peak=0)

   Converts a CT image to a mu-map given SPECT projection data. The CT data must be aligned with the projection data already; this is a helper function for ``get_attenuation_map_from_CT_slices``.

   :param CT: CT object in units of HU
   :type CT: torch.tensor
   :param files_CT: Filepaths of all CT slices
   :type files_CT: Sequence[str]
   :param file_NM: Filepath of SPECT projectio ndata
   :type file_NM: str
   :param index_peak: Index of EnergyInformationSequence corresponding to the photopeak. Defaults to 0.
   :type index_peak: int, optional

   :returns: Attenuation map in units of 1/cm
   :rtype: torch.tensor


.. py:function:: get_attenuation_map_from_CT_slices(files_CT, file_NM = None, index_peak = 0, keep_as_HU = False, mode = 'constant', CT_output_shape = None, apply_affine = True)

   Converts a sequence of DICOM CT files (corresponding to a single scan) into a torch.Tensor object usable as an attenuation map in PyTomography.

   :param files_CT: List of all files corresponding to an individual CT scan
   :type files_CT: Sequence[str]
   :param file_NM: File corresponding to raw PET/SPECT data (required to align CT with projections). If None, then no alignment is done. Defaults to None.
   :type file_NM: str
   :param index_peak: Index corresponding to photopeak in projection data. Defaults to 0.
   :type index_peak: int, optional
   :param keep_as_HU: If True, then don't convert to linear attenuation coefficient and keep as Hounsfield units. Defaults to False
   :type keep_as_HU: bool
   :param CT_output_shape: If not None, then the CT is returned with the desired dimensions. Otherwise, it defaults to the shape in the file_NM data.
   :type CT_output_shape: Sequence, optional
   :param apply_affine: Whether or not to align CT with NM.
   :type apply_affine: bool

   :returns: Tensor of shape [Lx, Ly, Lz] corresponding to attenuation map.
   :rtype: torch.Tensor


.. py:function:: _get_affine_spect_projections(filename)

   Computes an affine matrix corresponding the coordinate system of a SPECT DICOM file of projections.

   :param ds: DICOM dataset of projection data
   :type ds: Dataset

   :returns: Affine matrix
   :rtype: np.array


.. py:function:: load_multibed_projections(files_NM, index_lower = 20, index_upper = 106)

   This function loads projection data from each of the files in files_NM; for locations outside the FOV in each projection, it appends the data from the adjacent projection. The field of view (in z) is specified by ``index_lower`` and ``index_upper``. The default values of 20 and 106 seem to be sufficient for most scanners.

   :param files_NM: Filespaths for each of the projections
   :type files_NM: str
   :param index_lower: Z-pixel index specifying the lower boundary of the FOV. Defaults to 20.
   :type index_lower: int, optional
   :param index_upper: Z-pixel index specifying the upper boundary of the FOV. Defaults to 106.
   :type index_upper: int, optional

   :returns: Tensor of shape ``[N_bed_positions, N_energy_windows, Ltheta, Lr, Lz]``.
   :rtype: torch.Tensor


.. py:function:: load_multibed_projections(files_NM)

   This function loads projection data from each of the files in files_NM; for locations outside the FOV in each projection, it appends the data from the adjacent projection (it uses the midway point between the projection overlap).

   :param files_NM: Filespaths for each of the projections
   :type files_NM: str

   :returns: Tensor of shape ``[N_bed_positions, N_energy_windows, Ltheta, Lr, Lz]``.
   :rtype: torch.Tensor


.. py:function:: stitch_multibed(recons, files_NM, return_stitching_weights = False)

   Stitches together multiple reconstructed objects corresponding to different bed positions.

   :param recons: Reconstructed objects. The first index of the tensor corresponds to different bed positions
   :type recons: torch.Tensor[n_beds, Lx, Ly, Lz]
   :param files_NM: List of length ``n_beds`` corresponding to the DICOM file of each reconstruction
   :type files_NM: list
   :param return_stitching_weights: If true, instead of returning stitched reconstruction, instead returns the stitching weights (and z location in the stitched image) for each bed position (this is used as a tool for uncertainty estimation in multi bed positions). Defaults to False
   :type return_stitching_weights: bool

   :returns: Stitched together DICOM file. Note the new z-dimension size :math:`L_z'`.
   :rtype: torch.Tensor[Lx, Ly, Lz']


.. py:function:: get_aligned_rtstruct(file_RT, file_NM, dicom_series_path, rt_struct_name, cutoff_value=0.5, shape=None)

   Loads an RT struct file and aligns it with SPECT projection data corresponding to ``file_NM``.

   :param file_RT: Filepath of the RT Struct file
   :type file_RT: str
   :param file_NM: Filepath of the NM file (used to align the RT struct)
   :type file_NM: str
   :param dicom_series_path: Filepath of the DICOM series linked to the RTStruct file (required for loading RTStructs).
   :type dicom_series_path: str
   :param rt_struct_name: Name of the desired RT struct.
   :type rt_struct_name: str
   :param cutoff_value: After interpolation is performed to align the mask in the new frame, mask voxels with values less than this are excluded. Defaults to 0.5.
   :type cutoff_value: float, optional

   :returns: RTStruct mask aligned with SPECT data.
   :rtype: torch.Tensor


.. py:function:: get_aligned_nifti_mask(file_nifti, file_NM, dicom_series_path, mask_idx, cutoff_value=0.5, shape=None)

   Loads an RT struct file and aligns it with SPECT projection data corresponding to ``file_NM``.

   :param file_nifti: Filepath of the nifti file containing the reconstruction mask
   :type file_nifti: str
   :param file_NM: Filepath of the NM file (used to align the RT struct)
   :type file_NM: str
   :param dicom_series_path: Filepath of the DICOM series linked to the RTStruct file (required for loading RTStructs).
   :type dicom_series_path: str
   :param mask_idx: Integer in nifti mask corresponding to ROI.
   :type mask_idx: str
   :param cutoff_value: After interpolation is performed to align the mask in the new frame, mask voxels with values less than this are excluded. Defaults to 0.5.
   :type cutoff_value: float, optional

   :returns: RTStruct mask aligned with SPECT data.
   :rtype: torch.Tensor


.. py:function:: save_dcm(save_path, object, file_NM, recon_name = '', return_ds = False, single_dicom_file = False, scale_by_number_projections = False)

   Saves the reconstructed object `object` to a series of DICOM files in the folder given by `save_path`. Requires the filepath of the projection data `file_NM` to get Study information.

   :param object: Reconstructed object of shape [Lx,Ly,Lz].
   :type object: torch.Tensor
   :param save_path: Location of folder where to save the DICOM output files.
   :type save_path: str
   :param file_NM: File path of the projection data corresponding to the reconstruction.
   :type file_NM: str
   :param recon_name: Type of reconstruction performed. Obtained from the `recon_method_str` attribute of a reconstruction algorithm class.
   :type recon_name: str
   :param return_ds: If true, returns the DICOM dataset objects instead of saving to file. Defaults to False.
   :type return_ds: bool


