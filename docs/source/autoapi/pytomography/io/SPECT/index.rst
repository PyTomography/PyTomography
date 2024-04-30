:py:mod:`pytomography.io.SPECT`
===============================

.. py:module:: pytomography.io.SPECT

.. autoapi-nested-parse::

   Input/output functions for the SPECT imaging modality. Currently, the data types supported are SIMIND and DICOM files.



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   dicom/index.rst
   shared/index.rst
   simind/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.SPECT.get_attenuation_map
   pytomography.io.SPECT.get_projections
   pytomography.io.SPECT.get_attenuation_map_from_file
   pytomography.io.SPECT.get_attenuation_map_from_CT_slices
   pytomography.io.SPECT.get_energy_window_scatter_estimate
   pytomography.io.SPECT.get_psfmeta_from_scanner_params
   pytomography.io.SPECT.CT_to_mumap
   pytomography.io.SPECT.subsample_amap
   pytomography.io.SPECT.subsample_projections_and_modify_metadata



.. py:function:: get_attenuation_map(headerfile)

   Opens attenuation data from SIMIND output

   :param headerfile: Path to header file
   :type headerfile: str

   :returns: Tensor containing attenuation map required for attenuation correction in SPECT/PET imaging.
   :rtype: torch.Tensor[batch_size, Lx, Ly, Lz]


.. py:function:: get_projections(file, index_peak = None, index_time = None)

   Gets projections from a .dcm file.

   :param file: Path to the .dcm file of SPECT projection data.
   :type file: str
   :param index_peak: If not none, then the returned projections correspond to the index of this energy window. Otherwise returns all energy windows. Defaults to None.
   :type index_peak: int
   :param index_time: If not none, then the returned projections correspond to the index of the time slot in gated SPECT. Otherwise returns all time slots. Defaults to None
   :type index_time: int

   :returns: (SPECTObjectMeta, SPECTProjMeta, torch.Tensor[..., Ltheta, Lr, Lz]) where ... depends on if time slots are considered.


.. py:function:: get_attenuation_map_from_file(file_AM)

   Gets an attenuation map from a DICOM file. This data is usually provided by the manufacturer of the SPECT scanner.

   :param file_AM: File name of attenuation map
   :type file_AM: str

   :returns: Tensor of shape [batch_size, Lx, Ly, Lz] corresponding to the atteunation map in units of cm:math:`^{-1}`
   :rtype: torch.Tensor


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


.. py:function:: subsample_amap(amap, N)

   Subsamples 3D attenuation map by averaging over N x N x N regions

   :param amap: Original attenuation map
   :type amap: torch.Tensor
   :param N: Factor to reduce by
   :type N: int

   :returns: Subsampled attenuation map
   :rtype: torch.Tensor


.. py:function:: subsample_projections_and_modify_metadata(object_meta, proj_meta, projections, N_pixel = 1, N_angle = 1, N_angle_start = 0)

   Subsamples SPECT projection and modifies metadata accordingly

   :param object_meta: Object metadata
   :type object_meta: ObjectMeta
   :param proj_meta: Projection metadata
   :type proj_meta: SPECTProjMeta
   :param projections: Projections to subsample
   :type projections: torch.Tensor
   :param N_pixel: Pixel reduction factor (1 means no reduction). Defaults to 1.
   :type N_pixel: int
   :param N_angle: Angle reduction factor (1 means no reduction). Defaults to 1.
   :type N_angle: int
   :param N_angle_start: Angle index to start at. Defaults to 0.
   :type N_angle_start: int

   :returns: Modified object metadata, modified projection metadata, subsampled projections
   :rtype: Sequence


