:py:mod:`pytomography.io.PET.shared`
====================================

.. py:module:: pytomography.io.PET.shared


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.PET.shared.sinogram_coordinates
   pytomography.io.PET.shared.sinogram_to_spatial
   pytomography.io.PET.shared.listmode_to_sinogram
   pytomography.io.PET.shared._listmodeTOF_to_sinogramTOF
   pytomography.io.PET.shared.get_detector_ids_from_trans_axial_ids
   pytomography.io.PET.shared.get_axial_trans_ids_from_info
   pytomography.io.PET.shared.get_scanner_LUT
   pytomography.io.PET.shared.sinogram_to_listmode
   pytomography.io.PET.shared.smooth_randoms_sinogram
   pytomography.io.PET.shared.randoms_sinogram_to_sinogramTOF



.. py:function:: sinogram_coordinates(info)

   Obtains two tensors: the first yields the sinogram coordinates (r/theta) given two crystal IDs (shape [N_crystals_per_ring, N_crystals_per_ring, 2]), the second yields the sinogram index given two ring IDs (shape [Nrings, Nrings])

   :param info: PET geometry information dictionary
   :type info: dict

   :returns: LOR coordinates and sinogram index lookup tensors
   :rtype: Sequence[torch.Tensor]


.. py:function:: sinogram_to_spatial(info)

   Returns two tensors: the first yields the detector coordinates (x1/y1/x2/y2) of each of the two crystals given the element of the sinogram (shape [N_crystals_per_ring, N_crystals_per_ring, 2, 2]), the second yields the ring coordinates (z1/z2) given two ring IDs (shape [Nrings*Nrings, 2])

   :param info: PET geometry information dictionary
   :type info: dict

   :returns: Two tensors yielding spatial coordinates
   :rtype: Sequence[torch.Tensor]


.. py:function:: listmode_to_sinogram(detector_ids, info, weights = None, normalization = False, tof_meta = None)

   Converts PET listmode data to sinogram

   :param detector_ids: Listmode detector ID data
   :type detector_ids: torch.Tensor
   :param info: PET geometry information dictionary
   :type info: dict
   :param weights: Binning weights for each listmode event. Defaults to None.
   :type weights: torch.Tensor, optional
   :param normalization: Whether or not this is a normalization sinogram (need to do some extra steps). Defaults to False.
   :type normalization: bool, optional
   :param tof_meta: PET TOF metadata. Defaults to None.
   :type tof_meta: PETTOFMeta, optional

   :returns: PET sinogram
   :rtype: torch.Tensor


.. py:function:: _listmodeTOF_to_sinogramTOF(detector_ids, info, tof_meta, weights = None)

   Helper function to ``listmode_to_sinogram`` for TOF data

   :param detector_ids: Listmode detector ID data
   :type detector_ids: torch.Tensor
   :param info: PET geometry information dictionary
   :type info: dict
   :param weights: Binning weights for each listmode event. Defaults to None.
   :type weights: torch.Tensor, optional
   :param tof_meta: PET TOF metadata. Defaults to None.
   :type tof_meta: PETTOFMeta, optional

   :returns: PET TOF sinogram
   :rtype: torch.Tensor


.. py:function:: get_detector_ids_from_trans_axial_ids(ids_trans_crystal, ids_trans_submodule, ids_trans_module, ids_trans_rsector, ids_axial_crystal, ids_axial_submodule, ids_axial_module, ids_axial_rsector, info)

   Obtain detector IDs from individual part IDs

   :param ids_trans_crystal: Transaxial crystal IDs
   :type ids_trans_crystal: torch.Tensor
   :param ids_trans_submodule: Transaxial submodule IDs
   :type ids_trans_submodule: torch.Tensor
   :param ids_trans_module: Transaxial module IDs
   :type ids_trans_module: torch.Tensor
   :param ids_trans_rsector: Transaxial rsector IDs
   :type ids_trans_rsector: torch.Tensor
   :param ids_axial_crystal: Axial crystal IDs
   :type ids_axial_crystal: torch.Tensor
   :param ids_axial_submodule: Axial submodule IDs
   :type ids_axial_submodule: torch.Tensor
   :param ids_axial_module: Axial module IDs
   :type ids_axial_module: torch.Tensor
   :param ids_axial_rsector: Axial rsector IDs
   :type ids_axial_rsector: torch.Tensor
   :param info: PET geometry information dictionary
   :type info: dict

   :returns: Tensor containing (spatial) detector IDs
   :rtype: torch.Tensor


.. py:function:: get_axial_trans_ids_from_info(info, return_combinations = False, sort_by_detector_ids = False)

   Get axial and transaxial IDs corresponding to each crystal in the scanner

   :param info: PET geometry information dictionary
   :type info: dict
   :param return_combinations: Whether or not to return all possible combinations (crystal pairs). Defaults to False.
   :type return_combinations: bool, optional
   :param sort_by_detector_ids: Whether or not to sort by increasing detector IDs. Defaults to False.
   :type sort_by_detector_ids: bool, optional

   :returns: IDs corresponding to axial/transaxial components of each part
   :rtype: Sequence[torch.Tensor]


.. py:function:: get_scanner_LUT(info)

   Obtains scanner lookup table (gives x/y/z coordinates for each detector ID)

   :param info: PET geometry information dictionary
   :type info: dict

   :returns: Lookup table
   :rtype: torch.Tensor[N_detectors, 3]


.. py:function:: sinogram_to_listmode(detector_ids, sinogram, info)

   Obtains listmode data from a sinogram at the given detector IDs

   :param detector_ids: Detector IDs at which to obtain listmode data
   :type detector_ids: torch.Tensor
   :param sinogram: PET sinogram
   :type sinogram: torch.Tensor
   :param info: PET geometry information dictionary
   :type info: dict

   :returns: Listmode data
   :rtype: torch.Tensor


.. py:function:: smooth_randoms_sinogram(sinogram_random, info, sigma_r = 4, sigma_theta = 4, sigma_z = 4, kernel_size_r = 21, kernel_size_theta = 21, kernel_size_z = 21)

   Smooths a PET randoms sinogram using a Gaussian filter in the r, theta, and z direction. Rebins the sinogram into (r,theta,z1,z2) before blurring (same blurring applied to z1 and z2)

   :param sinogram_random: PET sinogram of randoms
   :type sinogram_random: torch.Tensor
   :param info: PET geometry information dictionary
   :type info: dict
   :param sigma_r: Blurring (in pixel size) in r direction. Defaults to 4.
   :type sigma_r: float, optional
   :param sigma_theta: Blurring (in pixel size) in r direction. Defaults to 4.
   :type sigma_theta: float, optional
   :param sigma_z: Blurring (in pixel size) in z direction. Defaults to 4.
   :type sigma_z: float, optional
   :param kernel_size_r: Kernel size in r direction. Defaults to 21.
   :type kernel_size_r: int, optional
   :param kernel_size_theta: Kernel size in theta direction. Defaults to 21.
   :type kernel_size_theta: int, optional
   :param kernel_size_z: Kernel size in z1/z2 diretions. Defaults to 21.
   :type kernel_size_z: int, optional

   :returns: Smoothed randoms sinogram
   :rtype: torch.Tensor


.. py:function:: randoms_sinogram_to_sinogramTOF(sinogram_random, tof_meta, coincidence_timing_width)

   Converts a non-TOF randoms sinogram to a TOF randoms sinogram.

   :param sinogram_random: Randoms sinogram (non-TOF)
   :type sinogram_random: torch.Tenor
   :param tof_meta: PET TOF metadata
   :type tof_meta: PETTOFMeta
   :param coincidence_timing_width: Coincidence timing width used for the acceptance of coincidence events
   :type coincidence_timing_width: float

   :returns: Randoms sinogram (TOF)
   :rtype: torch.Tensor


