:py:mod:`pytomography.io.PET.gate`
==================================

.. py:module:: pytomography.io.PET.gate


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.PET.gate.get_aligned_attenuation_map
   pytomography.io.PET.gate.get_detector_info
   pytomography.io.PET.gate.get_axial_trans_ids_from_ROOT
   pytomography.io.PET.gate.get_detector_ids_from_root
   pytomography.io.PET.gate.get_symmetry_histogram_from_ROOTfile
   pytomography.io.PET.gate.get_symmetry_histogram_all_combos
   pytomography.io.PET.gate.get_normalization_weights_cylinder_calibration
   pytomography.io.PET.gate.get_norm_sinogram_from_listmode_data
   pytomography.io.PET.gate.get_norm_sinogram_from_root_data
   pytomography.io.PET.gate.get_sinogram_from_root_data
   pytomography.io.PET.gate.get_radius
   pytomography.io.PET.gate.get_angle
   pytomography.io.PET.gate.remove_events_out_of_bounds
   pytomography.io.PET.gate.get_attenuation_map_nifti



.. py:function:: get_aligned_attenuation_map(headerfile, object_meta)

   Returns an aligned attenuation map in units of inverse mm for reconstruction. This assumes that the attenuation map shares the same center point with the reconstruction space.

   :param headerfile: Filepath to the header file of the attenuation map
   :type headerfile: str
   :param object_meta: Object metadata providing spatial information about the reconstructed dimensions.
   :type object_meta: ObjectMeta

   :returns: Aligned attenuation map
   :rtype: torch.Tensor


.. py:function:: get_detector_info(path, init_volume_name = 'crystal', mean_interaction_depth = 0, min_rsector_difference = 0)

   Generates detector geometry information dictionary from GATE macro file

   :param path: Path to GATE macro file that defines geometry: should end in ".mac"
   :type path: str
   :param init_volume_name: Initial volume name in the GATE file. Defaults to 'crystal'.
   :type init_volume_name: str, optional
   :param mean_interaction_depth: Mean interaction depth of photons within crystal. Defaults to 0.
   :type mean_interaction_depth: float, optional
   :param min_rsector_difference: Minimum r_sector difference for retained events. Defaults to 0.
   :type min_rsector_difference: int, optional

   :returns: PET geometry information dictionary
   :rtype: dict


.. py:function:: get_axial_trans_ids_from_ROOT(f, info, j = None, substr = 'Coincidences')

   Obtain transaxial and axial IDS (for crystals, submodules, modules, and rsectors) corresponding to each listmode event in an opened ROOT file

   :param f: Opened ROOT file
   :type f: object
   :param info: PET geometry information dictionary
   :type info: dict
   :param j: Which of the detectors to consider in a coincidence event OR which detector to consider for a single (None). Defaults to None.
   :type j: int, optional
   :param substr: Whether to consider coincidences or singles. Defaults to 'Coincidences'.
   :type substr: str, optional

   :returns: Sequence of IDs (transaxial/axial) for all components (crystals, submodules, modules, and rsectors)
   :rtype: Sequence[torch.Tensor]


.. py:function:: get_detector_ids_from_root(paths, info, tof_meta=None, substr = 'Coincidences', include_randoms = True, include_scatters = True, randoms_only = False, scatters_only = False)

   Obtain detector IDs corresponding to each listmode event in a set of ROOT files

   :param paths: List of ROOT files to consider
   :type paths: Sequence[str]
   :param info: PET geometry information dictionary
   :type info: dict
   :param tof_meta: PET time of flight metadata for binning. If none, then TOF is not considered Defaults to None.
   :type tof_meta: PETTOFMeta, optional
   :param substr: Name of events to consider in the ROOT file. Defaults to 'Coincidences'.
   :type substr: str, optional
   :param include_randoms: Whether or not to include random events in the returned listmode events. Defaults to True.
   :type include_randoms: bool, optional
   :param include_scatters: Whether or not to include scatter events in the returned listmode events. Defaults to True.
   :type include_scatters: bool, optional
   :param randoms_only: Flag to return only random events. Defaults to False.
   :type randoms_only: bool, optional
   :param scatters_only: Flag to return only scatter events. Defaults to False.
   :type scatters_only: bool, optional

   :returns: Tensor of shape [N_events,2] (non-TOF) or [N_events,3] (TOF)
   :rtype: torch.Tensor


.. py:function:: get_symmetry_histogram_from_ROOTfile(f, info, substr = 'Coincidences', include_randoms = True)

   Obtains a histogram that exploits symmetries when computing normalization factors from calibration ROOT scans

   :param f: Opened ROOT file
   :type f: object
   :param info: PET geometry information dictionary
   :type info: dict
   :param substr: Name of events to consider in ROOT file. Defaults to 'Coincidences'.
   :type substr: str, optional
   :param include_randoms: Whether or not to include random events from data. Defaults to True.
   :type include_randoms: bool, optional

   :returns: Symmetry histogram
   :rtype: torch.Tensor


.. py:function:: get_symmetry_histogram_all_combos(info)

   Obtains the symmetry histogram for detector sensitivity corresponding to all possible detector pair combinations

   :param info: PET geometry information dictionary
   :type info: dict

   :returns: Histogram corresponding to all possible detector pair combinations. This simply counts the number of detector pairs in each bin of the histogram.
   :rtype: torch.Tensor


.. py:function:: get_normalization_weights_cylinder_calibration(paths, info, cylinder_radius, include_randoms = True)

   Function to get sensitivty factor from a cylindrical calibration phantom

   :param paths: List of paths corresponding to calibration scan
   :type paths: Sequence[str]
   :param info: PET geometry information dictionary
   :type info: dict
   :param cylinder_radius: Radius of cylindrical phantom used in scan
   :type cylinder_radius: float
   :param include_randoms: Whether or not to include random events from the cylinder calibration. Defaults to True.
   :type include_randoms: bool, optional

   :returns: Sensitivty factor for all possible detector combinations
   :rtype: torch.tensor


.. py:function:: get_norm_sinogram_from_listmode_data(weights_sensitivity, info)

   Obtains normalization "sensitivty" sinogram from listmode data

   :param weights_sensitivity: Sensitivty weight corresponding to all possible detector pairs
   :type weights_sensitivity: torch.Tensor
   :param info: PET geometry information dictionary
   :type info: dict

   :returns: PET sinogram
   :rtype: torch.Tensor


.. py:function:: get_norm_sinogram_from_root_data(normalization_paths, info, cylinder_radius, include_randoms = True)

   Obtain normalization "sensitivity" sinogram directly from ROOT files

   :param normalization_paths: Paths to all ROOT files corresponding to calibration scan
   :type normalization_paths: Sequence[str]
   :param info: PET geometry information dictionary
   :type info: dict
   :param cylinder_radius: Radius of cylinder used in calibration scan
   :type cylinder_radius: float
   :param include_randoms: Whether or not to include randoms in loaded data. Defaults to True.
   :type include_randoms: bool, optional

   :returns: PET sinogram
   :rtype: torch.Tensor


.. py:function:: get_sinogram_from_root_data(paths, info, include_randoms = True, include_scatters = True, randoms_only = False, scatters_only = False)

   Get PET sinogram directly from ROOT data

   :param paths: GATE generated ROOT files
   :type paths: Sequence[str]
   :param info: PET geometry information dictionary
   :type info: dict
   :param include_randoms: Whether or not to include random events in the sinogram. Defaults to True.
   :type include_randoms: bool, optional
   :param include_scatters: Whether or not to include scatter events in the sinogram. Defaults to True.
   :type include_scatters: bool, optional
   :param randoms_only: Flag for only binning randoms. Defaults to False.
   :type randoms_only: bool, optional
   :param scatters_only: Flag for only binning scatters. Defaults to False.
   :type scatters_only: bool, optional

   :returns: PET sinogram
   :rtype: torch.Tensor


.. py:function:: get_radius(detector_ids, scanner_LUT)

   Gets the radial position of all LORs

   :param detector_ids: Detector ID pairs corresponding to LORs
   :type detector_ids: torch.tensor
   :param scanner_LUT: scanner look up table
   :type scanner_LUT: torch.tensor

   :returns: radii of all detector ID pairs provided
   :rtype: torch.tensor


.. py:function:: get_angle(detector_ids, scanner_LUT)

   Gets the angular position of all LORs

   :param detector_ids: Detector ID pairs corresponding to LORs
   :type detector_ids: torch.tensor
   :param scanner_LUT: scanner look up table
   :type scanner_LUT: torch.tensor

   :returns: angle of all detector ID pairs provided
   :rtype: torch.tensor


.. py:function:: remove_events_out_of_bounds(detector_ids, scanner_LUT, object_meta)

   Removes all detected LORs outside of the reconstruced volume given by ``object_meta``.

   :param detector_ids: :math:`N \times 2` (non-TOF) or :math:`N \times 3` (TOF) tensor that provides detector ID pairs (and TOF bin) for coincidence events.
   :type detector_ids: torch.tensor
   :param scanner_LUT: scanner lookup table that provides spatial coordinates for all detector ID pairs
   :type scanner_LUT: torch.tensor
   :param object_meta: object metadata providing the region of reconstruction
   :type object_meta: ObjectMeta

   :returns: all detector ID pairs corresponding to coincidence events
   :rtype: torch.tensor


.. py:function:: get_attenuation_map_nifti(path, object_meta)


