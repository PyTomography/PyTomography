:py:mod:`pytomography.utils.sss`
================================

.. py:module:: pytomography.utils.sss


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.utils.sss.total_compton_cross_section
   pytomography.utils.sss.photon_energy_after_compton_scatter_511kev
   pytomography.utils.sss.diff_compton_cross_section
   pytomography.utils.sss.detector_efficiency
   pytomography.utils.sss.tof_efficiency
   pytomography.utils.sss.get_sample_scatter_points
   pytomography.utils.sss.get_sample_detector_ids
   pytomography.utils.sss.compute_sss_sparse_sinogram
   pytomography.utils.sss.compute_sss_sparse_sinogram_TOF
   pytomography.utils.sss.interpolate_sparse_sinogram
   pytomography.utils.sss.scale_estimated_scatter
   pytomography.utils.sss.get_sss_scatter_estimate



.. py:function:: total_compton_cross_section(energy)

   Computes the total compton cross section of interaction :math:`\sigma` at the given photon energies

   :param energy: Energies of photons considered
   :type energy: torch.Tensor

   :returns: Cross section at each corresponding energy
   :rtype: torch.Tensor


.. py:function:: photon_energy_after_compton_scatter_511kev(cos_theta)

   Computes the corresponding photon energy after a 511keV photon scatters

   :param cos_theta: Angle of scatter
   :type cos_theta: torch.Tensor

   :returns: Photon energy after scattering.
   :rtype: torch.Tensor


.. py:function:: diff_compton_cross_section(cos_theta, energy)

   Computes the differential cross section :math:`d\sigma/d\omega` at given photon energies and scattering angles

   :param cos_theta: Cosine of the scattering angle
   :type cos_theta: torch.Tensor
   :param energy: Energy of the incident photon before scattering
   :type energy: torch.Tensor

   :returns: Differential compton cross section
   :rtype: torch.Tensor


.. py:function:: detector_efficiency(scatter_energy, energy_resolution = 0.15, energy_threshhold = 430)

   Computes the probability a photon of given energy is detected within the energy limits of the detector

   :param scatter_energy: Energy of the photon impinging the detector
   :type scatter_energy: torch.Tensor
   :param energy_resolution: Energy resolution of the crystals (represented as a fraction of 511keV). This is the uncertainty of energy measurements. Defaults to 0.15.
   :type energy_resolution: float, optional
   :param energy_threshhold: Lower limit of energies detected by the crystal which are registered as events. Defaults to 430.
   :type energy_threshhold: float, optional

   :returns: Probability that the photon gets detected
   :rtype: torch.Tensor


.. py:function:: tof_efficiency(offset, tof_bins_dense_centers, tof_meta)

   Computes the probability that a coincidence event with timing difference offset is detected in each of the TOF bins specified by ``tof_bins_dense_centers``.

   :param offset: Timing offset (in spatial units) between a coincidence event. When this function is used in SSS, ``offset`` has shape :math:`(N_{TOF}, N_{coinc})` where :math:`N_{coinc}` is the number of coincidence events considered, and :math:`N_{TOF}` is the number of time of flight bins in the sinogram.
   :type offset: torch.Tensor
   :param tof_bins_dense_centers: The centers of each of the dense TOF bins. These are seperate from the TOF bins of the sinogram: these TOF bins correspond to the partioning of the integrals in Watson(2007) Equation 2. When used in SSS, this tensor has shape :math:`(N_{coinc}, N_{denseTOF})` where :math:`N_{denseTOF}` are the number of dense TOF bins considered.
   :type tof_bins_dense_centers: torch.Tensor
   :param tof_meta: TOF metadata for the sinogram
   :type tof_meta: PETTOFMeta

   :returns: Relative probability of detecting the event at offset ``offset`` in each of the ``tof_bins_dense_centers`` locations.
   :rtype: torch.Tensor


.. py:function:: get_sample_scatter_points(attenuation_map, stepsize = 4, attenuation_cutoff = 0.004)

   Selects a subset of points in the attenuation map used as scatter points.

   :param attenuation_map: Attenuation map
   :type attenuation_map: torch.Tensor
   :param stepsize: Stepsize in x/y/z between sampled points. Defaults to 4.
   :type stepsize: float, optional
   :param attenuation_cutoff: Only consider points above this threshhold. Defaults to 0.004.
   :type attenuation_cutoff: float, optional

   :returns: Tensor of coordinates
   :rtype: torch.Tensor


.. py:function:: get_sample_detector_ids(proj_meta, sinogram_interring_stepsize = 4, sinogram_intraring_stepsize = 4)

   Selects a subset of detector IDs in the PET scanner used for obtaining scatter estimates in the sparse sinogram

   :param proj_meta: PET projection metadata (sinogram/listmode)
   :type proj_meta: ProjMeta
   :param sinogram_interring_stepsize: Axial stepsize between rings. Defaults to 4.
   :type sinogram_interring_stepsize: int, optional
   :param sinogram_intraring_stepsize: Stepsize of crystals within a given ring. Defaults to 4.
   :type sinogram_intraring_stepsize: int, optional

   :returns: Crystal index within ring, ring index, and detector ID pairs corresponding to all sampled LORs.
   :rtype: Sequence[torch.Tensor, torch.Tensor, torch.Tensor]


.. py:function:: compute_sss_sparse_sinogram(object_meta, proj_meta, pet_image, attenuation_image, image_stepsize = 4, attenuation_cutoff = 0.004, sinogram_interring_stepsize = 4, sinogram_intraring_stepsize = 4)

   Generates a sparse single scatter simulation sinogram for non-TOF PET data.

   :param object_meta: Object metadata corresponding to reconstructed PET image used in the simulation
   :type object_meta: ObjectMeta
   :param proj_meta: Projection metadata specifying the details of the PET scanner
   :type proj_meta: ProjMeta
   :param pet_image: PET image used to estimate the scatter
   :type pet_image: torch.Tensor
   :param attenuation_image: Attenuation map used in scatter simulation
   :type attenuation_image: torch.Tensor
   :param image_stepsize: Stepsize in x/y/z between sampled scatter points. Defaults to 4.
   :type image_stepsize: int, optional
   :param attenuation_cutoff: Only consider points above this threshhold. Defaults to 0.004.
   :type attenuation_cutoff: float, optional
   :param sinogram_interring_stepsize: Axial stepsize between rings. Defaults to 4.
   :type sinogram_interring_stepsize: int, optional
   :param sinogram_intraring_stepsize: Stepsize of crystals within a given ring. Defaults to 4.
   :type sinogram_intraring_stepsize: int, optional

   :returns: Estimated sparse single scatter simulation sinogram.
   :rtype: torch.Tensor


.. py:function:: compute_sss_sparse_sinogram_TOF(object_meta, proj_meta, pet_image, attenuation_image, tof_meta, image_stepsize = 4, attenuation_cutoff = 0.004, sinogram_interring_stepsize = 4, sinogram_intraring_stepsize = 4, num_dense_tof_bins = 25)

   Generates a sparse single scatter simulation sinogram for TOF PET data.

   :param object_meta: Object metadata corresponding to reconstructed PET image used in the simulation
   :type object_meta: ObjectMeta
   :param proj_meta: Projection metadata specifying the details of the PET scanner
   :type proj_meta: ProjMeta
   :param pet_image: PET image used to estimate the scatter
   :type pet_image: torch.Tensor
   :param attenuation_image: Attenuation map used in scatter simulation
   :type attenuation_image: torch.Tensor
   :param tof_meta: PET TOF Metadata corresponding to the sinogram estimate
   :type tof_meta: PETTOFMeta
   :param attenuation_image: Attenuation map used in scatter simulation
   :type attenuation_image: torch.Tensor
   :param image_stepsize: Stepsize in x/y/z between sampled scatter points. Defaults to 4.
   :type image_stepsize: int, optional
   :param attenuation_cutoff: Only consider points above this threshhold. Defaults to 0.004.
   :type attenuation_cutoff: float, optional
   :param sinogram_interring_stepsize: Axial stepsize between rings. Defaults to 4.
   :type sinogram_interring_stepsize: int, optional
   :param sinogram_intraring_stepsize: Stepsize of crystals within a given ring. Defaults to 4.
   :type sinogram_intraring_stepsize: int, optional
   :param num_dense_tof_bins: Number of dense TOF bins used when partioning the emission integrals (these integrals must be partioned for TOF-based estimation). Defaults to 25.
   :type num_dense_tof_bins: int, optional

   :returns: Estimated sparse single scatter simulation sinogram.
   :rtype: torch.Tensor


.. py:function:: interpolate_sparse_sinogram(scatter_sinogram_sparse, proj_meta, idx_intraring, idx_ring)

   Interpolates a sparse SSS sinogram estimate using linear interpolation on all oblique planes.

   :param scatter_sinogram_sparse: Estimated sparse SSS sinogram from the ``compute_sss_sparse_sinogram`` or ``compute_sss_sparse_sinogram_TOF`` functions
   :type scatter_sinogram_sparse: torch.Tensor
   :param proj_meta: PET projection metadata corresponding to the sinogram
   :type proj_meta: ProjMeta
   :param idx_intraring: Intraring indices corresponding to non-zero locations of the sinogram (obtained via the ``get_sample_detector_ids`` function)
   :type idx_intraring: torch.Tensor
   :param idx_ring: Interring indices corresponding to non-zero locations of the sinogram (obtained via the ``get_sample_detector_ids`` function)
   :type idx_ring: torch.Tensor

   :returns: Interpolated SSS sinogram
   :rtype: torch.Tensor


.. py:function:: scale_estimated_scatter(proj_scatter, system_matrix, proj_data, attenuation_image, attenuation_image_cutoff = 0.004, sinogram_random = None)

   Given an interpolated (but unscaled) SSS sinogram/listmode, scales the scatter estimate by considering back projection of masked data. The mask corresponds to all locations below a certain attenuation value, where it is likely that all detected events are purely due to scatter.

   :param proj_scatter: Estimated (but unscaled) SSS data.
   :type proj_scatter: torch.Tensor
   :param system_matrix: PET system matrix
   :type system_matrix: SystemMatrix
   :param proj_data: PET projection data corresponding to all detected events
   :type proj_data: torch.Tensor
   :param attenuation_image: Attenuation map
   :type attenuation_image: torch.Tensor
   :param attenuation_image_cutoff: Mask considers regions below this value (forward projected). In particular, the attenuation map is masked above this value, then forward projected. Regions equal to zero in the forward projection are considered for the mask. This allows for hollow regions within the attenuation map to still be considered. Defaults to 0.004.
   :type attenuation_image_cutoff: float, optional
   :param sinogram_random: Projection data of estimated random events. Defaults to None.
   :type sinogram_random: torch.Tensor | None, optional

   :returns: Scaled SSS projection data (sinogram/listmode).
   :rtype: torch.Tensor


.. py:function:: get_sss_scatter_estimate(object_meta, proj_meta, pet_image, attenuation_image, system_matrix, proj_data = None, image_stepsize = 4, attenuation_cutoff = 0.004, sinogram_interring_stepsize = 4, sinogram_intraring_stepsize = 4, sinogram_random = None, tof_meta = None, num_dense_tof_bins = 25)

   Main function used to get SSS scatter estimation during PET reconstruction

   :param object_meta: Object metadata corresponding to ``pet_image``.
   :type object_meta: ObjectMeta
   :param proj_meta: Projection metadata corresponding to ``proj_data``.
   :type proj_meta: ProjMeta
   :param pet_image: Reconstructed PET image used to get SSS estimate
   :type pet_image: torch.Tensor
   :param attenuation_image: Attenuation map corresponding to PET image
   :type attenuation_image: torch.Tensor
   :param system_matrix: PET system matrix
   :type system_matrix: SystemMatrix
   :param proj_data: All measured coincident events (sinogram/listmode). If None, then assumes listmode (coincidence events stored in ``proj_meta``).
   :type proj_data: torch.Tensor | None
   :param image_stepsize: Spacing between points in object space used to obtain initial sparse sinogram estimate. Defaults to 4.
   :type image_stepsize: int, optional
   :param attenuation_cutoff: Only consider point located at attenuation values above this value as scatter points. Defaults to 0.004.
   :type attenuation_cutoff: float, optional
   :param sinogram_interring_stepsize: Sinogram interring spacing for initial sparse sinogram estimate. Defaults to 4.
   :type sinogram_interring_stepsize: int, optional
   :param sinogram_intraring_stepsize: Sinogram intraring spacing for initial sparse sinogram estimate. Defaults to 4.
   :type sinogram_intraring_stepsize: int, optional
   :param sinogram_random: Estimated randoms. Defaults to None.
   :type sinogram_random: torch.Tensor | None, optional
   :param tof_meta: TOFMetadata corresponding to ``proj_data`` (if TOF is considered). Defaults to None.
   :type tof_meta: PETTOFMeta, optional
   :param num_dense_tof_bins: Number of dense TOF bins to use for partioning emission integrals when performing a TOF estimate. This is seperate from TOF bins used in the PET data. Defaults to 25.
   :type num_dense_tof_bins: int, optional

   :returns: Estimated SSS projection data (sinogram/listmode)
   :rtype: torch.Tensor


