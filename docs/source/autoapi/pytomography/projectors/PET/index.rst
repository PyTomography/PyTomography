:py:mod:`pytomography.projectors.PET`
=====================================

.. py:module:: pytomography.projectors.PET


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   pet_sinogram_system_matrix/index.rst
   petlm_system_matrix/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.projectors.PET.PETLMSystemMatrix
   pytomography.projectors.PET.PETSinogramSystemMatrix



Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.projectors.PET.create_sinogramSM_from_LMSM



.. py:class:: PETLMSystemMatrix(object_meta, proj_meta, obj2obj_transforms = [], attenuation_map = None, scale_projection_by_sensitivity = False, N_splits = 1, device = pytomography.device)

   Bases: :py:obj:`pytomography.projectors.SystemMatrix`

   System matrix of PET list mode data. Forward projections corresponds to computing the expected counts along all LORs specified: in particular it approximates :math:`g_i = \int_{\text{LOR}_i} h(r) f(r) dr` where index :math:`i` corresponds to a particular detector pair and :math:`h(r)` is a Gaussian function that incorporates time-of-flight information (:math:`h(r)=1` for non-time-of-flight). The integral is approximated in the discrete object space using Joseph3D projections. In general, the system matrix implements two different projections, the quantity :math:`H` which projects to LORs corresponding to all detected events, and the quantity :math:`\tilde{H}` which projects to all valid LORs. The quantity :math:`H` is used for standard forward/back projection, while :math:`\tilde{H}` is used to compute the sensitivity image.

   :param object_meta: Metadata of object space, containing information on voxel size and dimensions.
   :type object_meta: SPECTObjectMeta
   :param proj_meta: PET listmode projection space metadata. This information contains the detector ID pairs of all detected events, as well as a scanner lookup table and time-of-flight metadata. In addition, this metadata contains all information regarding event weights, typically corresponding to the effects of attenuation :math:`\mu` and :math:`\eta`.
   :type proj_meta: PETLMProjMeta
   :param obj2obj_transforms: Object to object space transforms applied before forward projection and after back projection. These are typically used for PSF modeling in PET imaging.
   :type obj2obj_transforms: Sequence[Transform]
   :param attenuation_map: Attenuation map used for attenuation modeling. If provided, all weights will be scaled by detection probabilities derived from this map. Note that this scales on top of any weights provided in ``proj_meta``, so if attenuation is already accounted for there, this is not needed. Defaults to None.
   :type attenuation_map: torch.tensor[float] | None, optional
   :param scale_projection_by_sensitivity: Whether or not to scale the projections by :math:`\mu \eta`. This is not needed in reconstruction algorithms using a PoissonLogLikelihood. Defaults to False.
   :type scale_projection_by_sensitivity: bool, optional
   :param N_splits: Splits up computation of forward/back projection to save GPU memory. Defaults to 1.
   :type N_splits: int
   :param device: The device on which forward/back projection tensors are output. This is seperate from ``pytomography.device``, which handles internal computations. The reason for having the option of a second device is that the projection space may be very large, and certain GPUs may not have enough memory to store the projections. If ``device`` is not the same as ``pytomography.device``, then one must also specify the same ``device`` in any reconstruction algorithm used. Defaults to ``pytomography.device``.
   :type device: str

   .. py:method:: _get_object_initial(device=pytomography.device)

      Returns an initial object estimate used in reconstruction algorithms. By default, this is a tensor of ones with the same shape as the object metadata.

      :returns: Initial object used in image reconstruction algorithms.
      :rtype: torch.Tensor


   .. py:method:: _compute_attenuation_probability_projection(idx)

      Computes probabilities of photons being detected along an LORs corresponding to ``idx``.

      :param idx: Indices of the detector pairs.
      :type idx: torch.tensor

      :returns: The probabilities of photons being detected along the detector pairs.
      :rtype: torch.Tensor


   .. py:method:: _compute_sensitivity_projection(all_ids = True)

      Computes the sensitivty projection (when back projected, gives normalization factor)

      :param all_ids: Compute for all detector IDs. Defaults to True.
      :type all_ids: bool, optional

      :returns: Sesitivity factor for detector IDs
      :rtype: torch.Tensor


   .. py:method:: _backward_full(N_splits = 10)

      Computes full back projection :math:`\tilde{H}^T w g` where :math:`w` is the weighting specified in the projection metadata that accounts for attenuation/normalization correction. If ``proj`` ($g$) is not provided, then uses a tensor of all ones (this is used to compute the normalization factor).

      :param N_splits: Optionally splits up computation to save memory on GPU. Defaults to 10.
      :type N_splits: int, optional


   .. py:method:: set_n_subsets(n_subsets)

      Returns a list where each element consists of an array of indices corresponding to a partitioned version of the projections.

      :param n_subsets: Number of subsets to partition the projections into
      :type n_subsets: int

      :returns: List of arrays where each array corresponds to the projection indices of a particular subset.
      :rtype: list


   .. py:method:: get_projection_subset(projections, subset_idx)

      Obtains subsampled projections :math:`g_m` corresponding to subset index :math:`m`. For LM PET, its always the case that :math:`g_m=1`, but this function is still required for subsampling scatter :math:`s_m` as is required in certain reconstruction algorithms

      :param projections: total projections :math:`g`
      :type projections: torch.Tensor
      :param subset_idx: subset index :math:`m`
      :type subset_idx: int

      :returns: subsampled projections :math:`g_m`.
      :rtype: torch.Tensor


   .. py:method:: get_weighting_subset(subset_idx)

      Computes the relative weighting of a given subset (given that the projection space is reduced). This is used for scaling parameters relative to :math:`\tilde{H}_m^T 1` in reconstruction algorithms, such as prior weighting :math:`\beta`

      :param subset_idx: Subset index
      :type subset_idx: int

      :returns: Weighting for the subset.
      :rtype: float


   .. py:method:: compute_normalization_factor(subset_idx = None)

      Function called by reconstruction algorithms to get the sensitivty image :math:`\tilde{H}_m^T w`.

      :param subset_idx: Subset index :math:`m`. If none, then considers backprojection over all subsets. Defaults to None.
      :type subset_idx: int | None, optional

      :returns: Normalization factor.
      :rtype: torch.tensor


   .. py:method:: forward(object, subset_idx = None)

      Computes forward projection. In the case of list mode PET, this corresponds to the expected number of detected counts along each LOR corresponding to a particular object.

      :param object: Object to be forward projected
      :type object: torch.tensor
      :param subset_idx: Subset index :math:`m` of the projection. If None, then assumes projection to the entire projection space. Defaults to None.
      :type subset_idx: int, optional

      :returns: Projections corresponding to the expected number of counts along each LOR.
      :rtype: torch.tensor


   .. py:method:: backward(proj, subset_idx = None, return_norm_constant = False)

      Computes back projection. This corresponds to tracing a sequence of LORs into object space.

      :param proj: Projections to be back projected
      :type proj: torch.tensor
      :param subset_idx: Subset index :math:`m` of the projection. If None, then assumes projection to the entire projection space. Defaults to None.
      :type subset_idx: int, optional
      :param return_norm_constant: Whether or not to return the normalization constant: useful in reconstruction algorithms that require :math:`H_m^T 1`. Defaults to False.
      :type return_norm_constant: bool, optional

      :returns: _description_
      :rtype: torch.tensor



.. py:class:: PETSinogramSystemMatrix(object_meta, proj_meta, obj2obj_transforms = [], attenuation_map = None, sinogram_sensitivity = None, scale_projection_by_sensitivity = False, N_splits = 1, device = pytomography.device)

   Bases: :py:obj:`pytomography.projectors.SystemMatrix`

   System matrix for sinogram-based PET reconstruction.

   :param object_meta: Metadata of object space, containing information on voxel size and dimensions.
   :type object_meta: ObjectMeta
   :param proj_meta: PET sinogram projection space metadata. This information contains the scanner lookup table and time-of-flight metadata.
   :type proj_meta: PETSinogramPolygonProjMeta
   :param obj2obj_transforms: Object to object space transforms applied before forward projection and after back projection. These are typically used for PSF modeling in PET imaging. Defaults to [].
   :type obj2obj_transforms: list[Transform], optional
   :param attenuation_map: Attenuation map used for attenuation modeling. If provided, all weights will be scaled by detection probabilities derived from this map. Note that this scales on top of ``sinogram_sensitivity``, so if attenuation is already accounted for there, this is not needed. Defaults to None.
   :type attenuation_map: torch.tensor | None, optional
   :param sinogram_sensitivity: Normalization sinogram used to scale projections after forward projection. This factor may include detector normalization :math:`\eta` and/or attenuation modeling :math:`\mu`. The attenuation modeling :math:`\mu` should not be included if ``attenuation_map`` is provided as an argument to the function. Defaults to None.
   :type sinogram_sensitivity: torch.tensor | None, optional
   :param scale_projection_by_sensitivity: Whether or not to scale the projections by :math:`\mu \eta`. This is not needed in reconstruction algorithms using a PoissonLogLikelihood. Defaults to False.
   :type scale_projection_by_sensitivity: bool, optional
   :param N_splits: Splits up computation of forward/back projection to save GPU memory. Defaults to 1.
   :type N_splits: int, optional
   :param device: The device for any objects in projection space projection space (what it outputs in forward projection and what it expects for back projection). This is seperate from ``pytomography.device`` since the internal functionality may still use GPU even if this is CPU. This is used to save GPU memory since sinograms are often very large. Defaults to pytomography.device.
   :type device: str, optional

   .. py:method:: _get_xyz_sinogram_coordinates(subset_idx = None)

      Get the XYZ coordinates corresponding to the pair of crystals of the projection angle

      :param subset_idx: Subset index for ths sinogram. If None, considers all elements. Defaults to None.
      :type subset_idx: int, optional

      :returns: XYZ coordinates of crystal 1 and XYZ coordinates of crystal 2 corresponding to all elements in the sinogram.
      :rtype: Sequence[torch.Tensor, torch.Tensor]


   .. py:method:: _compute_atteunation_probability_projection(subset_idx)

      Compute the probability of a photon not being attenuated for a certain sinogram element.

      :param subset_idx: Subset index for ths sinogram.
      :type subset_idx: torch.tensor

      :returns: Probability sinogram
      :rtype: torch.tensor


   .. py:method:: _compute_sensitivity_sinogram(subset_idx = None)

      Computes the sensitivity sinogram :math:`\mu \eta` that accounts for attenuation effects and normalization effects.

      :param subset_idx: Subset index for ths sinogram. If None, considers all elements. Defaults to None..
      :type subset_idx: int, optional

      :returns: Sensitivity sinogram.
      :rtype: torch.Tensor


   .. py:method:: set_n_subsets(n_subsets)

      Returns a list where each element consists of an array of indices corresponding to a partitioned version of the projections.

      :param n_subsets: Number of subsets to partition the projections into
      :type n_subsets: int

      :returns: List of arrays where each array corresponds to the projection indices of a particular subset.
      :rtype: list


   .. py:method:: get_projection_subset(projections, subset_idx)

      Obtains subsampled projections :math:`g_m` corresponding to subset index :math:`m`. Sinogram PET partitions projections based on angle.

      :param projections: total projections :math:`g`
      :type projections: torch.Tensor
      :param subset_idx: subset index :math:`m`
      :type subset_idx: int

      :returns: subsampled projections :math:`g_m`.
      :rtype: torch.Tensor


   .. py:method:: get_weighting_subset(subset_idx)

      Computes the relative weighting of a given subset (given that the projection space is reduced). This is used for scaling parameters relative to :math:`\tilde{H}_m^T 1` in reconstruction algorithms, such as prior weighting :math:`\beta`

      :param subset_idx: Subset index
      :type subset_idx: int

      :returns: Weighting for the subset.
      :rtype: float


   .. py:method:: compute_normalization_factor(subset_idx = None)

      Computes the normalization factor :math:`H^T \mu \eta`

      :param subset_idx: Subset index for ths sinogram. If None, considers all elements. Defaults to None..
      :type subset_idx: int, optional

      :returns: Normalization factor.
      :rtype: torch.Tensor


   .. py:method:: forward(object, subset_idx = None)

      PET Sinogram forward projection

      :param object: Object to be forward projected
      :type object: torch.tensor
      :param subset_idx: Subset index for ths sinogram. If None, considers all elements. Defaults to None.
      :type subset_idx: int, optional
      :param scale_by_sensitivity: Whether or not to scale the projections by :math:`\mu \eta`. This is not necessarily needed in reconstruction algorithms. Defaults to False.
      :type scale_by_sensitivity: bool, optional

      :returns: Forward projection
      :rtype: torch.tensor


   .. py:method:: backward(proj, subset_idx = None, force_scale_by_sensitivity=False, force_nonTOF=False)

      PET Sinogram back projection

      :param proj: Sinogram to be back projected
      :type proj: torch.tensor
      :param subset_idx: Subset index for ths sinogram. If None, considers all elements. Defaults to None.
      :type subset_idx: int, optional
      :param scale_by_sensitivity: Whether or not to scale the projections by :math:`\mu \eta`. This is not necessarily needed in reconstruction algorithms. Defaults to False.
      :type scale_by_sensitivity: bool, optional
      :param force_nonTOF: Force non-TOF projection, even if TOF metadata is contained in the projection metadata. This is used for computing normalization factors (which don't depend on TOF). Defaults to False.
      :type force_nonTOF: bool, optional

      :returns: Back projection.
      :rtype: torch.tensor



.. py:function:: create_sinogramSM_from_LMSM(lm_system_matrix, device='cpu')

   Generates a sinogram system matrix from a listmode system matrix. This is used in the single scatter simulation algorithm.

   :param lm_system_matrix: A listmode PET system matrix
   :type lm_system_matrix: SystemMatrix
   :param device: The device for any objects in projection space projection space (what it outputs in forward projection and what it expects for back projection). This is seperate from ``pytomography.device`` since the internal functionality may still use GPU even if this is CPU. This is used to save GPU memory since sinograms are often very large. Defaults to pytomography.device.
   :type device: str, optional

   :returns: PET sinogram system matrix generated via a corresponding PET listmode system matrix.
   :rtype: SystemMatrix


