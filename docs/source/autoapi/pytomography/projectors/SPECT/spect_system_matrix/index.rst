:py:mod:`pytomography.projectors.SPECT.spect_system_matrix`
===========================================================

.. py:module:: pytomography.projectors.SPECT.spect_system_matrix


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.projectors.SPECT.spect_system_matrix.SPECTSystemMatrix
   pytomography.projectors.SPECT.spect_system_matrix.SPECTCompleteSystemMatrix




.. py:class:: SPECTSystemMatrix(obj2obj_transforms, proj2proj_transforms, object_meta, proj_meta, object_initial_based_on_camera_path = False)

   Bases: :py:obj:`pytomography.projectors.system_matrix.SystemMatrix`

   System matrix for SPECT imaging implemented using the rotate+sum technique.

   :param obj2obj_transforms: Sequence of object mappings that occur before forward projection.
   :type obj2obj_transforms: Sequence[Transform]
   :param proj2proj_transforms: Sequence of proj mappings that occur after forward projection.
   :type proj2proj_transforms: Sequence[Transform]
   :param object_meta: SPECT Object metadata.
   :type object_meta: SPECTObjectMeta
   :param proj_meta: SPECT projection metadata.
   :type proj_meta: SPECTProjMeta
   :param object_initial_based_on_camera_path: Whether or not to initialize the object estimate based on the camera path; this sets voxels to zero that are outside the SPECT camera path. Defaults to False.
   :type object_initial_based_on_camera_path: bool

   .. py:method:: _get_object_initial(device=None)

      Returns an initial object estimate used in reconstruction algorithms. By default, this is a tensor of ones with the same shape as the object metadata.

      :returns: Initial object used in reconstruction algorithm.
      :rtype: torch.Tensor


   .. py:method:: set_n_subsets(n_subsets)

      Sets the subsets for this system matrix given ``n_subsets`` total subsets.

      :param n_subsets: number of subsets used in OSEM
      :type n_subsets: int


   .. py:method:: get_projection_subset(projections, subset_idx)

      Gets the subset of projections :math:`g_m` corresponding to index :math:`m`.

      :param projections: full projections :math:`g`
      :type projections: torch.tensor
      :param subset_idx: subset index :math:`m`
      :type subset_idx: int

      :returns: subsampled projections :math:`g_m`
      :rtype: torch.tensor


   .. py:method:: get_weighting_subset(subset_idx)

      Computes the relative weighting of a given subset (given that the projection space is reduced). This is used for scaling parameters relative to :math:`H_m^T 1` in reconstruction algorithms, such as prior weighting :math:`\beta`

      :param subset_idx: Subset index
      :type subset_idx: int

      :returns: Weighting for the subset.
      :rtype: float


   .. py:method:: compute_normalization_factor(subset_idx = None)

      Function used to get normalization factor :math:`H^T_m 1` corresponding to projection subset :math:`m`.

      :param subset_idx: Index of subset. If none, then considers all projections. Defaults to None.
      :type subset_idx: int | None, optional

      :returns: normalization factor :math:`H^T_m 1`
      :rtype: torch.Tensor


   .. py:method:: forward(object, subset_idx = None)

      Applies forward projection to ``object`` for a SPECT imaging system.

      :param object: The object to be forward projected
      :type object: torch.tensor[Lx, Ly, Lz]
      :param subset_idx: Only uses a subset of angles :math:`g_m` corresponding to the provided subset index :math:`m`. If None, then defaults to the full projections :math:`g`.
      :type subset_idx: int, optional

      :returns: forward projection estimate :math:`g_m=H_mf`
      :rtype: torch.tensor


   .. py:method:: backward(proj, subset_idx = None)

      Applies back projection to ``proj`` for a SPECT imaging system.

      :param proj: projections :math:`g` which are to be back projected
      :type proj: torch.tensor
      :param subset_idx: Only uses a subset of angles :math:`g_m` corresponding to the provided subset index :math:`m`. If None, then defaults to the full projections :math:`g`.
      :type subset_idx: int, optional
      :param return_norm_constant: Whether or not to return :math:`H_m^T 1` along with back projection. Defaults to 'False'.
      :type return_norm_constant: bool

      :returns: the object :math:`\hat{f} = H_m^T g_m` obtained via back projection.
      :rtype: torch.tensor



.. py:class:: SPECTCompleteSystemMatrix(object_meta, proj_meta, attenuation_map, object_meta_amap, psf_kernel, store_system_matrix=None, mask_based_on_attenuation=False, photopeak=None, n_parallel=1)

   Bases: :py:obj:`SPECTSystemMatrix`

   Class presently under construction.


   .. py:method:: _get_proj_positions(idx)


   .. py:method:: _get_object_positions()


   .. py:method:: _compute_system_matrix_components(idx)


   .. py:method:: _compute_projections_mask(photopeak)


   .. py:method:: forward(object, subset_idx = None)

      Applies forward projection to ``object`` for a SPECT imaging system.

      :param object: The object to be forward projected
      :type object: torch.tensor[Lx, Ly, Lz]
      :param subset_idx: Only uses a subset of angles :math:`g_m` corresponding to the provided subset index :math:`m`. If None, then defaults to the full projections :math:`g`.
      :type subset_idx: int, optional

      :returns: forward projection estimate :math:`g_m=H_mf`
      :rtype: torch.tensor


   .. py:method:: backward(proj, subset_idx = None)

      Applies back projection to ``proj`` for a SPECT imaging system.

      :param proj: projections :math:`g` which are to be back projected
      :type proj: torch.tensor
      :param subset_idx: Only uses a subset of angles :math:`g_m` corresponding to the provided subset index :math:`m`. If None, then defaults to the full projections :math:`g`.
      :type subset_idx: int, optional
      :param return_norm_constant: Whether or not to return :math:`H_m^T 1` along with back projection. Defaults to 'False'.
      :type return_norm_constant: bool

      :returns: the object :math:`\hat{f} = H_m^T g_m` obtained via back projection.
      :rtype: torch.tensor



