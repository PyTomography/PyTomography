:py:mod:`pytomography.projectors.CT.ct_conebeam_flatpanel_system_matrix`
========================================================================

.. py:module:: pytomography.projectors.CT.ct_conebeam_flatpanel_system_matrix


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.projectors.CT.ct_conebeam_flatpanel_system_matrix.CTConeBeamFlatPanelSystemMatrix



Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.projectors.CT.ct_conebeam_flatpanel_system_matrix.get_discrete_ramp_FFT
   pytomography.projectors.CT.ct_conebeam_flatpanel_system_matrix.FBP_filter



.. py:function:: get_discrete_ramp_FFT(n)


.. py:function:: FBP_filter(proj, device=pytomography.device)


.. py:class:: CTConeBeamFlatPanelSystemMatrix(object_meta, proj_meta, N_splits = 1, device = pytomography.device)

   Bases: :py:obj:`pytomography.projectors.SystemMatrix`

   System matrix for a cone beam CT system with a flat detector panel. Backprojection supports FBP, but only for non-helical (i.e. fixed z) geometries.

   :param object_meta: Metadata for object space
   :type object_meta: ObjectMeta
   :param proj_meta: Projection metadata for the CT system
   :type proj_meta: CTConeBeamFlatPanelProjMeta
   :param N_splits: Splits up computation of forward/back projection to save GPU memory. Defaults to 1.
   :type N_splits: int, optional
   :param device: Device on which projections are output. Defaults to pytomography.device.
   :type device: str, optional

   .. py:method:: _get_FBP_scale()


   .. py:method:: _get_FBP_preweight(idx)


   .. py:method:: _get_FBP_postweight(idx)


   .. py:method:: set_n_subsets(n_subsets)

      Returns a list where each element consists of an array of indices corresponding to a partitioned version of the projections.

      :param n_subsets: Number of subsets to partition the projections into
      :type n_subsets: int

      :returns: List of arrays where each array corresponds to the projection indices of a particular subset.
      :rtype: list


   .. py:method:: get_projection_subset(projections, subset_idx)

      Obtains subsampled projections :math:`g_m` corresponding to subset index :math:`m`. CT conebeam flat panel partitions projections based on angle.

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

      Computes the normalization factor :math:`H^T 1`

      :param subset_idx: Subset index for ths sinogram. If None, considers all elements. Defaults to None..
      :type subset_idx: int, optional

      :returns: Normalization factor.
      :rtype: torch.Tensor


   .. py:method:: forward(object, subset_idx = None, FBP_post_weight = None, projection_type='matched')

      Computes forward projection

      :param object: Object to be forward projected
      :type object: torch.Tensor
      :param subset_idx: Subset index :math:`m` of the projection. If None, then projects to entire projection space. Defaults to None.
      :type subset_idx: int | None, optional
      :param FBP_post_weight: _description_. Defaults to None.
      :type FBP_post_weight: torch.Tensor, optional
      :param projection_type: Type of forward projection to use; defaults to mathced. (For implementing the adjoint of FBP, we need the option of using FBP weights in the forward projection).
      :type projection_type: str

      :returns: Projections corresponding to :math:`\int \mu dx` along all LORs.
      :rtype: torch.Tensor


   .. py:method:: backward(proj, subset_idx = None, projection_type='matched')

      Computes back projection.

      :param proj: Projections to be back projected
      :type proj: torch.Tensor
      :param subset_idx: Subset index :math:`m` of the projection. Defaults to None.
      :type subset_idx: int | None, optional
      :param projection_type: Type of back projection to use. To use with filtered back projection, use ``'FBP'``, which weights all LORs accordingly for this geometry. Defaults to ``'matched'``.
      :type projection_type: str, optional

      :returns: _description_
      :rtype: torch.Tensor



