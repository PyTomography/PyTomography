:py:mod:`pytomography.likelihoods.poisson_log_likelihood`
=========================================================

.. py:module:: pytomography.likelihoods.poisson_log_likelihood


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.likelihoods.poisson_log_likelihood.PoissonLogLikelihood




.. py:class:: PoissonLogLikelihood(system_matrix, projections = None, additive_term = None, additive_term_variance_estimate = None)

   Bases: :py:obj:`pytomography.likelihoods.likelihood.Likelihood`

   The log-likelihood function for Poisson random variables. The likelihood is given by :math:`L(g|f) = \sum_i g_i [\ln(Hf)]_i - [Hf]_i - ...`. The :math:`...` contains terms that are not dependent on :math:`f`.

   :param system_matrix: The system matrix :math:`H` modeling the particular system whereby the projections were obtained
   :type system_matrix: SystemMatrix
   :param projections: Acquired data (assumed to have Poisson statistics).
   :type projections: torch.Tensor
   :param additive_term: Additional term added after forward projection by the system matrix. This term might include things like scatter and randoms. Defaults to None.
   :type additive_term: torch.Tensor, optional

   .. py:method:: compute_gradient(object, subset_idx = None, norm_BP_subset_method = 'subset_specific')

      Computes the gradient for the Poisson log likelihood given by :math:`\nabla_f L(g|f) =  H^T (g / Hf) - H^T 1`.

      :param object: Object :math:`f` on which the likelihood is computed
      :type object: torch.Tensor
      :param subset_idx: Specifies the subset for forward/back projection. If none, then forward/back projection is done over all subsets, and the entire projections :math:`g` are used. Defaults to None.
      :type subset_idx: int | None, optional
      :param norm_BP_subset_method: Specifies how :math:`H^T 1` is calculated when subsets are used. If 'subset_specific', then uses :math:`H_m^T 1`. If `average_of_subsets`, then uses the average of all :math:`H_m^T 1`s for any given subset (scaled to the relative size of the subset if subsets are not equal size). Defaults to 'subset_specific'.
      :type norm_BP_subset_method: str, optional

      :returns: The gradient of the Poisson likelihood.
      :rtype: torch.Tensor


   .. py:method:: compute_gradient_ff(object, precomputed_forward_projection = None, subset_idx = None)

      Computes the second order derivative :math:`\nabla_{ff} L(g|f) = -H^T (g/(Hf+s)^2) H`.

      :param object: Object :math:`f` used in computation.
      :type object: torch.Tensor
      :param precomputed_forward_projection: The quantity :math:`Hf`. If this value is None, then the forward projection is recomputed. Defaults to None.
      :type precomputed_forward_projection: torch.Tensor | None, optional
      :param subset_idx: Specifies the subset for all computations. Defaults to None.
      :type subset_idx: int, optional

      :returns: The operator given by the second order derivative.
      :rtype: Callable


   .. py:method:: compute_gradient_gf(object, precomputed_forward_projection=None, subset_idx=None)

      Computes the second order derivative :math:`\nabla_{gf} L(g|f) = 1/(Hf+s) H`.

      :param object: Object :math:`f` used in computation.
      :type object: torch.Tensor
      :param precomputed_forward_projection: The quantity :math:`Hf`. If this value is None, then the forward projection is recomputed. Defaults to None.
      :type precomputed_forward_projection: torch.Tensor | None, optional
      :param subset_idx: Specifies the subset for all computations. Defaults to None.
      :type subset_idx: int, optional

      :returns: The operator given by the second order derivative.
      :rtype: Callable


   .. py:method:: compute_gradient_sf(object, precomputed_forward_projection=None, subset_idx=None)

      Computes the second order derivative :math:`\nabla_{sf} L(g|f,s) = -g/(Hf+s)^2 H` where :math:`s` is an additive term representative of scatter.

      :param object: Object :math:`f` used in computation.
      :type object: torch.Tensor
      :param precomputed_forward_projection: The quantity :math:`Hf`. If this value is None, then the forward projection is recomputed. Defaults to None.
      :type precomputed_forward_projection: torch.Tensor | None, optional
      :param subset_idx: Specifies the subset for all computations. Defaults to None.
      :type subset_idx: int, optional

      :returns: The operator given by the second order derivative.
      :rtype: Callable



