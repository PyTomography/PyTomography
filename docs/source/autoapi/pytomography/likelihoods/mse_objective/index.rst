:py:mod:`pytomography.likelihoods.mse_objective`
================================================

.. py:module:: pytomography.likelihoods.mse_objective


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.likelihoods.mse_objective.NegativeMSELikelihood
   pytomography.likelihoods.mse_objective.SARTWeightedNegativeMSELikelihood




.. py:class:: NegativeMSELikelihood(system_matrix, projections = None, additive_term = None, scaling_constant = 1.0)

   Bases: :py:obj:`pytomography.likelihoods.likelihood.Likelihood`

   Negative mean squared error likelihood function :math:`L(g|f) = -\frac{1}{2} \alpha \sum_i \left(g_i-(Hf)_i\right)^2` where :math:`g` is the acquired data, :math:`H` is the system matrix, :math:`f` is the object being reconstructed, and :math:`\alpha` is the scaling constant. The negative is taken so that the it works in gradient ascent (as opposed to descent) algorithms

   :param system_matrix: The system matrix modeling the particular system whereby the projections were obtained
   :type system_matrix: SystemMatrix
   :param projections: Acquired data
   :type projections: torch.Tensor
   :param additive_term: Additional term added after forward projection by the system matrix. This term might include things like scatter and randoms. Defaults to None.
   :type additive_term: torch.Tensor, optional
   :param additive_term_variance_estimate: Variance estimate of the additive term. If none, then uncertainty estimation does not include contribution from the additive term. Defaults to None.
   :type additive_term_variance_estimate: torch.tensor, optional

   .. py:method:: compute_gradient(object, subset_idx = None, norm_BP_subset_method = 'subset_specific')

      Computes the gradient for the mean squared error objective function given by :math:`\nabla_f L(g|f) =  H^T \left(g-Hf\right)`.

      :param object: Object :math:`f` on which the likelihood is computed
      :type object: torch.Tensor
      :param subset_idx: Specifies the subset for forward/back projection. If none, then forward/back projection is done over all subsets, and the entire projections :math:`g` are used. Defaults to None.
      :type subset_idx: int | None, optional
      :param norm_BP_subset_method: Specifies how :math:`H^T 1` is calculated when subsets are used. If 'subset_specific', then uses :math:`H_m^T 1`. If `average_of_subsets`, then uses the average of all :math:`H_m^T 1`s for any given subset (scaled to the relative size of the subset if subsets are not equal size). Defaults to 'subset_specific'.
      :type norm_BP_subset_method: str, optional

      :returns: The gradient of the Poisson likelihood.
      :rtype: torch.Tensor



.. py:class:: SARTWeightedNegativeMSELikelihood(system_matrix, projections, additive_term = None)

   Bases: :py:obj:`pytomography.likelihoods.likelihood.Likelihood`

   Generic likelihood class in PyTomography. Subclasses may implement specific likelihoods with methods to compute the likelihood itself as well as particular gradients of the likelihood

   :param system_matrix: The system matrix modeling the particular system whereby the projections were obtained
   :type system_matrix: SystemMatrix
   :param projections: Acquired data. If listmode, then this argument need not be provided, and it is set to a tensor of ones. Defaults to None.
   :type projections: torch.Tensor | None
   :param additive_term: Additional term added after forward projection by the system matrix. This term might include things like scatter and randoms. Defaults to None.
   :type additive_term: torch.Tensor, optional
   :param additive_term_variance_estimate: Variance estimate of the additive term. If none, then uncertainty estimation does not include contribution from the additive term. Defaults to None.
   :type additive_term_variance_estimate: torch.tensor, optional

   .. py:method:: compute_gradient(object, subset_idx = None, norm_BP_subset_method = 'subset_specific')

      Computes the gradient for the mean squared error objective function given by :math:`\nabla_f L(g|f) =  H^T \left(g-Hf\right)`.

      :param object: Object :math:`f` on which the likelihood is computed
      :type object: torch.Tensor
      :param subset_idx: Specifies the subset for forward/back projection. If none, then forward/back projection is done over all subsets, and the entire projections :math:`g` are used. Defaults to None.
      :type subset_idx: int | None, optional
      :param norm_BP_subset_method: Specifies how :math:`H^T 1` is calculated when subsets are used. If 'subset_specific', then uses :math:`H_m^T 1`. If `average_of_subsets`, then uses the average of all :math:`H_m^T 1`s for any given subset (scaled to the relative size of the subset if subsets are not equal size). Defaults to 'subset_specific'.
      :type norm_BP_subset_method: str, optional

      :returns: The gradient of the Poisson likelihood.
      :rtype: torch.Tensor



