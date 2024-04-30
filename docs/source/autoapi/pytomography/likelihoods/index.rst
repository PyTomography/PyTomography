:py:mod:`pytomography.likelihoods`
==================================

.. py:module:: pytomography.likelihoods


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   likelihood/index.rst
   mse_objective/index.rst
   poisson_log_likelihood/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.likelihoods.NegativeMSELikelihood
   pytomography.likelihoods.SARTWeightedNegativeMSELikelihood
   pytomography.likelihoods.PoissonLogLikelihood
   pytomography.likelihoods.Likelihood




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



.. py:class:: Likelihood(system_matrix, projections = None, additive_term = None, additive_term_variance_estimate = None)

   Generic likelihood class in PyTomography. Subclasses may implement specific likelihoods with methods to compute the likelihood itself as well as particular gradients of the likelihood

   :param system_matrix: The system matrix modeling the particular system whereby the projections were obtained
   :type system_matrix: SystemMatrix
   :param projections: Acquired data. If listmode, then this argument need not be provided, and it is set to a tensor of ones. Defaults to None.
   :type projections: torch.Tensor | None
   :param additive_term: Additional term added after forward projection by the system matrix. This term might include things like scatter and randoms. Defaults to None.
   :type additive_term: torch.Tensor, optional
   :param additive_term_variance_estimate: Variance estimate of the additive term. If none, then uncertainty estimation does not include contribution from the additive term. Defaults to None.
   :type additive_term_variance_estimate: torch.tensor, optional

   .. py:method:: _set_n_subsets(n_subsets)

      Sets the number of subsets to be used when computing the likelihood

      :param n_subsets: Number of subsets
      :type n_subsets: int


   .. py:method:: _get_projection_subset(projections, subset_idx = None)

      Method for getting projection subset corresponding to given subset index

      :param projections: Projection data
      :type projections: torch.Tensor
      :param subset_idx: Subset index
      :type subset_idx: int

      :returns: Subset projection data
      :rtype: torch.Tensor


   .. py:method:: _get_normBP(subset_idx, return_sum = False)

      Gets normalization factor (back projection of ones)

      :param subset_idx: Subset index
      :type subset_idx: int
      :param return_sum: Sum normalization factor from all subsets. Defaults to False.
      :type return_sum: bool, optional

      :returns: Normalization factor
      :rtype: torch.Tensor


   .. py:method:: compute_gradient(*args, **kwargs)
      :abstractmethod:

      Function used to compute the gradient of the likelihood :math:`\nabla_{f} L(g|f)`

      :raises NotImplementedError: Must be implemented by sub classes


   .. py:method:: compute_gradient_ff(*args, **kwargs)
      :abstractmethod:

      Function used to compute the second order gradient (with respect to the object twice) of the likelihood :math:`\nabla_{ff} L(g|f)`

      :raises NotImplementedError: Must be implemented by sub classes


   .. py:method:: compute_gradient_gf(*args, **kwargs)
      :abstractmethod:

      Function used to compute the second order gradient (with respect to the object then image) of the likelihood :math:`\nabla_{gf} L(g|f)`

      :raises NotImplementedError: Must be implemented by sub classes


   .. py:method:: compute_gradient_sf(*args, **kwargs)
      :abstractmethod:

      Function used to compute the second order gradient (with respect to the object then additive term) of the likelihood :math:`\nabla_{sf} L(g|f,s)`

      :raises NotImplementedError: Must be implemented by sub classes



