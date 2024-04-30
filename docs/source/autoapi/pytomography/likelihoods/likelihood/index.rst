:py:mod:`pytomography.likelihoods.likelihood`
=============================================

.. py:module:: pytomography.likelihoods.likelihood


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.likelihoods.likelihood.Likelihood




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



