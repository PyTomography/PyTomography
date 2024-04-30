:py:mod:`pytomography.algorithms`
=================================

.. py:module:: pytomography.algorithms

.. autoapi-nested-parse::

   This module contains all the available reconstruction algorithms in PyTomography.



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   dip_recon/index.rst
   fbp/index.rst
   preconditioned_gradient_ascent/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.algorithms.PreconditionedGradientAscentAlgorithm
   pytomography.algorithms.OSEM
   pytomography.algorithms.OSMAPOSL
   pytomography.algorithms.BSREM
   pytomography.algorithms.KEM
   pytomography.algorithms.RBIEM
   pytomography.algorithms.RBIMAP
   pytomography.algorithms.SART
   pytomography.algorithms.PGAAMultiBedSPECT
   pytomography.algorithms.MLEM
   pytomography.algorithms.FilteredBackProjection
   pytomography.algorithms.DIPRecon




.. py:class:: PreconditionedGradientAscentAlgorithm(likelihood, prior = None, object_initial = None, addition_after_iteration = 0, **kwargs)

   Generic class for preconditioned gradient ascent algorithms: i.e. those that have the form :math:`f^{n+1} = f^{n} + C^{n}(f^{n}) \left[\nabla_{f} L(g^n|f^{n}) - \beta \nabla_{f} V(f^{n}) \right]`.

   :param likelihood: Likelihood class that facilitates computation of :math:`L(g^n|f^{n})` and its associated derivatives.
   :type likelihood: Likelihood
   :param prior: Prior class that faciliates the computation of function :math:`V(f)` and its associated derivatives. If None, then no prior is used Defaults to None.
   :type prior: Prior, optional
   :param object_initial: Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
   :type object_initial: torch.Tensor | None, optional
   :param addition_after_iteration: Value to add to the object after each iteration. This prevents image voxels getting "locked" at values of 0 for certain algorithms. Defaults to 0.
   :type addition_after_iteration: float, optional

   .. py:method:: _set_n_subsets(n_subsets)

      Sets the number of subsets used in the reconstruction algorithm.

      :param n_subsets: Number of subsets
      :type n_subsets: int


   .. py:method:: _compute_preconditioner(object, n_iter, n_subset)
      :abstractmethod:

      Computes the preconditioner factor :math:`C^{n}(f^{n})`. Must be implemented by any reconstruction algorithm that inherits from this generic class.

      :param object: Object :math:`f^n`
      :type object: torch.Tensor
      :param n_iter: Iteration number
      :type n_iter: int
      :param n_subset: Subset number
      :type n_subset: int

      :raises NotImplementedError: .


   .. py:method:: _compute_callback(n_iter, n_subset)

      Method for computing callbacks after each reconstruction iteration

      :param n_iter: Number of iterations
      :type n_iter: int
      :param n_subset: Number of subsets
      :type n_subset: int


   .. py:method:: __call__(n_iters, n_subsets = 1, n_subset_specific = None, callback = None)

      _summary_

      :param Args:
      :param n_iters: Number of iterations
      :type n_iters: int
      :param n_subsets: Number of subsets
      :type n_subsets: int
      :param n_subset_specific: Ignore all updates except for this subset.
      :type n_subset_specific: int
      :param callback: Callback function to be called after each subiteration. Defaults to None.
      :type callback: Callback, optional

      :returns: Reconstructed object.
      :rtype: torch.Tensor



.. py:class:: OSEM(likelihood, object_initial = None)

   Bases: :py:obj:`LinearPreconditionedGradientAscentAlgorithm`

   Implementation of the ordered subset expectation maximum algorithm :math:`f^{n+1} = f^{n} + \frac{f^n}{H_n^T} \nabla_{f} L(g^n|f^{n})`.

   :param likelihood: Likelihood function :math:`L`.
   :type likelihood: Likelihood
   :param object_initial: Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
   :type object_initial: torch.Tensor | None, optional

   .. py:method:: _linear_preconditioner_factor(n_iter, n_subset)

      Computes the linear preconditioner factor :math:`D^n = 1/H_n^T 1`

      :param n_iter: iteration number
      :type n_iter: int
      :param n_subset: subset number
      :type n_subset: int

      :returns: linear preconditioner factor
      :rtype: torch.Tensor



.. py:class:: OSMAPOSL(likelihood, object_initial = None, prior = None)

   Bases: :py:obj:`PreconditionedGradientAscentAlgorithm`

   Implementation of the ordered subset maximum a posteriori one step late algorithm :math:`f^{n+1} = f^{n} + \frac{f^n}{H_n^T+\nabla_f V(f^n)} \left[ \nabla_{f} L(g^n|f^{n}) - \nabla_f V(f^n) \right]`

   :param likelihood: Likelihood function :math:`L`.
   :type likelihood: Likelihood
   :param object_initial: Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
   :type object_initial: torch.Tensor | None, optional
   :param prior: Prior class that faciliates the computation of function :math:`V(f)` and its associated derivatives. If None, then no prior is used. Defaults to None.
   :type prior: Prior, optional

   .. py:method:: _compute_preconditioner(object, n_iter, n_subset)

      Computes the preconditioner factor :math:`C^n(f^n) = \frac{f^n}{H_n^T+\nabla_f V(f^n)}`

      :param object: Object estimate :math:`f^n`
      :type object: torch.Tensor
      :param n_iter: iteration number
      :type n_iter: int
      :param n_subset: subset number
      :type n_subset: int

      :returns: preconditioner factor.
      :rtype: torch.Tensor



.. py:class:: BSREM(likelihood, object_initial = None, prior = None, relaxation_sequence = lambda _: 1, addition_after_iteration=0.0001)

   Bases: :py:obj:`LinearPreconditionedGradientAscentAlgorithm`

   Implementation of the block sequential regularized expectation maximum algorithm :math:`f^{n+1} = f^{n} + \frac{\alpha(n)}{\omega_n H^T 1} \left[\nabla_{f} L(g^n|f^{n}) - \nabla_f V(f^n) \right]`

   :param likelihood: likelihood function :math:`L`
   :type likelihood: Likelihood
   :param object_initial: Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
   :type object_initial: torch.Tensor | None, optional
   :param prior: Prior class that faciliates the computation of function :math:`V(f)` and its associated derivatives. If None, then no prior is used. Defaults to None.
   :type prior: Prior, optional
   :param relaxation_sequence: Relxation sequence :math:`\alpha(n)` used to scale future updates. Defaults to 1 for all :math:`n`. Note that when this function is provided, it takes the iteration number (not the subiteration) so that e.g. if 4 iterations and 8 subsets are used, it would call :math:`\alpha(4)` for all 8 subiterations of the final iteration.
   :type relaxation_sequence: Callable, optional
   :param addition_after_iteration: Value to add to the object after each iteration. This prevents image voxels getting "locked" at values of 0. Defaults to 1e-4.
   :type addition_after_iteration: float, optional

   .. py:method:: _linear_preconditioner_factor(n_iter, n_subset)

      Computes the linear preconditioner factor :math:`D^n = 1/(\omega_n H^T 1)` where :math:`\omega_n` corresponds to the fraction of subsets at subiteration :math:`n`.

      :param n_iter: iteration number
      :type n_iter: int
      :param n_subset: subset number
      :type n_subset: int

      :returns: linear preconditioner factor
      :rtype: torch.Tensor



.. py:class:: KEM(likelihood, object_initial = None)

   Bases: :py:obj:`OSEM`

   Implementation of the ordered subset expectation maximum algorithm :math:`\alpha^{n+1} = \alpha^{n} + \frac{\alpha^n}{\tilde{H}_n^T} \nabla_{f} L(g^n|\alpha^{n})` and where the final predicted object is :math:`f^n = K \hat{\alpha}^{n}`. The system matrix :math:`\tilde{H}` includes the kernel transform :math:`K`.

   :param likelihood: Likelihood function :math:`L`.
   :type likelihood: Likelihood
   :param object_initial: Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
   :type object_initial: torch.Tensor | None, optional

   .. py:method:: _compute_callback(n_iter, n_subset)

      Method for computing callbacks after each reconstruction iteration. This is reimplemented for KEM because the callback needs to be called on :math:`f^n = K \hat{\alpha}^{n}` as opposed to :math:`\hat{\alpha}^{n}`

      :param n_iter: Number of iterations
      :type n_iter: int
      :param n_subset: Number of subsets
      :type n_subset: int


   .. py:method:: __call__(*args, **kwargs)

      Reimplementation of the call method such that :math:`f^n = K \hat{\alpha}^{n}` is returned as opposed to :math:`\hat{\alpha}^{n}`

      :returns: reconstructed object
      :rtype: torch.Tensor



.. py:class:: RBIEM(likelihood, object_initial = None, prior = None)

   Bases: :py:obj:`LinearPreconditionedGradientAscentAlgorithm`

   Implementation of the rescaled block iterative expectation maximum algorithm

   :param likelihood: Likelihood function :math:`L`.
   :type likelihood: Likelihood
   :param object_initial: Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
   :type object_initial: torch.Tensor | None, optional
   :param prior: Prior class that faciliates the computation of function :math:`V(f)` and its associated derivatives. If None, then no prior is used. Defaults to None.
   :type prior: Prior, optional

   .. py:method:: _compute_preconditioner(object, n_iter, n_subset)

      Computes the preconditioner factor :math:`C^n(f^n) = \frac{f^n}{H_n^T+\nabla_f V(f^n)}`

      :param object: Object estimate :math:`f^n`
      :type object: torch.Tensor
      :param n_iter: iteration number
      :type n_iter: int
      :param n_subset: subset number
      :type n_subset: int

      :returns: preconditioner factor.
      :rtype: torch.Tensor



.. py:class:: RBIMAP(likelihood, object_initial = None, prior = None)

   Bases: :py:obj:`PreconditionedGradientAscentAlgorithm`

   Implementation of the rescaled block iterative maximum a posteriori algorithm

   :param likelihood: Likelihood function :math:`L`.
   :type likelihood: Likelihood
   :param object_initial: Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
   :type object_initial: torch.Tensor | None, optional
   :param prior: Prior class that faciliates the computation of function :math:`V(f)` and its associated derivatives. If None, then no prior is used. Defaults to None.
   :type prior: Prior, optional

   .. py:method:: _compute_preconditioner(object, n_iter, n_subset)

      Computes the preconditioner factor :math:`C^n(f^n) = \frac{f^n}{H_n^T+\nabla_f V(f^n)}`

      :param object: Object estimate :math:`f^n`
      :type object: torch.Tensor
      :param n_iter: iteration number
      :type n_iter: int
      :param n_subset: subset number
      :type n_subset: int

      :returns: preconditioner factor.
      :rtype: torch.Tensor



.. py:class:: SART(system_matrix, projections, additive_term = None, object_initial = None)

   Bases: :py:obj:`OSEM`

   Implementation of the SART algorithm (OSEM with SARTWeightedNegativeMSELikelihood). This algorithm takes as input the system matrix and projections (as opposed to a likelihood) since SART is OSEM with a negative MSE likelihood.

   :param system_matrix: System matrix for the imaging system.
   :type system_matrix: SystemMatrix
   :param projections: Projections for the imaging system.
   :type projections: torch.Tensor
   :param additive_term: Additive term for the imaging system. If None, then no additive term is used. Defaults to None.
   :type additive_term: torch.Tensor | None, optional
   :param object_initial: Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
   :type object_initial: torch.Tensor | None, optional


.. py:class:: PGAAMultiBedSPECT(files_NM, reconstruction_algorithms)

   Bases: :py:obj:`PreconditionedGradientAscentAlgorithm`

   Assistant class for performing reconstruction on multi-bed SPECT data. This class is a wrapper around a reconstruction algorithm that is called for each bed and then the results are stitched together.

   :param files_NM: Sequence of SPECT raw data paths corresponding to each likelihood
   :type files_NM: Sequence[str]
   :param reconstruction_algorithm: Reconstruction algorithm used for reconstruction of each bed position
   :type reconstruction_algorithm: Algorithm

   .. py:method:: __call__(n_iters, n_subsets, callback = None)

      Perform reconstruction of each bed position for specified iteraitons and subsets, and return the stitched image

      :param n_iters: Number of iterations to perform reconstruction for.
      :type n_iters: int
      :param n_subsets: Number of subsets to perform reconstruction for.
      :type n_subsets: int
      :param callback: Callback function. If a single Callback is given, then the callback is computed for the stitched image. If a sequence of callbacks is given, then it must be the same length as the number of bed positions; each callback is called on the reconstruction for each bed position. If None, no Callback is used. Defaults to None.
      :type callback: Callback | Sequence[Callback] | None, optional

      :returns: _description_
      :rtype: torch.Tensor


   .. py:method:: _compute_callback(n_iter, n_subset)

      Computes the callback at iteration ``n_iter`` and subset ``n_subset``.

      :param n_iter: Iteration number
      :type n_iter: int
      :param n_subset: Subset index
      :type n_subset: int


   .. py:method:: _finalize_callback()

      Finalizes callbacks after reconstruction. This method is called after the reconstruction algorithm has finished.



   .. py:method:: compute_uncertainty(mask, data_storage_callbacks, subiteration_number = None, return_pct = False, include_additive_term = False)

      Estimates the uncertainty in a mask (should be same shape as the stitched image). Calling this method requires a sequence of ``DataStorageCallback`` instances that have been used in a reconstruction algorithm: these data storage contain required information for each bed position.

      :param mask: Masked region of the reconstructed object: a boolean Tensor. This mask should be the same shape as the stitched object.
      :type mask: torch.Tensor
      :param data_storage_callbacks: Sequence of data storage callbacks used in reconstruction corresponding to each bed position.
      :type data_storage_callbacks: Sequence[Callback]
      :param subiteration_number: Subiteration number to compute the uncertainty for. If None, then computes the uncertainty for the last iteration. Defaults to None.
      :type subiteration_number: int | None, optional
      :param return_pct: If true, then additionally returns the percent uncertainty for the sum of counts. Defaults to False.
      :type return_pct: bool, optional
      :param include_additive_term: Whether or not to include uncertainty contribution from the additive term. This requires the ``additive_term_variance_estimate`` as an argument to the initialized likelihood. Defaults to False.
      :type include_additive_term: bool

      :returns: Absolute uncertainty in the sum of counts in the masked region (if `return_pct` is False) OR absolute uncertainty and relative uncertainty in percent (if `return_pct` is True)
      :rtype: float | Sequence[float]



.. py:class:: MLEM(likelihood, object_initial = None)

   Bases: :py:obj:`OSEM`

   Implementation of the maximum likelihood expectation maximum algorithm :math:`f^{n+1} = f^{n} + \frac{f^n}{H^T} \nabla_{f} L(g|f^{n})`.

   :param likelihood: Likelihood function :math:`L`.
   :type likelihood: Likelihood
   :param object_initial: Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
   :type object_initial: torch.Tensor | None, optional

   .. py:method:: __call__(n_iters, callback=None)

      _summary_

      :param Args:
      :param n_iters: Number of iterations
      :type n_iters: int
      :param n_subsets: Number of subsets
      :type n_subsets: int
      :param n_subset_specific: Ignore all updates except for this subset.
      :type n_subset_specific: int
      :param callback: Callback function to be called after each subiteration. Defaults to None.
      :type callback: Callback, optional

      :returns: Reconstructed object.
      :rtype: torch.Tensor



.. py:class:: FilteredBackProjection(projections, system_matrix, filter=None)

   Implementation of filtered back projection reconstruction :math:`\hat{f} = \frac{\pi}{N_{\text{proj}}} \mathcal{R}^{-1}\mathcal{F}^{-1}\Pi\mathcal{F} g` where :math:`N_{\text{proj}}` is the number of projections, :math:`\mathcal{R}` is the 3D radon transform, :math:`\mathcal{F}` is the 2D Fourier transform (applied to each projection seperately), and :math:`\Pi` is the filter applied in Fourier space, which is by default the ramp filter.

   :param projections: projection data :math:`g` to be reconstructed
   :type projections: torch.Tensor
   :param system_matrix: system matrix for the imaging system. In FBP, phenomena such as attenuation and PSF should not be implemented in the system matrix
   :type system_matrix: SystemMatrix
   :param filter: Additional Fourier space filter (applied after Ramp Filter) used during reconstruction.
   :type filter: Callable, optional

   .. py:method:: __call__()

      Applies reconstruction

      :returns: Reconstructed object prediction
      :rtype: torch.tensor



.. py:class:: DIPRecon(likelihood, prior_network, rho = 0.003)

   Implementation of the Deep Image Prior reconstruction technique (see https://ieeexplore.ieee.org/document/8581448). This reconstruction technique requires an instance of a user-defined ``prior_network`` that implements two functions: (i) a ``fit`` method that takes in an ``object`` (:math:`x`) which the network ``f(z;\theta)`` is subsequently fit to, and (ii) a ``predict`` function that returns the current network prediction :math:`f(z;\theta)`. For more details, see the Deep Image Prior tutorial.

   :param likelihood: Initialized likelihood function for the imaging system considered
   :type likelihood: Likelihood
   :param prior_network: User defined prior network that implements the neural network :math:`f(z;\theta)` that predicts an object given a prior image :math:`z`. This network also implements a ``fit`` method that takes in an object and fits the network to the object (for a specified number of iterations: SubIt2 in the paper).
   :type prior_network: nn.Module
   :param rho: Value of :math:`\rho` used in the optimization procedure. Larger values of :math:`rho` give larger weight to the neural network, while smaller values of :math:`rho` give larger weight to the EM updates. Defaults to 1.
   :type rho: float, optional

   .. py:method:: _compute_callback(n_iter, n_subset)

      Method for computing callbacks after each reconstruction iteration

      :param n_iter: Number of iterations
      :type n_iter: int
      :param n_subset: Number of subsets
      :type n_subset: int


   .. py:method:: __call__(n_iters, subit1, n_subsets_osem=1, callback=None)

      Implementation of Algorithm 1 in https://ieeexplore.ieee.org/document/8581448. This implementation gives the additional option to use ordered subsets. The quantity SubIt2 specified in the paper is controlled by the user-defined ``prior_network`` class.

      :param n_iters: Number of iterations (MaxIt in paper)
      :type n_iters: int
      :param subit1: Number of OSEM iterations before retraining neural network (SubIt1 in paper)
      :type subit1: int
      :param n_subsets_osem: Number of subsets to use in OSEM reconstruction. Defaults to 1.
      :type n_subsets_osem: int, optional

      :returns: Reconstructed image
      :rtype: torch.Tensor



