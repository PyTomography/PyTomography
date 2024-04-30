:py:mod:`pytomography.algorithms.dip_recon`
===========================================

.. py:module:: pytomography.algorithms.dip_recon


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.algorithms.dip_recon.DIPRecon




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



