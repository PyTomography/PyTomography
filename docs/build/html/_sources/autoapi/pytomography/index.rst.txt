:py:mod:`pytomography`
======================

.. py:module:: pytomography


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   algorithms/index.rst
   corrections/index.rst
   io/index.rst
   metadata/index.rst
   priors/index.rst
   projections/index.rst
   utils/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.OSEMNet




.. py:class:: OSEMNet(object_initial, forward_projection_net, back_projection_net, prior=None)

   Bases: :py:obj:`torch.nn.Module`

   Network used to run OSEM reconstruction: :math:`f_i^{(n+1)} = \frac{f_i^{(n)}}{\sum_j c_{ij} + \beta \frac{\partial V}{\partial f_r}|_{f_i=f_i^{(n)}}} \sum_j c_{ij}\frac{g_j}{\sum_i c_{ij}f_i^{(n)}}`. Initializer initializes the reconstruction algorithm with the initial object guess :math:`f_i^{(0)}`, forward and back projections used (i.e. networks to compute :math:`\sum_i c_{ij} a_i` and :math:`\sum_j c_{ij} b_j`), and prior for Bayesian corrections. Note that OSEMNet uses the one step late (OSL algorithm to compute priors during reconstruction. Once the class is initialized, the number of iterations and subsets are specified at recon time when the forward method is called.

   :param object_initial: represents the initial object guess :math:`f_i^{(0)}` for the algorithm in object space
   :type object_initial: torch.tensor[batch_size, Lx, Ly, Lz]
   :param forward_projection_net: the forward projection network used to compute :math:`\sum_{i} c_{ij} a_i` where :math:`a_i` is the object being forward projected.
   :type forward_projection_net: ForwardProjectionNet
   :param back_projection_net: the back projection network used to compute :math:`\sum_{j} c_{ij} b_j` where :math:`b_j` is the image being back projected.
   :type back_projection_net: BackProjectionNet
   :param prior: the Bayesian prior; computes :math:`\beta \frac{\partial V}{\partial f_r}|_{f_i=f_i^{\text{old}}}`. If ``None``, then this term is 0. Defaults to None
   :type prior: Prior, optional

   .. py:method:: get_subset_splits(n_subsets, n_angles)

      Returns a list of arrays; each array contains indices, corresponding
          to projection numbers, that are used in ordered-subsets. For example,
          ``get_subsets_splits(2, 6)`` would return ``[[0,2,4],[1,3,5]]``.
      :param n_subsets: number of subsets used in OSEM
      :type n_subsets: int
      :param n_angles: total number of projections
      :type n_angles: int

      :returns: list of index arrays for each subset
      :rtype: list


   .. py:method:: set_image(image)

      Sets the projection data which is to be reconstructed

      :param image: image data
      :type image: torch.tensor[batch_size, Ltheta, Lr, Lz]


   .. py:method:: set_prior(prior)

      Sets the prior used for Bayesian modeling

      :param prior: The prior class corresponding to a particular model
      :type prior: Prior


   .. py:method:: forward(n_iters, n_subsets, comparisons=None, delta=1e-11)

      Performs the reconstruction using `n_iters` iterations and `n_subsets` subsets.

      :param n_iters: _description_
      :type n_iters: int
      :param n_subsets: _description_
      :type n_subsets: int
      :param comparisons: FIX. Defaults to None.
      :type comparisons: FIX, optional
      :param delta: Used to prevent division by zero when calculating ratio, defaults to 1e-11.
      :type delta: _type_, optional

      :returns: reconstructed object
      :rtype: torch.tensor[batch_size, Lx, Ly, Lz]



