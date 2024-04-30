:py:mod:`pytomography.priors`
=============================

.. py:module:: pytomography.priors

.. autoapi-nested-parse::

   This module contains classes/functionality for encorporating priors in statistical reconstruction algorithms. Under the modification :math:`L(\tilde{f}, f) \to L(\tilde{f}, f)e^{-\beta V(f)}`, the log-liklihood becomes :math:`\ln L(\tilde{f},f) - \beta V(f)`. Typically, the prior has a form :math:`V(f) = \sum_{r,s} w_{r,s} \phi(f_r,f_s)`. In this expression, :math:`r` represents a voxel in the object, :math:`s` represents a voxel nearby to voxel :math:`r`, and :math:`w_{r,s}` represents a weighting between the voxels.



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   nearest_neighbour/index.rst
   prior/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.priors.Prior
   pytomography.priors.NearestNeighbourPrior
   pytomography.priors.QuadraticPrior
   pytomography.priors.LogCoshPrior
   pytomography.priors.RelativeDifferencePrior
   pytomography.priors.NeighbourWeight
   pytomography.priors.EuclideanNeighbourWeight
   pytomography.priors.AnatomyNeighbourWeight
   pytomography.priors.TopNAnatomyNeighbourWeight




.. py:class:: Prior(beta, obj2obj_transforms = [])

   Abstract class for implementation of prior :math:`V(f)` where :math:`V` is from the log-posterior probability :math:`\ln L(\tilde{f}, f) - \beta V(f)`. Any function inheriting from this class should implement a ``foward`` method that computes the tensor :math:`\frac{\partial V}{\partial f_r}` where :math:`f` is an object tensor.

   :param beta: Used to scale the weight of the prior
   :type beta: float
   :param obj2obj_transforms: Sequence of transforms applied after computation of prior or gradients.
   :type obj2obj_transforms: Sequence

   .. py:method:: set_object_meta(object_meta)

      Sets object metadata parameters.

      :param object_meta: Object metadata describing the system.
      :type object_meta: ObjectMeta


   .. py:method:: set_beta_scale(factor)

      Sets a scale factor for :math:`\beta` required for OSEM when finite subsets are used per iteration.

      :param factor: Value by which to scale :math:`\beta`
      :type factor: float


   .. py:method:: set_object(object)

      Sets the object :math:`f_r` used to compute :math:`\frac{\partial V}{\partial f_r}`

      :param object: Tensor of size [batch_size, Lx, Ly, Lz] representing :math:`f_r`.
      :type object: torch.tensor


   .. py:method:: __call__()
      :abstractmethod:

      Abstract method to compute the gradient of the prior based on the ``self.object`` attribute.




.. py:class:: NearestNeighbourPrior(beta, weight = None, **kwargs)

   Bases: :py:obj:`pytomography.priors.prior.Prior`

   Generic class for the nearest neighbour prior.

   :param beta: Used to scale the weight of the prior
   :type beta: float
   :param weight: this specifies :math:`w_{r,s}` above. If ``None``, then uses EuclideanNeighbourWeight, which weights neighbouring voxels based on their euclidean distance. Defaults to None.
   :type weight: NeighbourWeight, optional

   .. py:method:: set_object_meta(object_meta)

      Sets object metadata parameters.

      :param object_meta: Object metadata describing the system.
      :type object_meta: ObjectMeta


   .. py:method:: _pair_contribution(phi, beta_scale=False, second_order_derivative_object = None, swap_object_and_neighbour = False)

      Helper function used to compute prior and associated gradients

      :returns: Tensor of shape [batch_size, Lx, Ly, Lz].
      :rtype: torch.tensor


   .. py:method:: phi0(fr, fs)
      :abstractmethod:


   .. py:method:: phi1(fr, fs)
      :abstractmethod:


   .. py:method:: phi2_1(fr, fs)
      :abstractmethod:


   .. py:method:: phi2_2(fr, fs)
      :abstractmethod:


   .. py:method:: __call__(derivative_order = 0)

      Used to compute the prior with gradient of specified order. If order 0, then returns a float (the value of the prior). If order 1, then returns a torch.Tensor representative of the prior gradient at each voxel. If order 2, then returns a callable function (representative of a higher order tensor but without storing each component).

      :param derivative_order: The order of the derivative to compute. This will specify the ouput; only possible values are 0, 1, or 2. Defaults to 0.
      :type derivative_order: int, optional

      :raises NotImplementedError: for cases where the derivative order is not between 0 and 2.

      :returns: The prior with derivative of specified order.
      :rtype: float | torch.Tensor | Callable



.. py:class:: QuadraticPrior(beta, weight = None, delta = 1)

   Bases: :py:obj:`NearestNeighbourPrior`

   Subclass of ``NearestNeighbourPrior`` corresponding to a quadratic prior: namely :math:`\phi_0(f_r, f_s) = 1/4 \left[(fr-fs)/\delta\right]^2` and where the gradient is determined by :math:`\phi_1(f_r, f_s) = (f_r-f_s)/\delta`

   :param beta: Used to scale the weight of the prior
   :type beta: float
   :param weight:
   :type weight: NeighbourWeight, optional
   :param delta: Parameter :math:`\delta` in equation above. Defaults to 1.
   :type delta: float, optional

   .. py:method:: phi0(fr, fs)


   .. py:method:: phi1(fr, fs)



.. py:class:: LogCoshPrior(beta, delta = 1, weight = None)

   Bases: :py:obj:`NearestNeighbourPrior`

   Subclass of ``NearestNeighbourPrior`` corresponding to a logcosh prior: namely :math:`\phi_0(f_r, f_s) = \tanh((f_r-f_s)/\delta)` and where the gradient is determined by :math:`\phi_1(f_r, f_s) = \log \cosh \left[(f_r-f_s)/\delta\right]`

   :param beta: Used to scale the weight of the prior
   :type beta: float
   :param delta: Parameter :math:`\delta` in equation above. Defaults to 1.
   :type delta: float, optional
   :param weight:
   :type weight: NeighbourWeight, optional

   .. py:method:: phi0(fr, fs)


   .. py:method:: phi1(fr, fs)



.. py:class:: RelativeDifferencePrior(beta, weight = None, gamma = 1)

   Bases: :py:obj:`NearestNeighbourPrior`

   Subclass of ``NearestNeighbourPrior`` corresponding to the relative difference prior: namely :math:`\phi_0(f_r, f_s) = \frac{(f_r-f_s)^2}{f_r+f_s+\gamma|f_r-f_s|}` and where the gradient is determined by :math:`\phi_1(f_r, f_s) = \frac{2(f_r-f_s)(\gamma|f_r-f_s|+3f_s + f_r)}{(\gamma|f_r-f_s|+f_r+f_s)^2}`

   :param beta: Used to scale the weight of the prior
   :type beta: float
   :param gamma: Parameter :math:`\gamma` in equation above. Defaults to 1.
   :type gamma: float, optional
   :param weight:
   :type weight: NeighbourWeight, optional

   .. py:method:: phi0(fr, fs)


   .. py:method:: phi1(fr, fs)


   .. py:method:: phi2_1(fr, fs)


   .. py:method:: phi2_2(fr, fs)



.. py:class:: NeighbourWeight

   Abstract class for assigning weight :math:`w_{r,s}` in nearest neighbour priors.


   .. py:method:: set_object_meta(object_meta)

      Sets object meta to get appropriate spacing information

      :param object_meta: Object metadata.
      :type object_meta: ObjectMeta


   .. py:method:: __call__(coords)
      :abstractmethod:

      Computes the weight :math:`w_{r,s}` given the relative position :math:`s` of the nearest neighbour

      :param coords: Tuple of coordinates ``(i,j,k)`` that represent the shift of neighbour :math:`s` relative to :math:`r`.
      :type coords: Sequence[int,int,int]



.. py:class:: EuclideanNeighbourWeight

   Bases: :py:obj:`NeighbourWeight`

   Implementation of ``NeighbourWeight`` where inverse Euclidean distance is the weighting between nearest neighbours.


   .. py:method:: __call__(coords)

      Computes the weight :math:`w_{r,s}` using inverse Euclidean distance between :math:`r` and :math:`s`.

      :param coords: Tuple of coordinates ``(i,j,k)`` that represent the shift of neighbour :math:`s` relative to :math:`r`.
      :type coords: Sequence[int,int,int]



.. py:class:: AnatomyNeighbourWeight(anatomy_image, similarity_function)

   Bases: :py:obj:`NeighbourWeight`

   Implementation of ``NeighbourWeight`` where inverse Euclidean distance and anatomical similarity is used to compute neighbour weight.

   :param anatomy_image: Object corresponding to an anatomical image (such as CT/MRI)
   :type anatomy_image: torch.Tensor[batch_size,Lx,Ly,Lz]
   :param similarity_function: User-defined function that computes the similarity between :math:`r` and :math:`s` in the anatomical image. The function should be bounded between 0 and 1 where 1 represets complete similarity and 0 represents complete dissimilarity.
   :type similarity_function: Callable

   .. py:method:: set_object_meta(object_meta)

      Sets object meta to get appropriate spacing information

      :param object_meta: Object metadata.
      :type object_meta: ObjectMeta


   .. py:method:: __call__(coords)

      Computes the weight :math:`w_{r,s}` using inverse Euclidean distance and anatomical similarity between :math:`r` and :math:`s`.

      :param coords: Tuple of coordinates ``(i,j,k)`` that represent the shift of neighbour :math:`s` relative to :math:`r`.
      :type coords: Sequence[int,int,int]



.. py:class:: TopNAnatomyNeighbourWeight(anatomy_image, N_neighbours)

   Bases: :py:obj:`NeighbourWeight`

   Implementation of ``NeighbourWeight`` where inverse Euclidean distance and anatomical similarity is used. In this case, only the top N most similar neighbours are used as weight

   :param anatomy_image: Object corresponding to an anatomical image (such as CT/MRI)
   :type anatomy_image: torch.Tensor[batch_size,Lx,Ly,Lz]
   :param N_neighbours: Number of most similar neighbours to use
   :type N_neighbours: int

   .. py:method:: set_object_meta(object_meta)

      Sets object meta to get appropriate spacing information

      :param object_meta: Object metadata.
      :type object_meta: ObjectMeta


   .. py:method:: compute_inclusion_tensor()


   .. py:method:: __call__(coords)

      Computes the weight :math:`w_{r,s}` using inverse Euclidean distance and anatomical similarity between :math:`r` and :math:`s`.

      :param coords: Tuple of coordinates ``(i,j,k)`` that represent the shift of neighbour :math:`s` relative to :math:`r`.
      :type coords: Sequence[int,int,int]



