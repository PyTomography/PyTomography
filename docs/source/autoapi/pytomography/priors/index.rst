:py:mod:`pytomography.priors`
=============================

.. py:module:: pytomography.priors

.. autoapi-nested-parse::

   Under the modification :math:`L(\tilde{f}, f) \to L(\tilde{f}, f)e^{-\beta V(f)}`, the log-liklihood becomes :math:`\ln L(\tilde{f},f) - \beta V(f)`. Typically, the prior has a form :math:`V(f) = \sum_{r,s} w_{r,s} \phi(f_r,f_s)`. In this expression, :math:`r` represents a voxel in the object, :math:`s` represents a voxel nearby to voxel :math:`r`, and :math:`w_{r,s}` is a weight that adjusts for the Euclidean distance between the voxels.



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




.. py:class:: Prior(beta)

   Abstract class for implementation of prior :math:`V(f)` where :math:`V` is from the log-posterior probability :math:`\ln L(\tilde{f}, f) - \beta V(f)`. Any function inheriting from this class should implement a ``foward`` method that computes the tensor :math:`\frac{\partial V}{\partial f_r}` where :math:`f` is an object tensor.

   :param beta: Used to scale the weight of the prior
   :type beta: float
   :param device: Pytorch device used for computation. Defaults to 'cpu'.
   :type device: float

   .. py:method:: set_object_meta(object_meta)

      Sets object metadata parameters.

      :param object_meta: Object metadata describing the system.
      :type object_meta: ObjectMeta


   .. py:method:: set_beta_scale(factor)

      Sets :math:`\beta`

      :param factor: Value of :math:`\beta`
      :type factor: float


   .. py:method:: set_object(object)

      Sets the object :math:`f_r` used to compute :math:`\frac{\partial V}{\partial f_r}`

      :param object: Tensor of size [batch_size, Lx, Ly, Lz] representing :math:`f_r`.
      :type object: torch.tensor


   .. py:method:: __call__()
      :abstractmethod:

      Abstract method to compute prior based on the ``self.object`` attribute.




.. py:class:: NearestNeighbourPrior(beta, phi, **kwargs)

   Bases: :py:obj:`pytomography.priors.prior.Prior`

   Implementation of priors where gradients depend on summation over nearest neighbours :math:`s` to voxel :math:`r` given by : :math:`\frac{\partial V}{\partial f_r}=\beta\sum_{r,s}w_{r,s}\phi(f_r, f_s)` where :math:`V` is from the log-posterior probability :math:`\ln L (\tilde{f}, f) - \beta V(f)`.

   :param beta: Used to scale the weight of the prior
   :type beta: float
   :param phi: Function :math:`\phi` used in formula above. Input arguments should be :math:`f_r`, :math:`f_s`, and any `kwargs` passed to this initialization function.
   :type phi: function
   :param device: Pytorch device used for computation. Defaults to 'cpu'.
   :type device: str, optional

   .. py:method:: __call__()

      Computes the prior on ``self.object``

      :returns: Tensor of shape [batch_size, Lx, Ly, Lz] representing :math:`\frac{\partial V}{\partial f_r}`
      :rtype: torch.tensor



.. py:class:: QuadraticPrior(beta, delta = 1)

   Bases: :py:obj:`NearestNeighbourPrior`

   Subclass of ``NearestNeighbourPrior`` where :math:`\phi(f_r, f_s)= (f_r-f_s)/\delta` corresponds to a quadratic prior :math:`V(f)=\frac{1}{4}\sum_{r,s} w_{r,s} \left(\frac{f_r-f_s}{\delta}\right)^2`

   :param beta: Used to scale the weight of the prior
   :type beta: float
   :param delta: Parameter :math:`\delta` in equation above. Defaults to 1.
   :type delta: float, optional


.. py:class:: LogCoshPrior(beta, delta = 1)

   Bases: :py:obj:`NearestNeighbourPrior`

   Subclass of ``NearestNeighbourPrior`` where :math:`\phi(f_r,f_s)=\tanh((f_r-f_s)/\delta)` corresponds to the logcosh prior :math:`V(f)=\sum_{r,s} w_{r,s} \log\cosh\left(\frac{f_r-f_s}{\delta}\right)`

   :param beta: Used to scale the weight of the prior
   :type beta: float
   :param delta: Parameter :math:`\delta` in equation above. Defaults to 1.
   :type delta: float, optional


.. py:class:: RelativeDifferencePrior(beta = 1, gamma = 1, epsilon = 1e-08)

   Bases: :py:obj:`NearestNeighbourPrior`

   Subclass of ``NearestNeighbourPrior`` where :math:`\phi(f_r,f_s)=\frac{2(f_r-f_s)(\gamma|f_r-f_s|+3f_s + f_r)}{(\gamma|f_r-f_s|+f_r+f_s)^2}` corresponds to the relative difference prior :math:`V(f)=\sum_{r,s} w_{r,s} \frac{(f_r-f_s)^2}{f_r+f_s+\gamma|f_r-f_s|}`

   :param beta: Used to scale the weight of the prior
   :type beta: float
   :param gamma: Parameter :math:`\gamma` in equation above. Defaults to 1.
   :type gamma: float, optional
   :param epsilon: Prevent division by 0, Defaults to 1e-8.
   :type epsilon: float, optional


