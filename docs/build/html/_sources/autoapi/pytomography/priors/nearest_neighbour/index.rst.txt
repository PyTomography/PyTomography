:py:mod:`pytomography.priors.nearest_neighbour`
===============================================

.. py:module:: pytomography.priors.nearest_neighbour

.. autoapi-nested-parse::

   For all priors implemented here, the neighbouring voxels considered are those directly surrounding a given voxel, so :math:`\sum_s` is a sum over 26 points.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.priors.nearest_neighbour.NearestNeighbourPrior
   pytomography.priors.nearest_neighbour.QuadraticPrior
   pytomography.priors.nearest_neighbour.LogCoshPrior
   pytomography.priors.nearest_neighbour.RelativeDifferencePrior




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


