:py:mod:`pytomography.priors.prior`
===================================

.. py:module:: pytomography.priors.prior


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.priors.prior.Prior




.. py:class:: Prior(beta)

   Abstract class for implementation of prior :math:`V(f)` where :math:`V` is from the log-posterior probability :math:`\ln L(\tilde{f}, f) - \beta V(f)`. Any function inheriting from this class should implement a ``foward`` method that computes the tensor :math:`\frac{\partial V}{\partial f_r}` where :math:`f` is an object tensor.

   :param beta: Used to scale the weight of the prior
   :type beta: float

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


   .. py:method:: compute_gradient()
      :abstractmethod:

      Abstract method to compute the gradient of the prior based on the ``self.object`` attribute.




