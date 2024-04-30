:py:mod:`pytomography.callbacks.callback`
=========================================

.. py:module:: pytomography.callbacks.callback


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.callbacks.callback.Callback




.. py:class:: Callback

   Abstract class used for callbacks. Subclasses must redefine the ``__init__`` and ``run`` methods. If a callback is used as an argument in an iterative reconstruction algorihtm, the ``__run__`` method is called after each subiteration.


   .. py:method:: run(object, n_iter)
      :abstractmethod:

      Abstract method for ``run``.

      :param object: Object at current iteration/subset in the reconstruction algorithm
      :type object: torch.Tensor[Lx, Ly, Lz]
      :param n_iter: The iteration number
      :type n_iter: int

      :returns: Modified object from callback. This must be returned by all callbacks (if the callback doesn't change the object, then the passed object is returned)
      :rtype: torch.Tensor


   .. py:method:: finalize(object)

      Abstract method for ``run``.

      :param object: Reconstructed object (all iterations/subsets completed)
      :type object: torch.Tensor[Lx, Ly, Lz]



