:py:mod:`pytomography.callbacks.callback`
=========================================

.. py:module:: pytomography.callbacks.callback


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.callbacks.callback.CallBack




.. py:class:: CallBack

   Abstract class used for callbacks. Subclasses must redefine the ``__init__`` and ``run`` methods. If a callback is used as an argument in an iterative reconstruction algorihtm, the ``__run__`` method is called after each subiteration.


   .. py:method:: run(obj, n_iter, n_subset)
      :abstractmethod:

      Abstract method for ``run``.

      :param obj: An object which one can compute various statistics from.
      :type obj: torch.tensor[batch_size, Lx, Ly, Lz]
      :param n_iter: The iteration number
      :type n_iter: int
      :param n_subset: The subiteration number
      :type n_subset: int



