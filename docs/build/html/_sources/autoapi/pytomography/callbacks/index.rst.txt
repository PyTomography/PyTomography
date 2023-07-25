:py:mod:`pytomography.callbacks`
================================

.. py:module:: pytomography.callbacks

.. autoapi-nested-parse::

   Callbacks can be used to compute various metrics on a reconstructed object throughout each iteration in an iterative reconstruction algorithm. For example, you may want to look at the noise in the liver as a function of iteration number in OSEM. A callback is simply a class which can take in an object and perform an operation. Callbacks are optional input to reconstruction algorithms; the ``run`` method of a callback is called after each subiteration of an iterative reconstruction algorithm. All user-defined callbacks should inherit from the base class ``CallBack``. A subclass of this class could be used to compute noise-bias curves provided the ``__init__`` method was redefined to take in some ground truth, and the run method was redefined to compare the obj to the ground truth.



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   callback/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.callbacks.CallBack




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



