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
   data_saving/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.callbacks.Callback
   pytomography.callbacks.DataStorageCallback




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



.. py:class:: DataStorageCallback(likelihood, object_initial)

   Bases: :py:obj:`pytomography.callbacks.callback.Callback`

   Callback that stores the object and forward projection at each iteration

   :param likelihood: Likelihood function used in reconstruction
   :type likelihood: Likelihood
   :param object_initial: Initial object in the reconstruction algorithm
   :type object_initial: torch.Tensor[Lx, Ly, Lz]

   .. py:method:: run(object, n_iter, n_subset)

      Applies the callback

      :param object: Object at current iteration
      :type object: torch.Tensor[Lx, Ly, Lz]
      :param n_iter: Current iteration number
      :type n_iter: int
      :param n_subset: Current subset index
      :type n_subset: int

      :returns: Original object passed (object is not modifed)
      :rtype: torch.Tensor


   .. py:method:: finalize(object)

      Finalizes the callback after all iterations are called

      :param object: Reconstructed object (all iterations/subsets completed)
      :type object: torch.Tensor[Lx, Ly, Lz]



