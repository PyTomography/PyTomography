:py:mod:`pytomography.callbacks.data_saving`
============================================

.. py:module:: pytomography.callbacks.data_saving


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.callbacks.data_saving.DataStorageCallback




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



