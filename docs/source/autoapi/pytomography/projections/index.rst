:py:mod:`pytomography.projections`
==================================

.. py:module:: pytomography.projections


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   back_projection/index.rst
   forward_projection/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.projections.ForwardProjectionNet
   pytomography.projections.BackProjectionNet




.. py:class:: ForwardProjectionNet(object_correction_nets, image_correction_nets, object_meta, image_meta, device='cpu')

   Bases: :py:obj:`torch.nn.Module`

   Implements a forward projection of mathematical form :math:`g_j = \sum_{i} c_{ij} f_i` where :math:`f_i` is an object, :math:`g_j` is the corresponding image, and :math:`c_{ij}` is the system matrix given by the various phenonemon modeled (atteunation correction/PSF).


   .. py:method:: forward(object, angle_subset=None)

      Implements forward projection on an object

      :param object: The object to be forward projected
      :type object: torch.tensor[batch_size, Lx, Ly, Lz]
      :param angle_subset: Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None,
      :type angle_subset: list, optional
      :param which assumes all angles are used.:

      :returns: Forward projected image where Ltheta is specified by `self.image_meta` and `angle_subset`.
      :rtype: torch.tensor[batch_size, Ltheta, Lx, Lz]



.. py:class:: BackProjectionNet(object_correction_nets, image_correction_nets, object_meta, image_meta, device='cpu')

   Bases: :py:obj:`torch.nn.Module`

   Implements a back projection of mathematical form :math:`f_i = \frac{1}{\sum_j c_{ij}}\sum_{j} c_{ij} g_j`. where :math:`f_j` is an object, :math:`g_j` is an image, and :math:`c_{ij}` is the system matrix given by the various phenonemon modeled (atteunation correction/PSF).

   .. py:method:: forward(image, angle_subset=None, prior=None, delta=1e-11)

      Implements back projection on an image, returning an object.

      :param image: image which is to be back projected
      :type image: torch.tensor[batch_size, Ltheta, Lr, Lz]
      :param angle_subset: Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.
      :type angle_subset: list, optional
      :param prior: If included, modifes normalizing factor to :math:`\frac{1}{\sum_j c_{ij} + P_i}` where :math:`P_i` is given by the prior. Used, for example, during in MAP OSEM. Defaults to None.
      :type prior: Prior, optional
      :param delta: Prevents division by zero when dividing by normalizing constant. Defaults to 1e-11.
      :type delta: float, optional

      :returns: the object obtained from back projection.
      :rtype: torch.tensor[batch_size, Lr, Lr, Lz]



