:py:mod:`pytomography.projections.system_matrix`
================================================

.. py:module:: pytomography.projections.system_matrix


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.projections.system_matrix.SystemMatrix




.. py:class:: SystemMatrix(obj2obj_transforms, im2im_transforms, object_meta, image_meta)

   Update this

   :param obj2obj_transforms: Sequence of object mappings that occur before forward projection.
   :type obj2obj_transforms: Sequence[Transform]
   :param im2im_transforms: Sequence of image mappings that occur after forward projection.
   :type im2im_transforms: Sequence[Transform]
   :param object_meta: Object metadata.
   :type object_meta: ObjectMeta
   :param image_meta: Image metadata.
   :type image_meta: ImageMeta

   .. py:method:: initialize_correction_nets()

      Initializes all mapping networks with the required object and image metadata corresponding to the projection network.



   .. py:method:: forward(object, angle_subset = None)

      Implements forward projection :math:`Hf` on an object :math:`f`.

      :param object: The object to be forward projected
      :type object: torch.tensor[batch_size, Lx, Ly, Lz]
      :param angle_subset: Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.
      :type angle_subset: list, optional

      :returns: Forward projected image where Ltheta is specified by `self.image_meta` and `angle_subset`.
      :rtype: torch.tensor[batch_size, Ltheta, Lx, Lz]


   .. py:method:: backward(image, angle_subset = None, prior = None, normalize = False, return_norm_constant = False, delta = 1e-11)

      Implements back projection :math:`H^T g` on an image :math:`g`.

      :param image: image which is to be back projected
      :type image: torch.tensor[batch_size, Ltheta, Lr, Lz]
      :param angle_subset: Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.
      :type angle_subset: list, optional
      :param prior: If included, modifes normalizing factor to :math:`\frac{1}{\sum_j c_{ij} + P_i}` where :math:`P_i` is given by the prior. Used, for example, during in MAP OSEM. Defaults to None.
      :type prior: Prior, optional
      :param normalize: Whether or not to divide result by :math:`\sum_j c_{ij}`
      :type normalize: bool
      :param return_norm_constant: Whether or not to return :math:`1/\sum_j c_{ij}` along with back projection. Defaults to 'False'.
      :type return_norm_constant: bool
      :param delta: Prevents division by zero when dividing by normalizing constant. Defaults to 1e-11.
      :type delta: float, optional

      :returns: the object obtained from back projection.
      :rtype: torch.tensor[batch_size, Lr, Lr, Lz]



