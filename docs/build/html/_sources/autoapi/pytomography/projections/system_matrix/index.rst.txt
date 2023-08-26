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

   Abstract class for a general system matrix :math:`H:\mathbb{U} \to \mathbb{V}` which takes in an object :math:`f \in \mathbb{U}` and maps it to a corresponding image :math:`g \in \mathbb{V}` that would be produced by the imaging system. A system matrix consists of sequences of object-to-object and image-to-image transforms that model various characteristics of the imaging system, such as attenuation and blurring. While the class implements the operator :math:`H:\mathbb{U} \to \mathbb{V}` through the ``forward`` method, it also implements :math:`H^T:\mathbb{V} \to \mathbb{U}` through the `backward` method, required during iterative reconstruction algorithms such as OSEM.

   :param obj2obj_transforms: Sequence of object mappings that occur before forward projection.
   :type obj2obj_transforms: Sequence[Transform]
   :param im2im_transforms: Sequence of image mappings that occur after forward projection.
   :type im2im_transforms: Sequence[Transform]
   :param object_meta: Object metadata.
   :type object_meta: ObjectMeta
   :param image_meta: Image metadata.
   :type image_meta: ImageMeta

   .. py:method:: initialize_transforms()

      Initializes all transforms used to build the system matrix



   .. py:method:: forward(object, **kwargs)
      :abstractmethod:

      Implements forward projection :math:`Hf` on an object :math:`f`.

      :param object: The object to be forward projected
      :type object: torch.tensor[batch_size, Lx, Ly, Lz]
      :param angle_subset: Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.
      :type angle_subset: list, optional

      :returns: Forward projected image where Ltheta is specified by `self.image_meta` and `angle_subset`.
      :rtype: torch.tensor[batch_size, Ltheta, Lx, Lz]


   .. py:method:: backward(image, angle_subset = None, return_norm_constant = False)
      :abstractmethod:

      Implements back projection :math:`H^T g` on an image :math:`g`.

      :param image: image which is to be back projected
      :type image: torch.Tensor
      :param angle_subset: Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.
      :type angle_subset: list, optional
      :param return_norm_constant: Whether or not to return :math:`1/\sum_j H_{ij}` along with back projection. Defaults to 'False'.
      :type return_norm_constant: bool

      :returns: the object obtained from back projection.
      :rtype: torch.tensor[batch_size, Lr, Lr, Lz]



