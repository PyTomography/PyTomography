:py:mod:`pytomography.transforms.shared.motion`
===============================================

.. py:module:: pytomography.transforms.shared.motion


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.transforms.shared.motion.DVFMotionTransform




.. py:class:: DVFMotionTransform(dvf_forward = None, dvf_backward = None)

   Bases: :py:obj:`pytomography.transforms.Transform`

   The parent class for all transforms used in reconstruction (obj2obj, im2im, obj2im). Subclasses must implement the ``__call__`` method.

   :param device: Pytorch device used for computation
   :type device: str

   .. py:method:: _get_vol_ratio(DVF)


   .. py:method:: _get_old_coordinates()

      Obtain meshgrid of coordinates corresponding to the object

      :returns: Tensor of coordinates corresponding to input object
      :rtype: torch.Tensor


   .. py:method:: _get_new_coordinates(old_coordinates, DVF)

      Obtain the new coordinates of each voxel based on the DVF.

      :param old_coordinates: Old coordinates of each voxel
      :type old_coordinates: torch.Tensor
      :param DVF: Deformation vector field.
      :type DVF: torch.Tensor

      :returns: _description_
      :rtype: _type_


   .. py:method:: _apply_dvf(DVF, vol_ratio, object_i)

      Applies the deformation vector field to the object

      :param DVF: Deformation vector field
      :type DVF: torch.Tensor
      :param object_i: Old object.
      :type object_i: torch.Tensor

      :returns: Deformed object.
      :rtype: torch.Tensor


   .. py:method:: forward(object_i)

      Forward transform of deformation vector field

      :param object_i: Original object.
      :type object_i: torch.Tensor

      :returns: Deformed object corresponding to forward transform.
      :rtype: torch.Tensor


   .. py:method:: backward(object_i)

      Backward transform of deformation vector field

      :param object_i: Original object.
      :type object_i: torch.Tensor

      :returns: Deformed object corresponding to backward transform.
      :rtype: torch.Tensor



