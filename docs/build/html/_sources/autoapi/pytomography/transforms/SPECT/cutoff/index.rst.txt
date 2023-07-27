:py:mod:`pytomography.transforms.SPECT.cutoff`
==============================================

.. py:module:: pytomography.transforms.SPECT.cutoff


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.transforms.SPECT.cutoff.CutOffTransform




.. py:class:: CutOffTransform(image)

   Bases: :py:obj:`pytomography.transforms.Transform`

   im2im transformation used to set pixel values equal to zero at the first and last few z slices. This is often required when reconstructing DICOM data due to the finite field of view of the projection data, where additional axial slices are included on the top and bottom, with zero measured detection events. This transform is included in the system matrix, to model the sharp cutoff at the finite FOV.

   :param image: Measured image data.
   :type image: torch.tensor

   .. py:method:: forward(image)

      Forward projection :math:`B:\mathbb{V} \to \mathbb{V}` of the cutoff transform.

      :param image: Tensor of size [batch_size, Ltheta, Lr, Lz] which transform is appplied to
      :type image: torch.Tensor

      :returns: Original image, but with certain z-slices equal to zero.
      :rtype: torch.tensor


   .. py:method:: backward(image, norm_constant = None)

      Back projection :math:`B^T:\mathbb{V} \to \mathbb{V}` of the cutoff transform. Since this is a diagonal matrix, the implementation is the same as forward projection, but with the optional `norm_constant` argument.

      :param image: Tensor of size [batch_size, Ltheta, Lr, Lz] which transform is appplied to
      :type image: torch.Tensor
      :param norm_constant: A tensor used to normalize the output during back projection. Defaults to None.
      :type norm_constant: torch.Tensor | None, optional

      :returns: Original image, but with certain z-slices equal to zero.
      :rtype: torch.tensor



