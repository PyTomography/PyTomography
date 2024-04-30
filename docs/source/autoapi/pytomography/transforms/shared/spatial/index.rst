:py:mod:`pytomography.transforms.shared.spatial`
================================================

.. py:module:: pytomography.transforms.shared.spatial


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.transforms.shared.spatial.RotationTransform




.. py:class:: RotationTransform(mode = 'bilinear')

   Bases: :py:obj:`pytomography.transforms.Transform`

   obj2obj transform used to rotate an object to angle :math:`\beta` in the DICOM reference frame. (Note that an angle of )

   :param mode: Interpolation mode used in the rotation.
   :type mode: str

   .. py:method:: forward(object, angles)

      Rotates an object to angle :math:`\beta` in the DICOM reference frame. Note that the scanner angle :math:`\beta` is related to :math:`\phi` (azimuthal angle) by :math:`\phi = 3\pi/2 - \beta`.

      :param object: Tensor of size [Lx, Ly, Lz] being rotated.
      :type object: torch.tensor
      :param angles: Tensor of size 1 corresponding to the rotation angle.
      :type angles: torch.Tensor

      :returns: Tensor of size [Lx, Ly, Lz] which is rotated
      :rtype: torch.tensor


   .. py:method:: backward(object, angles)

      Forward projection :math:`A:\mathbb{U} \to \mathbb{U}` of attenuation correction.

      :param object: Tensor of size [Lx, Ly, Lz] being rotated.
      :type object: torch.tensor
      :param angles: Tensor of size 1 corresponding to the rotation angle.
      :type angles: torch.Tensor

      :returns: Tensor of size [Lx, Ly, Lz] which is rotated.
      :rtype: torch.tensor



