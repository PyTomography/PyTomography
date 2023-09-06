:py:mod:`pytomography.utils.spatial`
====================================

.. py:module:: pytomography.utils.spatial


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.utils.spatial.rotate_detector_z
   pytomography.utils.spatial.compute_pad_size
   pytomography.utils.spatial.compute_pad_size_padded
   pytomography.utils.spatial.pad_object
   pytomography.utils.spatial.unpad_object
   pytomography.utils.spatial.pad_proj
   pytomography.utils.spatial.unpad_proj
   pytomography.utils.spatial.pad_object_z
   pytomography.utils.spatial.unpad_object_z



.. py:function:: rotate_detector_z(x, angles, mode = 'bilinear', negative = False)

   Returns an object tensor in a rotated reference frame such that the scanner is located at the +x axis. Note that the scanner angle :math:`\beta` is related to :math:`\phi` (azimuthal angle) by :math:`\phi = 3\pi/2 - \beta`.

   :param x: Tensor aligned with cartesian coordinate system specified
   :type x: torch.tensor[batch_size, Lx, Ly, Lz]
   :param by the manual.:
   :param angles: The angles :math:`\beta` where the scanner is located for each element in the batch x.
   :type angles: torch.Tensor
   :param mode: Method of interpolation used to get rotated object. Defaults to bilinear.
   :type mode: str, optional
   :param negative: If True, applies an inverse rotation. In this case, the tensor
   :type negative: bool, optional
   :param x is an object in a coordinate system aligned with :math:`\beta`:
   :param and the function rotates the:
   :param x back to the original cartesian coordinate system specified by the users manual. In particular:
   :param if one:
   :param uses this function on a tensor with negative=False:
   :param then applies this function to that returned:
   :param tensor with negative=True:
   :param it should return the same tensor. Defaults to False.:

   :returns: Rotated tensor.
   :rtype: torch.tensor[batch_size, Lx, Ly, Lz]


.. py:function:: compute_pad_size(width)

   Computes the pad width required such that subsequent rotation retains the entire object

   :param width: width of the corresponding axis (i.e. number of elements in the dimension)
   :type width: int

   :returns: the number of pixels by which the axis needs to be padded on each side
   :rtype: int


.. py:function:: compute_pad_size_padded(width)

   Computes the width by which an object was padded, given its padded size.

   :param width: width of the corresponding axis (i.e. number of elements in the dimension)
   :type width: int

   :returns: the number of pixels by which the object was padded to get to this width
   :rtype: int


.. py:function:: pad_object(object, mode='constant')

   Pads object tensors by enough pixels in the xy plane so that subsequent rotations don't crop out any of the object

   :param object: object tensor to be padded
   :type object: torch.Tensor[batch_size, Lx, Ly, Lz]
   :param mode: _description_. Defaults to 'constant'.
   :type mode: str, optional

   :returns: _description_
   :rtype: _type_


.. py:function:: unpad_object(object)

   Unpads a padded object tensor in the xy plane back to its original dimensions

   :param object: padded object tensor
   :type object: torch.Tensor[batch_size, Lx', Ly', Lz]

   :returns: Object tensor back to it's original dimensions.
   :rtype: torch.Tensor[batch_size, Lx, Ly, Lz]


.. py:function:: pad_proj(proj, mode = 'constant', value = 0)

   Pads projections along the Lr axis

   :param proj: Projections tensor.
   :type proj: torch.Tensor[batch_size, Ltheta, Lr, Lz]
   :param mode: Padding mode to use. Defaults to 'constant'.
   :type mode: str, optional
   :param value: If padding mode is constant, fill with this value. Defaults to 0.
   :type value: float, optional

   :returns: Padded projections tensor.
   :rtype: torch.Tensor[batch_size, Ltheta, Lr', Lz]


.. py:function:: unpad_proj(proj)

   Unpads the projections back to original Lr dimensions

   :param proj: Padded projections tensor
   :type proj: torch.Tensor[batch_size, Ltheta, Lr', Lz]

   :returns: Unpadded projections tensor
   :rtype: torch.Tensor[batch_size, Ltheta, Lr, Lz]


.. py:function:: pad_object_z(object, pad_size, mode='constant')

   Pads an object tensor along z. Useful for PSF modeling

   :param object: Object tensor
   :type object: torch.Tensor[batch_size, Lx, Ly, Lz]
   :param pad_size: Amount by which to pad in -z and +z
   :type pad_size: int
   :param mode: Padding mode. Defaults to 'constant'.
   :type mode: str, optional

   :returns: Padded object tensor along z.
   :rtype: torch.Tensor[torch.Tensor[batch_size, Lx, Ly, Lz']]


.. py:function:: unpad_object_z(object, pad_size)

   Unpads an object along the z dimension

   :param object: Padded object tensor along z.
   :type object: torch.Tensor[batch_size, Lx, Ly, Lz']
   :param pad_size: Amount by which the padded tensor was padded in the z direcion
   :type pad_size: int

   :returns: Unpadded object tensor.
   :rtype: torch.Tensor[batch_size, Lx, Ly, Lz]


