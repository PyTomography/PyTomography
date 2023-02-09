:py:mod:`pytomography.utils`
============================

.. py:module:: pytomography.utils


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   helper_functions/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.utils.rev_cumsum
   pytomography.utils.rotate_detector_z
   pytomography.utils.get_distance
   pytomography.utils.compute_pad_size
   pytomography.utils.pad_image
   pytomography.utils.pad_object
   pytomography.utils.unpad_image
   pytomography.utils.unpad_object



.. py:function:: rev_cumsum(x)

   Reverse cumulative sum along the first axis of a tensor of shape [batch_size, Lx, Ly, Lz].
   since this is used with CT correction, the initial voxel only contributes 1/2.

   :param x: Tensor to be summed
   :type x: torch.tensor[batch_size,Lx,Ly,Lz]

   :returns: The cumulatively summed tensor.
   :rtype: torch.tensor[batch_size, Lx, Ly, Lz]


.. py:function:: rotate_detector_z(x, angle, interpolation=InterpolationMode.BILINEAR, negative=False)

   Returns am object tensor in a rotated reference frame such that the scanner is located
   at the +x axis. Note that the scanner angle $eta$ is related to $\phi$ (azimuthal angle)
   by $\phi = 3\pi/2 - eta$.

   :param x: Tensor aligned with cartesian coordinate system specified
   :type x: torch.tensor[batch_size, Lx, Ly, Lz]
   :param by the manual.:
   :param angle: The angle $eta$ where the scanner is located.
   :type angle: float
   :param interpolation: Method of interpolation used to get rotated image.
   :type interpolation: InterpolationMode, optional
   :param Defaults to InterpolationMode.BILINEAR.:
   :param negative: If True, applies an inverse rotation. In this case, the tensor
   :type negative: bool, optional
   :param x is an object in a coordinate system aligned with $eta$:
   :param and the function rotates the:
   :param x back to the original cartesian coordinate system specified by the users manual. In particular:
   :param if one:
   :param uses this function on a tensor with negative=False:
   :param then applies this function to that returned:
   :param tensor with negative=True:
   :param it should return the same tensor. Defaults to False.:

   :returns: Rotated tensor.
   :rtype: torch.tensor[batch_size, Lx, Ly, Lz]


.. py:function:: get_distance(Lx, r, dx)

   Given the radial distance to center of object space from the scanner, computes the distance
     between each parallel plane (i.e. (y-z plane)) and a detector located at +x. This function is
     used for point spread function (PSF) blurring where the amount of blurring depends on the
     distance from the detector.

   :param Lx: The number of y-z planes to compute the distance of
   :type Lx: int
   :param r: The radial distance between the central y-z plane and the detector at +x.
   :type r: float
   :param dx: The spacing between y-z planes in Euclidean distance.
   :type dx: float

   :returns: An array of distances for each y-z plane to the detector.
   :rtype: np.array[Lx]


.. py:function:: compute_pad_size(width)


.. py:function:: pad_image(image)


.. py:function:: pad_object(object)


.. py:function:: unpad_image(image, original_shape)


.. py:function:: unpad_object(object, original_shape)


