:py:mod:`pytomography.utils.misc`
=================================

.. py:module:: pytomography.utils.misc


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.utils.misc.rev_cumsum
   pytomography.utils.misc.get_distance
   pytomography.utils.misc.get_object_nearest_neighbour
   pytomography.utils.misc.print_collimator_parameters
   pytomography.utils.misc.check_if_class_contains_method
   pytomography.utils.misc.get_1d_gaussian_kernel



.. py:function:: rev_cumsum(x)

   Reverse cumulative sum along the first axis of a tensor of shape [Lx, Ly, Lz].
   since this is used with SPECT attenuation correction, the initial voxel only contributes 1/2.

   :param x: Tensor to be summed
   :type x: torch.tensor[Lx,Ly,Lz]

   :returns: The cumulatively summed tensor.
   :rtype: torch.tensor[Lx, Ly, Lz]


.. py:function:: get_distance(Lx, r, dx)

   Given the radial distance to center of object space from the scanner, computes the distance between each parallel plane (i.e. (y-z plane)) and a detector located at +x. This function is used for SPECT PSF modeling where the amount of blurring depends on thedistance from the detector.

   :param Lx: The number of y-z planes to compute the distance of
   :type Lx: int
   :param r: The radial distance between the central y-z plane and the detector at +x.
   :type r: float
   :param dx: The spacing between y-z planes in Euclidean distance.
   :type dx: float

   :returns: An array of distances for each y-z plane to the detector.
   :rtype: np.array[Lx]


.. py:function:: get_object_nearest_neighbour(object, shifts)

   Given an object tensor, finds the nearest neighbour (corresponding to ``shifts``) for each voxel (done by shifting object by i,j,k)

   :param object: Original object
   :type object: torch.Tensor
   :param shifts: List of three integers [i,j,k] corresponding to neighbour location
   :type shifts: list[int]

   :returns: Shifted object whereby each voxel corresponding to neighbour [i,j,k] of the ``object``.
   :rtype: torch.Tensor


.. py:function:: print_collimator_parameters()

   Prints all the available SPECT collimator parameters



.. py:function:: check_if_class_contains_method(instance, method_name)

   Checks if class corresponding to instance implements the method ``method_name``

   :param instance: A python object
   :type instance: Object
   :param method_name: Name of the method of the object being checked
   :type method_name: str


.. py:function:: get_1d_gaussian_kernel(sigma, kernel_size, padding_mode='zeros')

   Returns a 1D gaussian blurring kernel

   :param sigma: Sigma (in pixels) of blurring pixels
   :type sigma: float
   :param kernel_size: Size of kernel used
   :type kernel_size: int
   :param padding_mode: Type of padding. Defaults to 'zeros'.
   :type padding_mode: str, optional

   :returns: Torch Conv1d layer corresponding to the gaussian kernel
   :rtype: Conv1d


