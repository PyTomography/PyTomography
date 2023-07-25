:py:mod:`pytomography.metadata.metadata`
========================================

.. py:module:: pytomography.metadata.metadata


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.metadata.metadata.ObjectMeta
   pytomography.metadata.metadata.ImageMeta
   pytomography.metadata.metadata.PSFMeta




.. py:class:: ObjectMeta(dr, shape)

   Metadata for object space

   :param dr: List of 3 elements specifying voxel dimensions in cm.
   :type dr: list[float]
   :param shape: List of 3 elements [Lx, Ly, Lz] specifying the length of each dimension.
   :type shape: list[int]

   .. py:method:: compute_padded_shape()

      Computes the padded shape of an object required when rotating the object (to avoid anything getting cut off).



   .. py:method:: __repr__()

      Return repr(self).



.. py:class:: ImageMeta(object_meta, angles, radii=None)

   Metadata for image space

   :param object_meta: Corresponding object space metadata
   :type object_meta: ObjectMeta
   :param angles: Specifies the detector angles for all projections in image space
   :type angles: list
   :param radii: Specifies the radial distance of the detector corresponding to each angle in `angles`; only required in certain cases (i.e. PSF correction). Defaults to None.
   :type radii: list, optional

   .. py:method:: compute_padded_shape()

      Computes the padded shape of an object required when rotating the object (to avoid anything getting cut off).



   .. py:method:: __repr__()

      Return repr(self).



.. py:class:: PSFMeta(sigma_fit_params, sigma_fit = lambda r, a, b: a * r + b, kernel_dimensions = '2D', min_sigmas = 3)

   Metadata for PSF correction. PSF blurring is implemented using Gaussian blurring with :math:`\sigma(r) = f(r,p)` where :math:`r` is the distance from the detector, :math`\sigma` is the width of the Gaussian blurring at that location, and :math:`f(r,p)` is the `sigma_fit` function which takes in additional parameters :math:`p` called `sigma_fit_params`. (By default, `sigma_fit` is a linear curve). As such, :math:`\frac{1}{\sigma\sqrt{2\pi}}e^{-r^2/(2\sigma(r)^2)}` is the point spread function. Blurring is implemented using convolutions with a specified kernel size.

   :param sigma_fit_params: Parameters to the sigma fit function
   :type sigma_fit_params: float
   :param sigma_fit: Function used to model blurring as a function of radial distance. Defaults to a 2 parameter linear model.
   :type sigma_fit: function
   :param kernel_dimensions: If '1D', blurring is done seperately in each axial plane (so only a 1 dimensional convolution is used). If '2D', blurring is mixed between axial planes (so a 2D convolution is used). Defaults to '2D'.
   :type kernel_dimensions: str
   :param min_sigmas: This is the number of sigmas to consider in PSF correction. PSF are modelled by Gaussian functions whose extension is infinite, so we need to crop the Gaussian when computing this operation numerically. Note that the blurring width is depth dependent, but the kernel size used for PSF blurring is constant. As such, this parameter is used to fix the kernel size such that all locations have at least ``min_sigmas`` of a kernel size.
   :type min_sigmas: float, optional

   .. py:method:: __repr__()

      Return repr(self).



