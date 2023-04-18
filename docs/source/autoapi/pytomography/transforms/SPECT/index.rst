:py:mod:`pytomography.transforms.SPECT`
=======================================

.. py:module:: pytomography.transforms.SPECT


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   atteunation/index.rst
   psf/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.transforms.SPECT.SPECTAttenuationTransform
   pytomography.transforms.SPECT.SPECTPSFTransform




.. py:class:: SPECTAttenuationTransform(CT)

   Bases: :py:obj:`pytomography.transforms.Transform`

   obj2obj transform used to model the effects of attenuation in SPECT.

   :param CT: Tensor of size [batch_size, Lx, Ly, Lz] corresponding to the attenuation coefficient in :math:`{\text{cm}^{-1}}` at the photon energy corresponding to the particular scan
   :type CT: torch.tensor

   .. py:method:: __call__(object_i, i, norm_constant = None)

      Applies attenuation modeling to an object that's being detected on the right of its first axis.

      :param object_i: Tensor of size [batch_size, Lx, Ly, Lz] being projected along ``axis=1``.
      :type object_i: torch.tensor
      :param i: The projection index: used to find the corresponding angle in image space corresponding to ``object_i``. In particular, the x axis (tensor `axis=1`) of the object is aligned with the detector at angle i.
      :type i: int
      :param norm_constant: A tensor used to normalize the output during back projection. Defaults to None.
      :type norm_constant: torch.tensor, optional

      :returns: Tensor of size [batch_size, Lx, Ly, Lz] such that projection of this tensor along the first axis corresponds to an attenuation corrected projection.
      :rtype: torch.tensor



.. py:class:: SPECTPSFTransform(psf_meta)

   Bases: :py:obj:`pytomography.transforms.Transform`

   obj2obj transform used to model the effects of PSF blurring in SPECT. The smoothing kernel used to apply PSF modeling uses a Gaussian kernel with width :math:`\sigma` dependent on the distance of the point to the detector; that information is specified in the ``PSFMeta`` parameter.

   :param psf_meta: Metadata corresponding to the parameters of PSF blurring
   :type psf_meta: PSFMeta

   .. py:method:: configure(object_meta, image_meta)

      Function used to initalize the transform using corresponding object and image metadata

      :param object_meta: Object metadata.
      :type object_meta: ObjectMeta
      :param image_meta: Image metadata.
      :type image_meta: ImageMeta


   .. py:method:: compute_kernel_size()

      Function used to compute the kernel size used for PSF blurring. In particular, uses the ``max_sigmas`` attribute of ``PSFMeta`` to determine what the kernel size should be such that the kernel encompasses at least ``max_sigmas`` at all points in the object.

      :returns: The corresponding kernel size used for PSF blurring.
      :rtype: int


   .. py:method:: get_sigma(radius, dx, shape, collimator_slope, collimator_intercept)

      Uses PSF Meta data information to get blurring :math:`\sigma` as a function of distance from detector. It is assumed that ``sigma=collimator_slope*d + collimator_intercept`` where :math:`d` is the distance from the detector.

      :param radius: The distance from the detector
      :type radius: float
      :param dx: Transaxial plane pixel spacing
      :type dx: float
      :param shape: Tuple containing (Lx, Ly, Lz): dimensions of object space
      :type shape: tuple
      :param collimator_slope: See collimator intercept
      :type collimator_slope: float
      :param collimator_intercept: Collimator slope and collimator intercept are defined such that sigma(d) = collimator_slope*d + collimator_intercept
      :type collimator_intercept: float
      :param where sigma corresponds to sigma of a Gaussian function that characterizes blurring as a function of distance from the detector.:

      :returns: An array of length Lx corresponding to blurring at each point along the 1st axis in object space
      :rtype: array


   .. py:method:: __call__(object_i, i, norm_constant = None)

      Applies PSF modeling for the situation where an object is being detector by a detector at the :math:`+x` axis.

      :param object_i: Tensor of size [batch_size, Lx, Ly, Lz] being projected along its first axis
      :type object_i: torch.tensor
      :param i: The projection index: used to find the corresponding angle in image space corresponding to ``object_i``. In particular, the x axis (tensor `axis=1`) of the object is aligned with the detector at angle i.
      :type i: int
      :param norm_constant: A tensor used to normalize the output during back projection. Defaults to None.
      :type norm_constant: torch.tensor, optional

      :returns: Tensor of size [batch_size, Lx, Ly, Lz] such that projection of this tensor along the first axis corresponds to n PSF corrected projection.
      :rtype: torch.tensor



