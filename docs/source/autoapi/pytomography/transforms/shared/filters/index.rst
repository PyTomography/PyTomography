:py:mod:`pytomography.transforms.shared.filters`
================================================

.. py:module:: pytomography.transforms.shared.filters


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.transforms.shared.filters.GaussianFilter




.. py:class:: GaussianFilter(FWHM, n_sigmas = 3)

   Bases: :py:obj:`pytomography.transforms.Transform`

   Applies a Gaussian smoothing filter to the reconstructed object with the specified full-width-half-max (FWHM)

   :param FWHM: Specifies the width of the gaussian
   :type FWHM: float
   :param n_sigmas: Number of sigmas to include before truncating the kernel.
   :type n_sigmas: float

   .. py:method:: configure(object_meta, proj_meta)

      Configures the transform to the object/proj metadata. This is done after creating the network so that it can be adjusted to the system matrix.

      :param object_meta: Object metadata.
      :type object_meta: ObjectMeta
      :param proj_meta: Projections metadata.
      :type proj_meta: ProjMeta


   .. py:method:: _get_kernels()

      Obtains required kernels for smoothing



   .. py:method:: __call__(object)

      Alternative way to call


   .. py:method:: forward(object)

      Applies the Gaussian smoothing

      :param object: Object to smooth
      :type object: torch.tensor

      :returns: Smoothed object
      :rtype: torch.tensor


   .. py:method:: backward(object, norm_constant=None)

      Applies Gaussian smoothing in back projection. Because the operation is symmetric, it is the same as the forward projection.

      :param object: Object to smooth
      :type object: torch.tensor
      :param norm_constant: Normalization constant used in iterative algorithms. Defaults to None.
      :type norm_constant: torch.tensor, optional

      :returns: Smoothed object
      :rtype: torch.tensor



