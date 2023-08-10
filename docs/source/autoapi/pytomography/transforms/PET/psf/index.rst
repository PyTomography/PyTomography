:py:mod:`pytomography.transforms.PET.psf`
=========================================

.. py:module:: pytomography.transforms.PET.psf


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.transforms.PET.psf.PETPSFTransform



Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.transforms.PET.psf.kernel_noncol
   pytomography.transforms.PET.psf.kernel_penetration
   pytomography.transforms.PET.psf.kernel_scattering



.. py:class:: PETPSFTransform(kerns)

   Bases: :py:obj:`pytomography.transforms.Transform`

   im2im transform used to model the effects of PSF blurring in PET. The smoothing kernel is assumed to be independent of :math:`\theta` and :math:`z`, but is dependent on :math:`r`.

   :param kerns: A sequence of PSF kernels applied to the Lr dimension of the image with shape [batch_size, Lr, Ltheta, Lz]
   :type kerns: Sequence[callable]

   .. py:method:: configure(object_meta, image_meta)

      Function used to initalize the transform using corresponding object and image metadata

      :param object_meta: Object metadata.
      :type object_meta: ObjectMeta
      :param image_meta: Image metadata.
      :type image_meta: ImageMeta


   .. py:method:: construct_matrix()

      Constructs the matrix used to apply PSF blurring.



   .. py:method:: forward(image)

      Applies the forward projection of PSF modeling :math:`B:\mathbb{V} \to \mathbb{V}` to a PET image.

      :param image: Tensor of size [batch_size, Ltheta, Lr, Lz] corresponding to the image
      :type image: torch.tensor]

      :returns: Tensor of size [batch_size, Ltheta, Lr, Lz] corresponding to the PSF corrected image.
      :rtype: torch.tensor


   .. py:method:: backward(image, norm_constant = None)

      Applies the back projection of PSF modeling :math:`B^T:\mathbb{V} \to \mathbb{V}` to a PET image.

      :param image: Tensor of size [batch_size, Ltheta, Lr, Lz] corresponding to the image
                    norm_constant (torch.tensor, optional): A tensor used to normalize the output during back projection. Defaults to None.
      :type image: torch.tensor]

      :returns: Tensor of size [batch_size, Ltheta, Lr, Lz] corresponding to the PSF corrected image.
      :rtype: torch.tensor



.. py:function:: kernel_noncol(x, r, R, delta=1e-08)


.. py:function:: kernel_penetration(x, r, R, mu=0.87, delta=1e-08)


.. py:function:: kernel_scattering(x, r, R, scatter_fact=0.327, delta=1e-08)


