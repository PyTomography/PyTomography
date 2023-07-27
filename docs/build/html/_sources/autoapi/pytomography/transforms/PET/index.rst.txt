:py:mod:`pytomography.transforms.PET`
=====================================

.. py:module:: pytomography.transforms.PET


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   attenuation/index.rst
   psf/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.transforms.PET.PETAttenuationTransform
   pytomography.transforms.PET.PETPSFTransform




.. py:class:: PETAttenuationTransform(CT)

   Bases: :py:obj:`pytomography.transforms.Transform`

   im2im mapping used to model the effects of attenuation in PET.

   :param CT: Tensor of size [batch_size, Lx, Ly, Lz] corresponding to the attenuation coefficient in :math:`{\text{cm}^{-1}}` at a photon energy of 511keV.
   :type CT: torch.tensor
   :param device: Pytorch device used for computation. If None, uses the default device `pytomography.device` Defaults to None.
   :type device: str, optional

   .. py:method:: configure(object_meta, image_meta)

      Function used to initalize the transform using corresponding object and image metadata

      :param object_meta: Object metadata.
      :type object_meta: ObjectMeta
      :param image_meta: Image metadata.
      :type image_meta: ImageMeta


   .. py:method:: forward(image)

      Applies forward projection of attenuation modeling :math:`B:\mathbb{V} \to \mathbb{V}` to a 2D PET image.

      :param image: Tensor of size [batch_size, Ltheta, Lr, Lz] which transform is appplied to
      :type image: torch.Tensor

      :returns: Tensor of size [batch_size, Ltheta, Lr, Lz]  corresponding to attenuation-corrected image.
      :rtype: torch.Tensor


   .. py:method:: backward(image, norm_constant = None)

      Applies back projection of attenuation modeling :math:`B^T:\mathbb{V} \to \mathbb{V}` to a 2D PET image. Since the matrix is diagonal, its the ``backward`` implementation is identical to the ``forward`` implementation; the only difference is the optional ``norm_constant`` which is needed if one wants to normalize the back projection.

      :param image: Tensor of size [batch_size, Ltheta, Lr, Lz] which transform is appplied to
      :type image: torch.Tensor
      :param norm_constant: A tensor used to normalize the output during back projection. Defaults to None.
      :type norm_constant: torch.Tensor | None, optional

      :returns: Tensor of size [batch_size, Ltheta, Lr, Lz]  corresponding to attenuation-corrected image.
      :rtype: torch.tensor



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



