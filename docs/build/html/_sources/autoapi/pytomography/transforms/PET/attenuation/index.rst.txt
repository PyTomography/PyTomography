:py:mod:`pytomography.transforms.PET.attenuation`
=================================================

.. py:module:: pytomography.transforms.PET.attenuation


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.transforms.PET.attenuation.PETAttenuationTransform



Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.transforms.PET.attenuation.get_prob_of_detection_matrix



.. py:function:: get_prob_of_detection_matrix(CT, dx)

   Converts an attenuation map of :math:`\text{cm}^{-1}` to a probability of photon detection projection (detector pair oriented along x axis). Note that this requires the attenuation map to be at the energy of photons being emitted (511keV).

   :param CT: Tensor of size [batch_size, Lx, Ly, Lz] corresponding to the attenuation coefficient in :math:`{\text{cm}^{-1}}`
   :type CT: torch.tensor
   :param dx: Axial plane pixel spacing.
   :type dx: float

   :returns: Tensor of size [batch_size, 1, Ly, Lz] corresponding to probability of photon being detected at a detector pairs oriented along the x axis.
   :rtype: torch.tensor


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



