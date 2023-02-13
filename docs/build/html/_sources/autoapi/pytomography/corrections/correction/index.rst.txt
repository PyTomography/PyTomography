:py:mod:`pytomography.corrections.correction`
=============================================

.. py:module:: pytomography.corrections.correction


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.corrections.correction.CorrectionNet




.. py:class:: CorrectionNet(object_meta, image_meta, device)

   Bases: :py:obj:`torch.nn.Module`

   Correction net is the parent class for all correction networks used in reconstruction. It must take in the object/image metadata, and the corresponding pytorch device used for computation

   :param object_meta: Metadata for object space.
   :type object_meta: ObjectMeta
   :param image_meta: Metadata for image space.
   :type image_meta: ImageMeta
   :param device: Pytorch device used for computation
   :type device: str

   .. py:method:: forward()
      :abstractmethod:

      Must be implemented by the child class correction network




