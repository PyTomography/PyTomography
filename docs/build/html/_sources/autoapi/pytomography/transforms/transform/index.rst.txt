:py:mod:`pytomography.transforms.transform`
===========================================

.. py:module:: pytomography.transforms.transform


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.transforms.transform.Transform




.. py:class:: Transform

   The parent class for all transforms used in reconstruction (obj2obj, im2im, obj2im). Subclasses must implement the ``__call__`` method.

   :param device: Pytorch device used for computation
   :type device: str

   .. py:method:: configure(object_meta, proj_meta)

      Configures the transform to the object/proj metadata. This is done after creating the network so that it can be adjusted to the system matrix.

      :param object_meta: Object metadata.
      :type object_meta: ObjectMeta
      :param proj_meta: Projections metadata.
      :type proj_meta: ProjMeta


   .. py:method:: forward(x)
      :abstractmethod:

      Abstract method; must be implemented in subclasses to apply a correction to an object/proj and return it



   .. py:method:: backward(x)
      :abstractmethod:

      Abstract method; must be implemented in subclasses to apply a correction to an object/proj and return it




