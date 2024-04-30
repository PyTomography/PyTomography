:py:mod:`pytomography.metadata.metadata`
========================================

.. py:module:: pytomography.metadata.metadata


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.metadata.metadata.ObjectMeta
   pytomography.metadata.metadata.ProjMeta




.. py:class:: ObjectMeta(dr, shape)

   Parent class for all different types of Object Space Metadata. In general, while this is fairly similar for all imaging modalities, required padding features/etc may be different for different modalities.


   .. py:method:: __repr__()

      Return repr(self).



.. py:class:: ProjMeta(angles)

   Parent class for all different types of Projection Space Metadata. Implementation and required parameters will differ significantly between different imaging modalities.


   .. py:method:: __repr__()

      Return repr(self).



