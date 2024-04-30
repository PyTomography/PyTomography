:py:mod:`pytomography.metadata`
===============================

.. py:module:: pytomography.metadata

.. autoapi-nested-parse::

   This module contains classes pertaining to metadata in PyTomography. Metadata classes contain required information for interpretting data; for example, metadata corresponding to an object (with object data stored in a ``torch.Tensor``) contains the voxel spacing and voxel dimensions.



Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   PET/index.rst
   SPECT/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   metadata/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.metadata.ObjectMeta
   pytomography.metadata.ProjMeta




.. py:class:: ObjectMeta(dr, shape)

   Parent class for all different types of Object Space Metadata. In general, while this is fairly similar for all imaging modalities, required padding features/etc may be different for different modalities.


   .. py:method:: __repr__()

      Return repr(self).



.. py:class:: ProjMeta(angles)

   Parent class for all different types of Projection Space Metadata. Implementation and required parameters will differ significantly between different imaging modalities.


   .. py:method:: __repr__()

      Return repr(self).



