:py:mod:`pytomography.metadata.PET.pet_sinogram_metadata`
=========================================================

.. py:module:: pytomography.metadata.PET.pet_sinogram_metadata


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.metadata.PET.pet_sinogram_metadata.PETSinogramPolygonProjMeta




.. py:class:: PETSinogramPolygonProjMeta(info, tof_meta = None)

   PET Sinogram metadata class for polygonal scanner geometry

   :param info: PET geometry information dictionary
   :type info: dict
   :param tof_meta: PET time of flight metadata. If None, then assumes no time of flight. Defaults to None.
   :type tof_meta: PETTOFMeta | None, optional


