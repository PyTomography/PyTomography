"""Metadata: what the projectors need to know about the object grid and the measured data.

Every reconstruction has two:

* an ``object_meta`` (:class:`ObjectMeta` or a subclass): the object's grid, voxel size and shape in mm, and where
  it is in the patient (``object_meta.affine``);
* a ``proj_meta`` (a subclass of :class:`ProjMeta`): the acquisition, e.g. the angles and radii of SPECT views, the
  scanner of a PET list mode acquisition, the geometry of a CT scan.

The readers in :mod:`pytomography.io` make both. The modality-specific classes are in
:mod:`pytomography.metadata.SPECT`, :mod:`pytomography.metadata.PET` and :mod:`pytomography.metadata.CT`.
"""
from .metadata import ObjectMeta, ProjMeta