"""Reading data from scanners and simulations into PyTomography's standard form, and saving images.

Every object is on a grid in DICOM's patient frame (LPS, mm) described by its ``object_meta``; ``object_meta.affine``
says where each voxel is. :class:`Image` is an image with its own geometry, and the writers :func:`save_dicom` and
:func:`save_nifti` take the place of the voxels from ``object_meta`` or from an :class:`Image`."""

from ._image import Image
from .shared.output import save_nifti, save_dicom
