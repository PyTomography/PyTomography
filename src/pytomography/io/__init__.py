"""Reading data from scanners and simulations into PyTomography's standard form, and saving images.

Every object is on a grid in DICOM's patient frame (LPS, mm) described by its ``object_meta``; ``object_meta.affine``
says where each voxel is. The readers of each modality return ``(projections, object_meta, proj_meta)`` (PET:
``(events, object_meta, proj_meta)``):

* :mod:`pytomography.io.SPECT`: DICOM, StarGuide and SIMIND projections;
* :mod:`pytomography.io.PET`: GATE, PETSIRD and GE list mode, and sinograms;
* :mod:`pytomography.io.CT`: DICOM-CT-PD and Nikon cone-beam projections.

Images and regions of interest, any modality:

* :func:`load_image` reads any image file into an :class:`Image`, an image with its own geometry, and
  :meth:`Image.on` puts it on a reconstruction grid;
* :func:`attenuation_map` gives the attenuation coefficients (1/mm) on a grid, from a CT or a mu map;
* :func:`load_mask` reads regions of interest (RTSTRUCT, NIfTI, NRRD) as a label image;
* :func:`save_dicom` and :func:`save_nifti` write an image, its voxels placed by ``object_meta`` or by the
  :class:`Image`.
"""

from ._image import Image
from ._load import load_image
from .shared.output import save_nifti, save_dicom
from ._attenuation import attenuation_map
from ._mask import load_mask
from . import CT, PET, SPECT
