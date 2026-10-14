"""Readers for CT projection data, corrections of the projections, and Hounsfield unit conversions.

Each reader returns PyTomography's standard form ``(projections, object_meta, proj_meta)``: the line integrals, the
object grid (in mm, units ``'1/mm'``, modality ``'CT'``) and the projection metadata.

* :func:`~pytomography.io.CT.read_dicom_ct_pd`: DICOM-CT-PD projections of a helical or axial fan beam scan from a
  third generation CT scanner, on a grid placed in the patient (LPS, k foot to head), with
  :class:`~pytomography.metadata.CT.CTGen3ProjMeta`.
* :func:`~pytomography.io.CT.read_nikon`: a Nikon XTek cone-beam micro-CT scan, on a grid in the scanner's frame, with
  :class:`~pytomography.metadata.CT.CTConeBeamFlatPanelProjMeta`.
* :func:`~pytomography.io.CT.to_hu` and :func:`~pytomography.io.CT.from_hu`: convert an image between linear
  attenuation coefficients (1/mm) and Hounsfield units.
* :mod:`~pytomography.io.CT.preprocessing`: corrections of DICOM-CT-PD projections before reconstruction (filtering
  of photon-starved rays, a per-column scale) and fits to a scanner's own images.
"""
from . import preprocessing
from .dicom_ct_pd import read_dicom_ct_pd
from .nikon import read_nikon
from .units import to_hu, from_hu
