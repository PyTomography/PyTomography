"""Reading CT data into PyTomography's standard form: DICOM-CT-PD fan beam projections (:func:`read_dicom_ct_pd`) and
Nikon XTek cone-beam micro-CT (:func:`read_nikon`). Each returns ``(projections, object_meta, proj_meta)``, the
object grid placed in the patient (or the scanner). :func:`to_hu` and :func:`from_hu` convert a reconstruction
between 1/mm and Hounsfield units; :mod:`preprocessing` holds the corrections of the projections."""
from . import preprocessing
from .dicom_ct_pd import read_dicom_ct_pd
from .nikon import read_nikon
from .units import to_hu, from_hu
