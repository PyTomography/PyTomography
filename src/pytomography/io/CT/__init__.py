"""Input/output functions for the CT imaging modality. Currently, the data types supported are DICOM files.
"""

from .dicom import open_CT_file, compute_max_slice_loc_CT, compute_slice_thickness_CT
from .attenuation_map import get_HU2mu_conversion