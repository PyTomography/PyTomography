"""Shared functionality between different imaging modalities. 
"""

from .dicom_creation import create_ds
from .interfile import get_header_value, get_attenuation_map_interfile
from .dicom import open_multifile, align_images_affine, _get_affine_multifile