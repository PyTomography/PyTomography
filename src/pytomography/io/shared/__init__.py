"""Shared functionality between different imaging modalities. 
"""

from .dicom_creation import create_ds
from .interfile import get_header_value, get_attenuation_map_interfile