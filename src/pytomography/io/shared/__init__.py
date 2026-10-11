"""Shared functionality between different imaging modalities. 
"""

from .dicom_creation import create_ds
from .interfile import get_header_value
from .output import save_nifti, save_dicom
