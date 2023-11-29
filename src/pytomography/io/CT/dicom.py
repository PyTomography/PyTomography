"""This module contains functionality for opening CT images."""

from __future__ import annotations
from typing import Sequence
import numpy as np
import pydicom

def open_CT_file(
    files_CT: Sequence[str],
    ) -> np.array:
    """Given a list of seperate DICOM files, opens them up and stacks them together into a single CT image. 

    Args:
        files_CT (Sequence[str]): List of CT DICOM filepaths corresponding to different z slices of the same scan.

    Returns:
        np.array: CT scan in units of Hounsfield Units at the effective CT energy.
    """
    CT = []
    slice_locs = []
    for file in files_CT:
        ds = pydicom.read_file(file)
        CT.append(ds.pixel_array)
        slice_locs.append(float(ds.ImagePositionPatient[2]))
    CT = np.transpose(np.array(CT)[np.argsort(slice_locs)[::-1]], (2,1,0)).astype(np.float32)
    CT = ds.RescaleSlope * CT + ds.RescaleIntercept
    return CT

def compute_max_slice_loc_CT(files_CT: Sequence[str]) -> float:
    """Obtains the maximum z-location from a list of CT DICOM files

    Args:
        files_CT (Sequence[str]): List of CT DICOM filepaths corresponding to different z slices of the same scan.

    Returns:
        float: Maximum z location
    """
    slice_locs = []
    for file in files_CT:
        ds = pydicom.read_file(file)
        slice_locs.append(float(ds.ImagePositionPatient[2]))
    return np.max(slice_locs)

def compute_slice_thickness_CT(files_CT: Sequence[str]) -> float:
    """Compute the slice thickness for files that make up a CT scan. Though this information is often contained in the DICOM file, it is sometimes inconsistent with the ImagePositionPatient attribute, which gives the true location of the slices.

    Args:
        files_CT (Sequence[str]): List of CT DICOM filepaths corresponding to different z slices of the same scan.

    Returns:
        float: Slice thickness of CT scan
    """
    slice_locs = []
    for file in files_CT:
        ds = pydicom.read_file(file)
        slice_locs.append(float(ds.ImagePositionPatient[2]))
    slice_locs = np.array(slice_locs)[np.argsort(slice_locs)]
    return slice_locs[1] - slice_locs[0]