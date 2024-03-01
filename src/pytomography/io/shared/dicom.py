from __future__ import annotations
import warnings
import os
import collections.abc
from collections.abc import Sequence
from pathlib import Path
from typing import Sequence
import numpy as np
import numpy.linalg as npl
from scipy.ndimage import affine_transform
import torch
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid
import pytomography
from pytomography.metadata import SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta
from pytomography.utils import (
    get_blank_below_above,
    compute_TEW,
    get_mu_from_spectrum_interp,
)

def _get_affine_multifile(files: Sequence[str]):
    """Computes an affine matrix corresponding the coordinate system of a CT DICOM file. Note that since CT scans consist of many independent DICOM files, ds corresponds to an individual one of these files. This is why the maximum z value is also required (across all seperate independent DICOM files).

    Args:
        ds (Dataset): DICOM dataset of CT data
        max_z (float): Maximum value of z across all axial slices that make up the CT scan

    Returns:
        np.array: Affine matrix corresponding to CT scan.
    """
    # Note: per DICOM convention z actually decreases as the z-index increases (initial z slices start with the head)
    ds = pydicom.read_file(files[0])
    dz = compute_slice_thickness_multifile(files)
    max_z = compute_max_slice_loc_multifile(files)
    M = np.zeros((4, 4))
    M[0:3, 0] = np.array(ds.ImageOrientationPatient[0:3]) * ds.PixelSpacing[0]
    M[0:3, 1] = np.array(ds.ImageOrientationPatient[3:]) * ds.PixelSpacing[1]
    M[0:3, 2] = -np.array([0, 0, 1]) * dz
    M[0:2, 3] = np.array(ds.ImagePositionPatient)[0:2]
    M[2, 3] = max_z
    M[3, 3] = 1
    return M

def open_multifile(
    files: Sequence[str],
    ) -> np.array:
    """Given a list of seperate DICOM files, opens them up and stacks them together into a single CT image. 

    Args:
        files (Sequence[str]): List of CT DICOM filepaths corresponding to different z slices of the same scan.

    Returns:
        np.array: CT scan in units of Hounsfield Units at the effective CT energy.
    """
    array = []
    slice_locs = []
    for file in files:
        ds = pydicom.read_file(file)
        array.append(ds.RescaleSlope*ds.pixel_array+ ds.RescaleIntercept)
        slice_locs.append(float(ds.ImagePositionPatient[2]))
    array = np.transpose(np.array(array)[np.argsort(slice_locs)[::-1]], (2,1,0)).astype(np.float32)
    return array[:,:,::-1].copy()

def compute_max_slice_loc_multifile(files: Sequence[str]) -> float:
    """Obtains the maximum z-location from a list of DICOM slice files

    Args:
        files (Sequence[str]): List of DICOM filepaths corresponding to different z slices of the same scan.

    Returns:
        float: Maximum z location
    """
    slice_locs = []
    for file in files:
        ds = pydicom.read_file(file)
        slice_locs.append(float(ds.ImagePositionPatient[2]))
    return np.max(slice_locs)

def compute_slice_thickness_multifile(files: Sequence[str]) -> float:
    """Compute the slice thickness for files that make up a scan. Though this information is often contained in the DICOM file, it is sometimes inconsistent with the ImagePositionPatient attribute, which gives the true location of the slices.

    Args:
        files (Sequence[str]): List of DICOM filepaths corresponding to different z slices of the same scan.

    Returns:
        float: Slice thickness of the scan
    """
    slice_locs = []
    for file in files:
        ds = pydicom.read_file(file)
        slice_locs.append(float(ds.ImagePositionPatient[2]))
    slice_locs = np.array(slice_locs)[np.argsort(slice_locs)]
    return slice_locs[1] - slice_locs[0]

def align_images_affine(im_fixed, im_moving, affine_fixed, affine_moving, cval = 0):
    # Note: must flip along Z to match affine 
    affine = npl.inv(affine_moving) @ affine_fixed
    im_moving_adjusted = affine_transform(
        im_moving[:,:,::-1],
        affine,
        output_shape=im_fixed.shape,
        mode='constant',
        order=1,
        cval=cval
    )[:,:,::-1]
    return im_moving_adjusted
    