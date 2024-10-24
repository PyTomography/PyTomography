from __future__ import annotations
from collections.abc import Sequence
from typing import Sequence
import numpy as np
import torch
import numpy.linalg as npl
from scipy.ndimage import affine_transform
import pytomography
from pytomography.metadata import ObjectMeta
import pydicom

def _get_affine_multifile(files: Sequence[str]):
    """Computes an affine matrix corresponding the coordinate system of a CT DICOM file. Note that since CT scans consist of many independent DICOM files, ds corresponds to an individual one of these files. This is why the maximum z value is also required (across all seperate independent DICOM files).

    Args:
        ds (Dataset): DICOM dataset of CT data
        max_z (float): Maximum value of z across all axial slices that make up the CT scan

    Returns:
        np.array: Affine matrix corresponding to CT scan.
    """
    # Note: per DICOM convention z actually decreases as the z-index increases (initial z slices start with the head)
    ds = pydicom.dcmread(files[0])
    dx, dy = ds.PixelSpacing
    dz = compute_slice_thickness_multifile(files)
    orientation = ds.ImageOrientationPatient
    Xxyz = orientation[:3]
    Yxyz = orientation[3:]
    Zxyz = np.cross(Xxyz, Yxyz)
    Sx, Sy = ds.ImagePositionPatient[:2]
    Sz = compute_min_slice_loc_multifile(files)
    M = np.zeros((4, 4))
    M[0] = np.array([dx*Xxyz[0], dy*Yxyz[0], dz*Zxyz[0], Sx])
    M[1] = np.array([dx*Xxyz[1], dy*Yxyz[1], dz*Zxyz[1], Sy])
    M[2] = np.array([dx*Xxyz[2], dy*Yxyz[2], dz*Zxyz[2], Sz])
    M[3] = np.array([0, 0, 0, 1])
    return M

def _get_affine_single_file(filename: str) -> np.array:
    """Obtain the affine matrix from a 3D medical image stored in a single file.

    Args:
        filename (str): Path of file

    Returns:
        np.array: Affine matrix
    """
    ds = pydicom.dcmread(filename)
    Sx, Sy, Sz = ds.ImagePositionPatient
    dx, dy = ds.PixelSpacing
    dz = ds.SliceThickness # assumption
    M = np.zeros((4, 4))
    M[0] = np.array([dx, 0, 0, Sx])
    M[1] = np.array([0, dy, 0, Sy])
    M[2] = np.array([0, 0, dz, Sz])
    M[3] = np.array([0, 0, 0, 1])
    return M



def open_multifile(
    files: Sequence[str],
    return_object_meta: bool = False
    ) -> torch.Tensor:
    """Given a list of seperate DICOM files, opens them up and stacks them together into a single CT image. 

    Args:
        files (Sequence[str]): List of CT DICOM filepaths corresponding to different z slices of the same scan.
        return_object_meta (bool): Whether or not to return object metadata corresponding to opened file

    Returns:
        np.array: CT scan in units of Hounsfield Units at the effective CT energy.
    """
    array = []
    slice_locs = []
    for file in files:
        ds = pydicom.dcmread(file)
        array.append(ds.RescaleSlope*ds.pixel_array+ ds.RescaleIntercept)
        slice_locs.append(float(ds.ImagePositionPatient[2]))
    array = np.transpose(np.array(array)[np.argsort(slice_locs)], (2,1,0)).astype(np.float32)
    array = torch.tensor(array).to(pytomography.device)
    if not return_object_meta:
        return array
    else:
        dx, dy = ds.PixelSpacing
        dz = compute_slice_thickness_multifile(files)
        shape = array.shape
        object_meta = ObjectMeta(dr=(dx,dy,dz), shape=shape)
        object_meta.affine_matrix = _get_affine_multifile(file)
        return array, object_meta
    
def open_singlefile(file: str) -> torch.Tensor:
    """Opens data from a single DICOM file.

    Args:
        file (str): Filepath

    Returns:
        torch.Tensor: 3D Image
    """
    ds = pydicom.dcmread(file)
    if 'RescaleIntercept' in ds.dir():
        intercept = ds.RescaleIntercept
    else:
        intercept = 0
    array = ds.RescaleSlope*ds.pixel_array+ intercept
    array = np.transpose(array, (2,1,0)).astype(np.float32)
    array = torch.tensor(array).to(pytomography.device)
    return array

def compute_max_slice_loc_multifile(files: Sequence[str]) -> float:
    """Obtains the maximum z-location from a list of DICOM slice files

    Args:
        files (Sequence[str]): List of DICOM filepaths corresponding to different z slices of the same scan.

    Returns:
        float: Maximum z location
    """
    slice_locs = []
    for file in files:
        ds = pydicom.dcmread(file)
        slice_locs.append(float(ds.ImagePositionPatient[2]))
    return np.max(slice_locs)

def compute_min_slice_loc_multifile(files: Sequence[str]) -> float:
    """Obtains the minimum z-location from a list of DICOM slice files

    Args:
        files (Sequence[str]): List of DICOM filepaths corresponding to different z slices of the same scan.

    Returns:
        float: Minimum location
    """
    slice_locs = []
    for file in files:
        ds = pydicom.dcmread(file)
        slice_locs.append(float(ds.ImagePositionPatient[2]))
    return np.min(slice_locs)

def compute_slice_thickness_multifile(files: Sequence[str]) -> float:
    """Compute the slice thickness for files that make up a scan. Though this information is often contained in the DICOM file, it is sometimes inconsistent with the ImagePositionPatient attribute, which gives the true location of the slices.

    Args:
        files (Sequence[str]): List of DICOM filepaths corresponding to different z slices of the same scan.

    Returns:
        float: Slice thickness of the scan
    """
    slice_locs = []
    for file in files:
        ds = pydicom.dcmread(file)
        slice_locs.append(float(ds.ImagePositionPatient[2]))
    slice_locs = np.array(slice_locs)[np.argsort(slice_locs)]
    return slice_locs[1] - slice_locs[0]

def align_images_affine(im_fixed, im_moving, affine_fixed, affine_moving, cval = 0):
    # Note: must flip along Z to match affine 
    affine = npl.inv(affine_moving) @ affine_fixed
    im_moving_adjusted = affine_transform(
        im_moving,
        affine,
        output_shape=im_fixed.shape,
        mode='constant',
        order=1,
        cval=cval
    )
    return im_moving_adjusted
    