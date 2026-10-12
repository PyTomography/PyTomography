from __future__ import annotations
from collections.abc import Sequence
from typing import Sequence
import warnings
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
    # Adjust by table height for Mediso
    try:
        if ds.Manufacturer=='Mediso':
            Sy = Sy - ds.TableHeight
    except:
        pass
    M = np.zeros((4, 4))
    M[0] = np.array([dx*Xxyz[0], dy*Yxyz[0], dz*Zxyz[0], Sx])
    M[1] = np.array([dx*Xxyz[1], dy*Yxyz[1], dz*Zxyz[1], Sy])
    M[2] = np.array([dx*Xxyz[2], dy*Yxyz[2], dz*Zxyz[2], Sz])
    M[3] = np.array([0, 0, 0, 1])
    return M

def _shared_group(ds, sequence: str):
    """The item of ``sequence`` in the Shared Functional Groups of an enhanced multi-frame file, or None."""
    shared = ds.get('SharedFunctionalGroupsSequence')
    if shared and sequence in shared[0]:
        return getattr(shared[0], sequence)[0]
    return None


def _frame_origin_and_spacing(ds, filename: str = '') -> tuple:
    """Position (mm) of the first frame of an image stored as the frames of one file, and the signed distance between
    consecutive frames along the frames' normal.

    The distance comes from the frames' positions (PlanePositionSequence of the Per-frame Functional Groups) when the
    file has them; otherwise from SpacingBetweenSlices (0018,0088), with the frames taken to run along +z; and only
    when neither is there, with a warning, from SliceThickness (0018,0050). SliceThickness is the thickness of a slice,
    which is not the distance between slices when they overlap or have gaps (TCIA LDCT-and-Projection-data C145: 1.25 mm
    thick, 1 mm apart)."""
    frames = ds.get('PerFrameFunctionalGroupsSequence') or []
    positions = [np.asarray(f.PlanePositionSequence[0].ImagePositionPatient, dtype=np.float64)
                 for f in frames if 'PlanePositionSequence' in f]
    origin = positions[0] if positions else np.asarray(ds.ImagePositionPatient, dtype=np.float64)
    if len(positions) > 1:
        orientation = ds.get('ImageOrientationPatient')
        plane = _shared_group(ds, 'PlaneOrientationSequence')
        if orientation is None and plane is not None:
            orientation = plane.ImageOrientationPatient
        if orientation is None:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = np.cross(np.asarray(orientation[:3], dtype=np.float64), np.asarray(orientation[3:], dtype=np.float64))
        return origin, float(np.dot(positions[1] - positions[0], normal / np.linalg.norm(normal)))
    if 'SpacingBetweenSlices' in ds:
        return origin, abs(float(ds.SpacingBetweenSlices))
    warnings.warn(f'{filename}: no frame positions and no SpacingBetweenSlices, so the distance between slices is taken '
                  f'to be SliceThickness ({float(ds.SliceThickness)} mm), which is wrong if the slices overlap or have gaps')
    return origin, float(ds.SliceThickness)


def _get_affine_single_file(filename: str) -> np.array:
    """Obtain the affine matrix (axis aligned, mm) of a 3D image stored as the frames of a single DICOM file: voxel
    (i, j, k) is at the position of the first frame plus (i dx, j dy, k dz), with ``dz`` the distance between frames
    from their positions, or from SpacingBetweenSlices (see :func:`_frame_origin_and_spacing`), never the slice
    thickness unless nothing else is there.

    Args:
        filename (str): Path of file

    Returns:
        np.array: Affine matrix
    """
    ds = pydicom.dcmread(filename, stop_before_pixels=True)
    (Sx, Sy, Sz), dz = _frame_origin_and_spacing(ds, str(filename))
    measures = _shared_group(ds, 'PixelMeasuresSequence')
    dx, dy = (float(v) for v in (ds.PixelSpacing if 'PixelSpacing' in ds else measures.PixelSpacing))
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
        object_meta.affine_matrix = _get_affine_multifile(files)
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
    """Distance between consecutive slices (mm) of a scan stored one slice per file, from their ImagePositionPatient.
    Despite the name, this is the slice spacing, not SliceThickness (0018,0050): the two differ when slices overlap
    or have gaps (TCIA LDCT-and-Projection-data C145: 1.25 mm thick, 1 mm apart), and the spacing is what places the
    slices.

    Args:
        files (Sequence[str]): List of DICOM filepaths corresponding to different z slices of the same scan.

    Returns:
        float: Distance between consecutive slices
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
    
