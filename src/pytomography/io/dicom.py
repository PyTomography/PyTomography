"""Note: This module is still being built and is not yet finished. 
"""

import numpy as np
import numpy.linalg as npl
from scipy.ndimage import affine_transform
import torch
import pydicom
from pytomography.metadata import ObjectMeta, ImageMeta
from pydicom.dataset import Dataset

def get_radii_and_angles(ds: Dataset):
    """Gets projections with corresponding radii and angles corresponding to projection data from a DICOM dataset.

    Args:
        ds (Dataset): pydicom dataset object.

    Returns:
        (torch.tensor[1,Ltheta, Lr, Lz], np.array, np.array): Required image data for reconstruction.
    """
    pixel_array = ds.pixel_array.reshape((ds.NumberOfEnergyWindows, -1, ds.Rows, ds.Columns))
    detectors = np.array(ds.DetectorVector)
    radii = np.array([])
    angles = np.array([])
    for detector in np.unique(detectors):
        radial_positions_detector = ds.DetectorInformationSequence[detector-1].RadialPosition
        n_angles = len(radial_positions_detector)
        radii = np.concatenate([radii, radial_positions_detector])
        delta_angle = ds.RotationInformationSequence[0].AngularStep
        angles = np.concatenate([angles, ds.DetectorInformationSequence[detector-1].StartAngle + delta_angle*np.arange(n_angles)])
    angles = (angles + 180)%360 # to detector angle convention
    sorted_idxs = np.argsort(angles)
    projections = np.transpose(pixel_array[:,sorted_idxs][:,:,::-1], (0,1,3,2)).astype(np.float32)
    projections= torch.tensor(projections.copy())
    return (projections,
             angles[sorted_idxs],
             radii[sorted_idxs]/10)

def dicom_projections_to_data(file):
    """Obtains ObjectMeta, ImageMeta, and projections from a .dcm file.

    Args:
        file (str): Path to the .dcm file

    Returns:
        (ObjectMeta, ImageMeta, torch.Tensor[1, Ltheta, Lr, Lz]): Required information for reconstruction in PyTomography.
    """
    ds = pydicom.read_file(file)
    dx = ds.PixelSpacing[0] / 10
    dz = ds.PixelSpacing[1] / 10
    dr = (dx, dx, dz)
    projections, angles, radii = get_radii_and_angles(ds)
    shape_proj= projections[0].shape
    shape_obj = (shape_proj[1], shape_proj[1], shape_proj[2])
    object_meta = ObjectMeta(dr,shape_obj)
    image_meta = ImageMeta(object_meta, angles, radii)
    return object_meta, image_meta, projections


a1 = 0.00014376
b1 = 0.1352
a2 = 0.00008787
b2 = 0.1352
def HU_to_mu(HU):
    mu = np.piecewise(HU, [HU <= 0, HU > 0],
                 [lambda x: a1*x + b1,
                  lambda x: a2*x + b2])
    mu[mu<0] = 0
    return mu

def get_affine_spect(ds):
    Sx, Sy, Sz = ds.DetectorInformationSequence[0].ImagePositionPatient
    dx = dy = ds.PixelSpacing[0]
    dz = ds.PixelSpacing[1]
    Sx -= ds.Rows / 2 * (-dx)
    Sy -= ds.Rows / 2 * (-dy)
    M = np.zeros((4,4))
    M[:,0] = np.array([-dx, 0, 0, 0])
    M[:,1] = np.array([0, -dy, 0, 0])
    M[:,2] = np.array([0, 0, -dz, 0])
    M[:,3] = np.array([Sx, Sy, Sz, 1])
    return M

def get_affine_CT(ds, max_z):
    M_CT = np.zeros((4,4))
    M_CT[0:3, 0] = np.array(ds.ImageOrientationPatient[0:3])*ds.PixelSpacing[0]
    M_CT[0:3, 1] = np.array(ds.ImageOrientationPatient[3:])*ds.PixelSpacing[1]
    M_CT[0:3, 2] = -np.array([0,0,1]) * ds.SliceThickness 
    M_CT[:-2,3] = ds.ImagePositionPatient[0] 
    M_CT[2, 3] = max_z
    M_CT[3, 3] = 1
    return M_CT

def dicom_CT_to_data(files_CT, file_NM=None):
    ds_NM = pydicom.read_file(file_NM)
    CT_scan = []
    slice_locs = []
    for file in files_CT:
        ds = pydicom.read_file(file)
        CT_scan.append(ds.pixel_array)
        slice_locs.append(float(ds.SliceLocation))
    CT_scan = np.transpose(np.array(CT_scan)[np.argsort(slice_locs)], (2,1,0)).astype(np.float32)
   # Affine matrix
    M_CT = get_affine_CT(ds, np.max(np.abs(slice_locs)))
    M_NM = get_affine_spect(pydicom.read_file(file_NM))
    # Resample CT and convert to mu at 208keV and save
    M = npl.inv(M_CT) @ M_NM
    CT_resampled = affine_transform(CT_scan, M[0:3,0:3], M[:3,3], output_shape=(ds_NM.Rows, ds_NM.Rows, ds_NM.Columns) )
    CT_HU = CT_resampled + ds.RescaleIntercept
    CT = HU_to_mu(CT_HU)
    CT = torch.tensor(CT[::-1,::-1,::-1].copy())
    return CT