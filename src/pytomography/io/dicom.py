"""Note: This module is still being built and is not yet finished. 
"""
from __future__ import annotations
import warnings
from typing import Sequence
from pathlib import Path
import numpy as np
import numpy.linalg as npl
from scipy.ndimage import affine_transform
import torch
import torch.nn as nn
import pydicom
from pytomography.metadata import ObjectMeta, ImageMeta
from pydicom.dataset import Dataset
from pytomography.metadata import ObjectMeta, ImageMeta
from pytomography.projections import ForwardProjectionNet, BackProjectionNet
from pytomography.mappings import SPECTAttenuationNet, SPECTPSFNet
from pytomography.priors import Prior
from pytomography.metadata import PSFMeta
from pytomography.algorithms import OSEMOSL

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

def dicom_MEW_to_data(file, type='DEW'):
    ds = pydicom.read_file(file)
    if type=='DEW':
        primary_window_width = ds.EnergyWindowInformationSequence[0].EnergyWindowRangeSequence[0].EnergyWindowUpperLimit - ds.EnergyWindowInformationSequence[0].EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
        scatter_window_width = ds.EnergyWindowInformationSequence[1].EnergyWindowRangeSequence[0].EnergyWindowUpperLimit - ds.EnergyWindowInformationSequence[1].EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
        object_meta, image_meta, projections = dicom_projections_to_data(file)
        projections_primary = projections[0].unsqueeze(dim=0)
        projections_scatter = projections[1].unsqueeze(dim=0) * primary_window_width / scatter_window_width
        return object_meta, image_meta, projections_primary, projections_scatter


def get_HU2mu_coefficients(ds):
    table = np.loadtxt('../../../data/HU_to_mu.csv', skiprows=1)
    energies = table.T[0]
    window_upper = ds.EnergyWindowInformationSequence[0].EnergyWindowRangeSequence[0].EnergyWindowUpperLimit
    window_lower = ds.EnergyWindowInformationSequence[0].EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
    energy = (window_lower + window_upper)/2
    index = np.argmin(np.abs(energies-energy))
    print(f'Based on primary window with range ({window_lower:.2f}, {window_upper:.2f})keV, using conversion between hounsfield to linear attenuation coefficient based on radionuclide with emission energy {table[index,0]}keV')
    return table[index,1:]
    
    
# conversion from https://www.sciencedirect.com/science/article/pii/S0969804308000067
def HU_to_mu(HU, a1, b1, a2, b2):
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
    CT = HU_to_mu(CT_HU, *get_HU2mu_coefficients(ds_NM))
    CT = torch.tensor(CT[::-1,::-1,::-1].copy())
    return CT

def get_SPECT_recon_algorithm_dicom(
    projections_file: str,
    atteunation_files: Sequence[str] = None,
    use_psf: bool = False,
    scatter_type: str|None = None,
    prior: Prior = None,
    recon_algorithm_class: nn.Module = OSEMOSL,
    object_initial: torch.Tensor | None = None,
    device: str = 'cpu'
) -> nn.Module:
    # Get projections/scatter estimate
    if scatter_type==None:
        object_meta, image_meta, projections = dicom_projections_to_data(projections_file)
        projections_scatter = 0 # equivalent to 0 estimated scatter everywhere
    else:
        object_meta, image_meta, projections, projections_scatter = dicom_MEW_to_data(projections_file, type=scatter_type)
    # obj2obj and im2im nets.
    object_correction_nets = []
    image_correction_nets = []
    # Load attenuation data
    if atteunation_files is not None:
        CT = dicom_CT_to_data(atteunation_files, projections_file)
        CT_net = SPECTAttenuationNet(CT.unsqueeze(dim=0).to(device), device=device)
        object_correction_nets.append(CT_net)
    # Load PSF parameters
    if use_psf:
        ds = pydicom.read_file(projections_file)
        if ds.Manufacturer =='SIEMENS NM':
            # Find a more consistent way to do this
            angular_FWHM = ds[0x0055, 0x107f][0]
            psf_meta = PSFMeta(collimator_slope = angular_FWHM/(2*np.sqrt(2*np.log(2))), collimator_intercept = 0.0)
            psf_net = SPECTPSFNet(psf_meta, device)
        else:
            raise Exception('Unable to compute PSF metadata from this DICOM file')
        object_correction_nets.append(psf_net)
    fp_net = ForwardProjectionNet(object_correction_nets, image_correction_nets, object_meta, image_meta, device=device)
    bp_net = BackProjectionNet(object_correction_nets, image_correction_nets, object_meta, image_meta, device=device)
    if prior is not None:
        prior.set_device(device)
    recon_algorithm = recon_algorithm_class(projections, fp_net, bp_net, object_initial, projections_scatter, prior)
    return recon_algorithm