"""Note: This module is still being built and is not yet finished. 
"""
from __future__ import annotations
import warnings
import os
import collections.abc
from typing import Sequence
from pathlib import Path
import numpy as np
import numpy.linalg as npl
from scipy.ndimage import affine_transform
import scipy.interpolate
import torch
import torch.nn as nn
import pydicom
from pydicom.dataset import Dataset
import pytomography
from pytomography.metadata import ObjectMeta, ImageMeta
from pytomography.metadata import ObjectMeta, ImageMeta
from pytomography.projections import SystemMatrix
from pytomography.metadata import PSFMeta
from pytomography.utils import get_blank_below_above, bilinear_transform

def get_radii_and_angles(ds: Dataset) -> Sequence[torch.Tensor, np.array, np.array]:
    """Gets projections with corresponding radii and angles corresponding to projection data from a DICOM file.

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
        n_angles = ds.RotationInformationSequence[0].NumberOfFramesInRotation
        delta_angle = ds.RotationInformationSequence[0].AngularStep
        try:
            start_angle = ds.DetectorInformationSequence[detector-1].StartAngle
        except:
            start_angle = ds.RotationInformationSequence[0].StartAngle
        rotation_direction = ds.RotationInformationSequence[0].RotationDirection
        if rotation_direction=='CC' or rotation_direction=='CCW':
            angles = np.concatenate([angles, start_angle + delta_angle*np.arange(n_angles)])
        else:
            angles = np.concatenate([angles, start_angle - delta_angle*np.arange(n_angles)])
        radial_positions_detector = ds.DetectorInformationSequence[detector-1].RadialPosition
        if not isinstance(radial_positions_detector, collections.abc.Sequence):
            radial_positions_detector = n_angles * [radial_positions_detector]
        radii = np.concatenate([radii, radial_positions_detector])
        
    angles = (angles + 180)%360 # to detector angle convention
    sorted_idxs = np.argsort(angles)
    projections = np.transpose(pixel_array[:,sorted_idxs][:,:,::-1], (0,1,3,2)).astype(np.float32)
    projections= torch.tensor(projections.copy())
    return (projections,
             angles[sorted_idxs],
             radii[sorted_idxs]/10)

def get_projections(
    file: str,
    index_peak: None | int = None
    ) -> Sequence[ObjectMeta, ImageMeta, torch.Tensor]:
    """Gets ObjectMeta, ImageMeta, and projections from a .dcm file.

    Args:
        file (str): Path to the .dcm file
        index_peak (int): If not none, then the returned projections correspond to the index of this energy window. Otherwise returns all energy windows. Defaults to None.
    Returns:
        (ObjectMeta, ImageMeta, torch.Tensor[1, Ltheta, Lr, Lz]): Required information for reconstruction in PyTomography.
    """
    ds = pydicom.read_file(file, force=True)
    dx = ds.PixelSpacing[0] / 10
    dz = ds.PixelSpacing[1] / 10
    dr = (dx, dx, dz)
    projections, angles, radii = get_radii_and_angles(ds)
    shape_proj= projections[0].shape
    shape_obj = (shape_proj[1], shape_proj[1], shape_proj[2])
    object_meta = ObjectMeta(dr,shape_obj)
    image_meta = ImageMeta(object_meta, angles, radii)
    if index_peak is not None:
        projections = projections[index_peak].unsqueeze(dim=0)
    return object_meta, image_meta, projections

# used in the function below
def get_window_width(ds: Dataset, index: int) -> float:
    """Computes the width of an energy window corresponding to a particular index in the DetectorInformationSequence DICOM attribute.

    Args:
        ds (Dataset): DICOM dataset.
        index (int): Energy window index corresponding to the DICOM dataset.

    Returns:
        float: Range of the energy window in keV
    """
    energy_window = ds.EnergyWindowInformationSequence[index]
    window_range1 = energy_window.EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
    window_range2 = energy_window.EnergyWindowRangeSequence[0].EnergyWindowUpperLimit
    return window_range2 - window_range1

def get_scatter_from_TEW(
    file: str,
    index_peak: int,
    index_lower: int,
    index_upper: int
    ) -> torch.Tensor:
    """Gets an estimate of scatter projection data from a DICOM file using the triple energy window method.

    Args:
        file (str): Filepath of the DICOM file
        index_peak (int): Index of the ``EnergyWindowInformationSequence`` DICOM attribute corresponding to the photopeak.
        index_lower (int): Index of the ``EnergyWindowInformationSequence`` DICOM attribute corresponding to lower scatter window.
        index_upper (int): Index of the ``EnergyWindowInformationSequence`` DICOM attribute corresponding to upper scatter window.

    Returns:
        torch.Tensor[1,Ltheta,Lr,Lz]: Tensor corresponding to the scatter estimate.
    """
    ds = pydicom.read_file(file, force=True)
    ww_peak = get_window_width(ds, index_peak)
    ww_lower = get_window_width(ds, index_lower)
    ww_upper = get_window_width(ds, index_upper)
    _, _, projections_all = get_projections(file)
    scatter = (projections_all[index_lower]/ww_lower + projections_all[index_upper]/ww_upper)* ww_peak / 2
    return scatter.unsqueeze(dim=0)

def get_attenuation_map_from_file(file_AM: str) -> torch.Tensor:
    """Gets an attenuation map from a DICOM file. This data is usually provided by the manufacturer of the SPECT scanner. 

    Args:
        file_AM (str): File name of attenuation map

    Returns:
        torch.Tensor: Tensor of shape [batch_size, Lx, Ly, Lz] corresponding to the atteunation map in units of cm:math:`^{-1}`
    """
    ds = pydicom.read_file(file_AM, force=True)
    attenuation_map =  ds.pixel_array / ds[0x033,0x1038].value
    return torch.tensor(np.transpose(attenuation_map, (2,1,0))).unsqueeze(dim=0)

def get_psfmeta_from_scanner_params(
    camera_model: str,
    collimator_name: str,
    energy_keV: float
    ) -> PSFMeta:
    """Gets PSF metadata from SPECT camera/collimator parameters. Performs linear interpolation to find linear attenuation coefficient for lead collimators for energy values within the range 100keV - 600keV.

    Args:
        camera_model (str): Name of SPECT camera. 
        collimator_name (str): Name of collimator used.
        energy_keV (float): Energy of the photopeak

    Returns:
        PSFMeta: PSF metadata.
    """

    module_path = os.path.dirname(os.path.abspath(__file__))
    
    scanner_datasheet = np.genfromtxt(os.path.join(module_path, '../../data/SPECT_collimator_parameters.csv'), skip_header=1, dtype=['U50,U50,float,float'], delimiter=',', unpack=True)
    attenuation_coefficient_energy = np.genfromtxt(os.path.join(module_path, '../../data/lead_attenuation_values.csv'), skip_header = 1, dtype=['float,float'], delimiter=',', unpack = True)
        
    energies_keV = (list(zip(*attenuation_coefficient_energy))[0])
    linear_attenuation_coef_vals = (list(zip(*attenuation_coefficient_energy))[1])

    linear_attenuation_coef_val_interp = scipy.interpolate.interp1d(energies_keV, linear_attenuation_coef_vals)

    for i in range(len(scanner_datasheet)):
        if camera_model == scanner_datasheet[i][0] and collimator_name == scanner_datasheet[i][1]:
            hole_diameter = scanner_datasheet[i][2]
            hole_length = scanner_datasheet[i][3]

    attenuation_coefficient = linear_attenuation_coef_val_interp(energy_keV)

    collimator_slope_FWHM = hole_diameter/(hole_length - (2/attenuation_coefficient))
    collimator_intercept_FWHM = hole_diameter

    # convert to units of sigma
    collimator_slope = collimator_slope_FWHM * (1/(2*np.sqrt(2*np.log(2))))
    collimator_intercept = collimator_intercept_FWHM * (1/(2*np.sqrt(2*np.log(2))))

    return PSFMeta(collimator_slope, collimator_intercept)

def get_attenuation_map_from_CT_slices(
    files_CT: Sequence[str],
    file_NM: str,
    index_peak: int = 0
    ) -> torch.Tensor:
    """Converts a sequence of DICOM CT files (corresponding to a single scan) into a torch.Tensor object usable as an attenuation map in PyTomography. Note that it is recommended by https://jnm.snmjournals.org/content/57/1/151.long to use the vendors attenuation map as opposed to creating your own. As such, the ``get_attenuation_map_from_file`` should be used preferentially over this function, if you have access to an attenuation map from the vendor.

    Args:
        files_CT (Sequence[str]): List of all files corresponding to an individual CT scan
        file_NM (str): File corresponding to raw PET/SPECT data (required to align CT with projections)
        index_peak (int, optional): Index corresponding to photopeak in projection data. Defaults to 0.

    Returns:
        torch.Tensor: Tensor of shape [Lx, Ly, Lz] corresponding to attenuation map.
    """
    ds_NM = pydicom.read_file(file_NM, force=True)
    CT_scan = []
    slice_locs = []
    for file in files_CT:
        ds = pydicom.read_file(file, force=True)
        CT_scan.append(ds.pixel_array)
        slice_locs.append(float(ds.SliceLocation))
    CT_scan = np.transpose(np.array(CT_scan)[np.argsort(slice_locs)], (2,1,0)).astype(np.float32)
   # Affine matrix
    M_CT = get_affine_CT(ds, np.max(np.abs(slice_locs)))
    M_NM = get_affine_spect(pydicom.read_file(file_NM, force=True))
    # Resample CT and convert to mu at 208keV and save
    M = npl.inv(M_CT) @ M_NM
    CT_resampled = affine_transform(CT_scan, M[0:3,0:3], M[:3,3], output_shape=(ds_NM.Rows, ds_NM.Rows, ds_NM.Columns) )
    CT_HU = CT_resampled + ds.RescaleIntercept
    CT = CT_to_attenuation_map(CT_HU, ds_NM, index_peak)
    CT = torch.tensor(CT[::-1,::-1,::-1].copy()).unsqueeze(dim=0)
    return CT

def CT_to_attenuation_map(
    CT_HU: np.array,
    ds: Dataset,
    photopeak_window_index: int = 0
    ) -> np.array:
    """Obtains an attenuation map from a CT file. Requires dataset corresponding to projection data because energy windows are used to convert from HU to linear attenuation coefficients. See https://www.sciencedirect.com/science/article/pii/S0969804308000067 for more details.

    Args:
        CT_HU (np.array): CT object in units of hounsfield units.
        ds (Dataset): DICOM data set of projection data
        primary_window_index (int, optional): The energy window corresponding to the photopeak. Defaults to 0.

    Returns:
        np.array: Array of length 4 containins the 4 coefficients required for the bilinear transformation.
    """
    module_path = os.path.dirname(os.path.abspath(__file__))
    table = np.loadtxt(os.path.join(module_path, '../../data/HU_to_mu.csv'), skiprows=1)
    energies = np.sort(table.T[0])
    window_upper = ds.EnergyWindowInformationSequence[photopeak_window_index].EnergyWindowRangeSequence[0].EnergyWindowUpperLimit
    window_lower = ds.EnergyWindowInformationSequence[photopeak_window_index].EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
    energy = (window_lower + window_upper)/2
    index_upper = np.searchsorted(energies, energy, side='left')
    index_lower = index_upper -1
    CT_mu_lower = bilinear_transform(CT_HU, *table[index_lower,1:])
    CT_mu_upper = bilinear_transform(CT_HU, *table[index_upper,1:])
    CT_mu_avg = CT_mu_lower + (CT_mu_upper - CT_mu_lower)/(energies[index_upper] - energies[index_lower]) * (energy-energies[index_lower])
    return CT_mu_avg

def get_affine_spect(ds: Dataset) -> np.array:
    """Computes an affine matrix corresponding the coordinate system of a SPECT DICOM file.

    Args:
        ds (Dataset): DICOM dataset of projection data

    Returns:
        np.array: Affine matrix.
    """
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

def get_affine_CT(ds: Dataset, max_z: float):
    """Computes an affine matrix corresponding the coordinate system of a CT DICOM file. Note that since CT scans consist of many independent DICOM files, ds corresponds to an individual one of these files. This is why the maximum z value is also required (across all seperate independent DICOM files).

    Args:
        ds (Dataset): DICOM dataset of CT data
        max_z (float): Maximum value of z across all axial slices that make up the CT scan

    Returns:
        np.array: Affine matrix corresponding to CT scan.
    """
    M_CT = np.zeros((4,4))
    M_CT[0:3, 0] = np.array(ds.ImageOrientationPatient[0:3])*ds.PixelSpacing[0]
    M_CT[0:3, 1] = np.array(ds.ImageOrientationPatient[3:])*ds.PixelSpacing[1]
    M_CT[0:3, 2] = -np.array([0,0,1]) * ds.SliceThickness 
    M_CT[:-2,3] = ds.ImagePositionPatient[0] 
    M_CT[2, 3] = max_z
    M_CT[3, 3] = 1
    return M_CT

def stitch_multibed(
    recons: torch.Tensor,
    files_NM: Sequence[str],
    method: str ='midslice'
    ) -> torch.Tensor:
    """Stitches together multiple reconstructed objects corresponding to different bed positions.

    Args:
        recons (torch.Tensor[n_beds, Lx, Ly, Lz]): Reconstructed objects. The first index of the tensor corresponds to different bed positions
        files_NM (list): List of length ``n_beds`` corresponding to the DICOM file of each reconstruction
        method (str, optional): Method to perform stitching (see https://doi.org/10.1117/12.2254096 for all methods described). Available methods include ``'midslice'``, ``'average'``, ``'crossfade'``, and ``'TEM;`` (transition error minimization).

    Returns:
        torch.Tensor[1, Lx, Ly, Lz']: Stitched together DICOM file. Note the new z-dimension size :math:`L_z'`.
    """
    dss = np.array([pydicom.read_file(file_NM) for file_NM in files_NM])
    zs = np.array([ds.DetectorInformationSequence[0].ImagePositionPatient[-1] for ds in dss])
    # Sort by increasing z-position
    order = np.argsort(zs)
    dss = dss[order]
    zs = zs[order]
    recons = recons[order]
    #convert to voxel height
    zs = np.round((zs - zs[0])/dss[0].PixelSpacing[1]).astype(int) 
    new_z_height = zs[-1] + recons.shape[-1]
    recon_aligned = torch.zeros((1, dss[0].Rows, dss[0].Rows, new_z_height)).to(pytomography.device)
    blank_below, blank_above = get_blank_below_above(files_NM[0])
    for i in range(len(zs)):
        recon_aligned[:,:,:,zs[i]+blank_below:zs[i]+blank_above] = recons[i,:,:,blank_below:blank_above]
    # Apply stitching method
    for i in range(1,len(zs)):
        zmin = zs[i] + blank_below
        zmax = zs[i-1] + blank_above
        dL = zmax - zmin
        half = round((zmax - zmin)/2)
        if zmax>zmin+1: #at least two voxels apart
            zmin_upper = blank_below
            zmax_lower = blank_above
            delta =  -(zs[i] - zs[i-1]) - blank_below + blank_above
            r1 = recons[i-1][:,:,zmax_lower-delta:zmax_lower]
            r2 = recons[i][:,:,zmin_upper:zmin_upper+delta]
            if method=='midslice':
                recon_aligned[:,:,:,zmin:zmin+half] = r1[:,:,:half]
                recon_aligned[:,:,:,zmin+half:zmax] = r2[:,:,half:] 
            elif method=='average':
                recon_aligned[:,:,:,zmin:zmax] = 0.5 * (r1 + r2)
            elif method=='crossfade':
                idx = torch.arange(dL).to(pytomography.device) + 0.5
                recon_aligned[:,:,:,zmin:zmax] = ((dL-idx)*r1 + idx*r2) / dL
            elif method=='TEM':
                stitch_index = torch.min(torch.abs(r1-r2), axis=2)[1]
                range_tensor = torch.arange(dL).unsqueeze(0).unsqueeze(0).to(pytomography.device)
                mask_tensor = range_tensor < stitch_index.unsqueeze(-1)
                expanded_mask = mask_tensor.expand(*stitch_index.shape, dL)
                recon_aligned[:,:,:,zmin:zmax][expanded_mask.unsqueeze(0)] = r1[expanded_mask]
                recon_aligned[:,:,:,zmin:zmax][~expanded_mask.unsqueeze(0)] = r2[~expanded_mask]
    return recon_aligned


# TODO: Update this function so that it includes photopeak energy window index, and psf should be computed using data tables corresponding to manufactorer data sheets.
'''
def get_SPECT_recon_algorithm_dicom(
    projections_file: str,
    atteunation_files: Sequence[str] = None,
    use_psf: bool = False,
    scatter_type: str|None = None,
    prior: Prior = None,
    recon_algorithm_class: OSML = OSEMOSL,
    object_initial: torch.Tensor | None = None,
) -> OSML:
    """Helper function to quickly create reconstruction algorithm given SPECT DICOM files and CT dicom files.

    Args:
        projections_file (str): DICOM filepath corresponding to SPECT data.
        atteunation_files (Sequence[str], optional): DICOM filepaths corresponding to CT data. If None, then atteunation correction is not used. Defaults to None.
        use_psf (bool, optional): Whether or not to use PSF modeling. Defaults to False.
        scatter_type (str | None, optional): Type of scatter correction used in reconstruction. Defaults to None.
        prior (Prior, optional): Bayesian Prior used in reconstruction algorithm. Defaults to None.
        recon_algorithm_class (nn.Module, optional): Type of reconstruction algorithm used. Defaults to OSEMOSL.
        object_initial (torch.Tensor | None, optional): Initial object used in reconstruction. If None, defaults to all ones. Defaults to None.

    Raises:
        Exception: If not able to compute relevant PSF parameters from DICOM data and corresponding data tables.

    Returns:
        OSML: Reconstruction algorithm used.
    """
    # Get projections/scatter estimate
    if scatter_type==None:
        object_meta, image_meta, projections = get_projections(projections_file)
        projections_scatter = 0 # equivalent to 0 estimated scatter everywhere
    else:
        object_meta, image_meta, projections, projections_scatter = dicom_MEW_to_data(projections_file, type=scatter_type)
    # obj2obj and im2im nets.
    object_correction_nets = []
    image_correction_nets = []
    # Load attenuation data
    if atteunation_files is not None:
        CT = dicom_CT_to_data(atteunation_files, projections_file)
        CT_net = SPECTAttenuationTransform(CT.unsqueeze(dim=0))
        object_correction_nets.append(CT_net)
    # Load PSF parameters
    if use_psf:
        ds = pydicom.read_file(projections_file, force=True)
        if ds.Manufacturer =='SIEMENS NM':
            # Find a more consistent way to do this
            angular_FWHM = ds[0x0055, 0x107f][0]
            psf_meta = PSFMeta(collimator_slope = angular_FWHM/(2*np.sqrt(2*np.log(2))), collimator_intercept = 0.0)
            psf_net = SPECTPSFTransform(psf_meta)
        else:
            raise Exception('Unable to compute PSF metadata from this DICOM file')
        object_correction_nets.append(psf_net)
    system_matrix = SystemMatrix(object_correction_nets, image_correction_nets, object_meta, image_meta)
    recon_algorithm = recon_algorithm_class(projections, system_matrix, object_initial, projections_scatter, prior)
    return recon_algorithm
'''