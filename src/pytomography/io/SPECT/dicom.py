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
from pytomography.utils import get_blank_below_above, compute_TEW, get_mu_from_spectrum_interp
from ..CT import get_HU2mu_conversion, open_CT_file, compute_max_slice_loc_CT, compute_slice_thickness_CT
from ..shared import create_ds

def parse_projection_dataset(ds: Dataset) -> Sequence[torch.Tensor, np.array, np.array, dict]:
    """Gets projections with corresponding radii and angles corresponding to projection data from a DICOM file.

    Args:
        ds (Dataset): pydicom dataset object.

    Returns:
        (torch.tensor[EWindows, TimeWindows, Ltheta, Lr, Lz], np.array, np.array): Returns (i) projection data (ii) angles (iii) radii and (iv) flags for whether or not multiple energy windows/time slots were detected.
    """
    flags = {'multi_energy_window': False, 'multi_time_slot': False}
    pixel_array = ds.pixel_array
    # Energy Window Vector
    energy_window_vector = np.array(ds.EnergyWindowVector)
    detector_vector = np.array(ds.DetectorVector)
    # Time slot vector
    try:
        time_slot_vector = np.array(ds.TimeSlotVector)
    except:
        time_slot_vector = np.ones(len(detector_vector)).astype(int)
    # Update flags
    if len(np.unique(energy_window_vector))>1:
        flags['multi_energy_window'] = True
    if len(np.unique(time_slot_vector))>1:
        flags['multi_time_slot'] = True
    # Get radii and angles
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
    
    projections = []
    for energy_window in np.unique(energy_window_vector):
        t_slot_projections = []
        for time_slot in np.unique(time_slot_vector):
            pixel_array_i = pixel_array[(time_slot_vector==time_slot)*(energy_window_vector==energy_window)]
            t_slot_projections.append(pixel_array_i)
        projections.append(t_slot_projections)
    projections = np.array(projections)
    
    angles = (angles + 180)%360 # to detector angle convention
    sorted_idxs = np.argsort(angles)
    projections = np.transpose(projections[:,:,sorted_idxs,::-1], (0,1,2,4,3)).astype(np.float32)
    projections= torch.tensor(projections.copy()).to(pytomography.dtype).to(pytomography.device) 
    return (projections,
             angles[sorted_idxs],
             radii[sorted_idxs]/10,
             flags)
    
def get_metadata(
    file: str,
    index_peak: int = 0,
) -> Sequence[SPECTObjectMeta, SPECTProjMeta]:
    """Gets PyTomography metadata from a .dcm file.

    Args:
        file (str): Path to the .dcm file of SPECT projection data.
        index_peak (int): EnergyWindowInformationSequence index corresponding to the photopeak. Defaults to 0.
    Returns:
        (ObjectMeta, ProjMeta): Required metadata information for reconstruction in PyTomography.
    """
    ds = pydicom.read_file(file, force=True)
    dx = ds.PixelSpacing[0] / 10
    dz = ds.PixelSpacing[1] / 10
    dr = (dx, dx, dz)
    projections, angles, radii, _ = parse_projection_dataset(ds)
    shape_proj= (projections.shape[-3], projections.shape[-2], projections.shape[-1])
    shape_obj = (shape_proj[1], shape_proj[1], shape_proj[2])
    object_meta = SPECTObjectMeta(dr,shape_obj)
    proj_meta = SPECTProjMeta((shape_proj[1], shape_proj[2]), angles, radii)
    object_meta.affine_matrix = _get_affine_spect_projections(file)
    proj_meta.filepath = file
    proj_meta.index_peak = index_peak
    return object_meta, proj_meta

def get_projections(
    file: str,
    index_peak: None | int = None,
    index_time: None | int = None,
    ) -> Sequence[SPECTObjectMeta, SPECTProjMeta, torch.Tensor]:
    """Gets projections from a .dcm file.

    Args:
        file (str): Path to the .dcm file of SPECT projection data.
        index_peak (int): If not none, then the returned projections correspond to the index of this energy window. Otherwise returns all energy windows. Defaults to None.
        index_time (int): If not none, then the returned projections correspond to the index of the time slot in gated SPECT. Otherwise returns all time slots. Defaults to None
    Returns:
        (SPECTObjectMeta, SPECTProjMeta, torch.Tensor[..., Ltheta, Lr, Lz]) where ... depends on if time slots are considered.
    """
    ds = pydicom.read_file(file, force=True)
    projections, _, _, flags = parse_projection_dataset(ds)
    if index_peak is not None:
        projections = projections[index_peak].unsqueeze(dim=0)
        flags['multi_energy_window'] = False
    if index_time is not None:
        projections = projections[:,index_time].unsqueeze(dim=1)
        flags['multi_time_slot'] = False
    projections = projections.squeeze()
    dimension_list = ['Ltheta', 'Lr', 'Lz']
    if flags['multi_time_slot']:
        dimension_list = ['N_timeslots'] + dimension_list
        if pytomography.verbose: print('Multiple time slots found')
    if flags['multi_energy_window']:
        dimension_list = ['N_energywindows'] + dimension_list
        if pytomography.verbose: print('Multiple energy windows found')
    if len(dimension_list)==3:
        dimension_list = ['1'] + dimension_list
        projections = projections.unsqueeze(dim=0)
    if pytomography.verbose: print(f'Returned projections have dimensions ({" ".join(dimension_list)})')
    return projections

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
    projections_all = get_projections(file)
    scatter = compute_TEW(projections_all[index_lower],projections_all[index_upper], ww_lower, ww_upper, ww_peak)
    return scatter.to(pytomography.device).unsqueeze(0)

def get_attenuation_map_from_file(file_AM: str) -> torch.Tensor:
    """Gets an attenuation map from a DICOM file. This data is usually provided by the manufacturer of the SPECT scanner. 

    Args:
        file_AM (str): File name of attenuation map

    Returns:
        torch.Tensor: Tensor of shape [batch_size, Lx, Ly, Lz] corresponding to the atteunation map in units of cm:math:`^{-1}`
    """
    ds = pydicom.read_file(file_AM, force=True)
    # DICOM header for scale factor that shows up sometimes
    if (0x033,0x1038) in ds:
        scale_factor = 1/ds[0x033,0x1038].value
    else:
        scale_factor = 1
    attenuation_map =  ds.pixel_array * scale_factor
    
    return torch.tensor(np.transpose(attenuation_map, (2,1,0))).unsqueeze(dim=0).to(pytomography.dtype).to(pytomography.device)

def get_psfmeta_from_scanner_params(
    collimator_name: str,
    energy_keV: float,
    min_sigmas: float = 3,
    material: str = 'lead',
    intrinsic_resolution: float = 0,
    ) -> SPECTPSFMeta:
    """Obtains SPECT PSF metadata given a unique collimator code and photopeak energy of radionuclide. For more information on collimator codes, see the "external data" section of the readthedocs page.

    Args:
        collimator_name (str): Code for the collimator used.
        energy_keV (float): Energy of the photopeak
        min_sigmas (float): Minimum size of the blurring kernel used. Fixes the convolutional kernel size so that all locations have at least ``min_sigmas`` in dimensions (some will be greater)
        material (str): Material of the collimator.
        intrinsic_resolution (float): Intrinsic resolution (FWHM) of the scintillator crystals. Defaults to 0.

    Returns:
        SPECTPSFMeta: PSF metadata.
    """

    module_path = os.path.dirname(os.path.abspath(__file__))
    collimator_filepath = os.path.join(module_path, '../../data/collim.col')
    with open(collimator_filepath) as f:
        collimator_data = f.readlines()
    collimator_data = np.array(collimator_data)
    try:
        line = collimator_data[np.char.find(collimator_data, collimator_name)>=0][0]
    except:
        Exception(f'Cannot find data for collimator name {collimator_name}. For a list of available collimator names, run `from pytomography.utils import print_collimator_parameters` and then `print_collimator_parameters()`.')
    
    # TODO: Support for other collimator types. Right now just parallel hole
    hole_length = float(line.split()[3])
    hole_diameter = float(line.split()[1])

    lead_attenuation = get_mu_from_spectrum_interp(os.path.join(module_path, f'../../data/NIST_attenuation_data/{material}.csv'), energy_keV)
    
    FWHM2sigma = 1/(2*np.sqrt(2*np.log(2)))
    collimator_slope = hole_diameter/(hole_length - (2/lead_attenuation)) * FWHM2sigma
    collimator_intercept = hole_diameter * FWHM2sigma
    intrinsic_resolution = intrinsic_resolution * FWHM2sigma
    
    sigma_fit = lambda r, a, b, c: np.sqrt((a*r+b)**2+c**2)
    sigma_fit_params = [collimator_slope, collimator_intercept, intrinsic_resolution]
    
    return SPECTPSFMeta(
        sigma_fit_params=sigma_fit_params,
        sigma_fit=sigma_fit,
        min_sigmas=min_sigmas
        )

def CT_to_mumap(
    CT: torch.tensor,
    files_CT: Sequence[str],
    file_NM: str,
    index_peak=0
    ) -> torch.tensor:
    """Converts a CT image to a mu-map given SPECT projection data. The CT data must be aligned with the projection data already; this is a helper function for ``get_attenuation_map_from_CT_slices``.

    Args:
        CT (torch.tensor): CT object in units of HU
        files_CT (Sequence[str]): Filepaths of all CT slices
        file_NM (str): Filepath of SPECT projectio ndata
        index_peak (int, optional): Index of EnergyInformationSequence corresponding to the photopeak. Defaults to 0.

    Returns:
        torch.tensor: Attenuation map in units of 1/cm
    """
    ds_NM = pydicom.read_file(file_NM)
    window_upper = ds_NM.EnergyWindowInformationSequence[index_peak].EnergyWindowRangeSequence[0].EnergyWindowUpperLimit
    window_lower = ds_NM.EnergyWindowInformationSequence[index_peak].EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
    E_SPECT = (window_lower + window_upper)/2
    KVP = pydicom.read_file(files_CT[0]).KVP
    HU2mu_conversion = get_HU2mu_conversion(files_CT, KVP, E_SPECT)
    return HU2mu_conversion(CT)
    

def get_attenuation_map_from_CT_slices(
    files_CT: Sequence[str],
    file_NM: str | None = None,
    index_peak: int = 0,
    keep_as_HU: bool = False,
    mode: str = 'constant',
    CT_output_shape: Sequence[int] | None = None,
    apply_affine: bool = True,
    ) -> torch.Tensor:
    """Converts a sequence of DICOM CT files (corresponding to a single scan) into a torch.Tensor object usable as an attenuation map in PyTomography.

    Args:
        files_CT (Sequence[str]): List of all files corresponding to an individual CT scan
        file_NM (str): File corresponding to raw PET/SPECT data (required to align CT with projections). If None, then no alignment is done. Defaults to None.
        index_peak (int, optional): Index corresponding to photopeak in projection data. Defaults to 0.
        keep_as_HU (bool): If True, then don't convert to linear attenuation coefficient and keep as Hounsfield units. Defaults to False
        CT_output_shape (Sequence, optional): If not None, then the CT is returned with the desired dimensions. Otherwise, it defaults to the shape in the file_NM data.
        apply_affine (bool): Whether or not to align CT with NM.

    Returns:
        torch.Tensor: Tensor of shape [Lx, Ly, Lz] corresponding to attenuation map.
    """
    
    CT_HU = open_CT_file(files_CT)
    
    if file_NM is None:
        return torch.tensor(CT_HU[:,:,::-1].copy()).unsqueeze(dim=0).to(pytomography.dtype).to(pytomography.device)
    ds_NM = pydicom.read_file(file_NM)
    # When doing affine transform, fill outside with point below -1000HU so it automatically gets converted to mu=0 after bilinear transform
    if CT_output_shape is None:
        CT_output_shape = (ds_NM.Rows, ds_NM.Rows, ds_NM.Columns)
    if apply_affine:
        # Align with SPECT:
        M_CT = _get_affine_CT(files_CT)
        M_NM = _get_affine_spect_projections(file_NM)
        # Resample CT and convert to mu at 208keV and save
        M = npl.inv(M_CT) @ M_NM
        CT_HU = affine_transform(CT_HU, M, output_shape=CT_output_shape, mode=mode, cval=-1500)
    if keep_as_HU:
        CT = CT_HU
    else:
        CT= CT_to_mumap(CT_HU, files_CT, file_NM, index_peak)
    CT = torch.tensor(CT[:,:,::-1].copy()).unsqueeze(dim=0).to(pytomography.dtype).to(pytomography.device)
    return CT

def _get_affine_spect_projections(filename:str) -> np.array:
    """Computes an affine matrix corresponding the coordinate system of a SPECT DICOM file of projections.

    Args:
        ds (Dataset): DICOM dataset of projection data

    Returns:
        np.array: Affine matrix
    """
    # Note: per DICOM convention z actually decreases as the z-index increases (initial z slices start with the head)
    ds = pydicom.read_file(filename)
    Sx, Sy, Sz = ds.DetectorInformationSequence[0].ImagePositionPatient
    dx = dy = ds.PixelSpacing[0]
    dz = ds.PixelSpacing[1]
    Sx -= ds.Rows / 2 * dx
    Sy -= ds.Rows / 2 * dy
    Sy -= ds.RotationInformationSequence[0].TableHeight
    M = np.zeros((4,4))
    M[0] = np.array([dx, 0, 0, Sx])
    M[1] = np.array([0, dy, 0, Sy])
    M[2] = np.array([0, 0, -dz, Sz])
    M[3] = np.array([0, 0, 0, 1])
    return M

def _get_affine_CT(filenames: Sequence[str]):
    """Computes an affine matrix corresponding the coordinate system of a CT DICOM file. Note that since CT scans consist of many independent DICOM files, ds corresponds to an individual one of these files. This is why the maximum z value is also required (across all seperate independent DICOM files).

    Args:
        ds (Dataset): DICOM dataset of CT data
        max_z (float): Maximum value of z across all axial slices that make up the CT scan

    Returns:
        np.array: Affine matrix corresponding to CT scan.
    """
    # Note: per DICOM convention z actually decreases as the z-index increases (initial z slices start with the head)
    ds = pydicom.read_file(filenames[0])
    dz = compute_slice_thickness_CT(filenames)
    max_z = compute_max_slice_loc_CT(filenames)
    M = np.zeros((4,4))
    M[0:3, 0] = np.array(ds.ImageOrientationPatient[0:3])*ds.PixelSpacing[0]
    M[0:3, 1] = np.array(ds.ImageOrientationPatient[3:])*ds.PixelSpacing[1]
    M[0:3, 2] = - np.array([0,0,1]) * dz 
    M[0:2, 3] = np.array(ds.ImagePositionPatient)[0:2] 
    M[2, 3] = max_z
    M[3, 3] = 1
    return M

def stitch_multibed(
    recons: torch.Tensor,
    files_NM: Sequence[str],
    method: str ='midslice',
    ) -> torch.Tensor:
    """Stitches together multiple reconstructed objects corresponding to different bed positions.

    Args:
        recons (torch.Tensor[n_beds, Lx, Ly, Lz]): Reconstructed objects. The first index of the tensor corresponds to different bed positions
        files_NM (list): List of length ``n_beds`` corresponding to the DICOM file of each reconstruction
        method (str, optional): Method to perform stitching (see https://doi.org/10.1117/12.2254096 for all methods described). Available methods include ``'midslice'``, ``'average'``, ``'crossfade'``, and ``'TEM'`` (transition error minimization).

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
    blank_below, blank_above = get_blank_below_above(get_projections(files_NM[0]))
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

def save_dcm(
    save_path: str,
    object: torch.Tensor,
    file_NM: str,
    recon_name: str = ''
    ) -> None:
    """Saves the reconstructed object `object` to a series of DICOM files in the folder given by `save_path`. Requires the filepath of the projection data `file_NM` to get Study information.

    Args:
        object (torch.Tensor): Reconstructed object of shape [1,Lx,Ly,Lz].
        save_path (str): Location of folder where to save the DICOM output files.
        file_NM (str): File path of the projection data corresponding to the reconstruction.
        recon_name (str): Type of reconstruction performed. Obtained from the `recon_method_str` attribute of a reconstruction algorithm class.
    """
    try:
        Path(save_path).resolve().mkdir(parents=True, exist_ok=False)
    except:
        raise Exception(f'Folder {save_path} already exists; new folder name is required.')
    # Convert tensor image to numpy array
    pixel_data = torch.permute(object.squeeze(),(2,1,0)).cpu().numpy()    
    scale_factor = (2**16 - 1) / pixel_data.max()
    pixel_data *= scale_factor #maximum dynamic range
    pixel_data = pixel_data.astype(np.uint16)
    # Get affine information
    ds_NM = pydicom.dcmread(file_NM)
    Sx, Sy, Sz = ds_NM.DetectorInformationSequence[0].ImagePositionPatient
    dx = dy = ds_NM.PixelSpacing[0]
    dz = ds_NM.PixelSpacing[1]
    Sx -= ds_NM.Rows / 2 * dx
    Sy -= ds_NM.Rows / 2 * dy
    # Y-Origin point at tableheight=0
    Sy -= ds_NM.RotationInformationSequence[0].TableHeight
    # Sz now refers to location of lowest slice
    Sz -= (pixel_data.shape[0]-1) * dz
    SOP_instance_UID = generate_uid()
    SOP_class_UID = '1.2.840.10008.5.1.4.1.1.128' #SPECT storage
    modality = 'PT' # SPECT storage
    ds = create_ds(ds_NM, SOP_instance_UID, SOP_class_UID, modality)
    ds.Rows, ds.Columns = pixel_data.shape[1:]
    ds.SeriesNumber = 1
    ds.NumberOfSlices = pixel_data.shape[0]
    ds.PixelSpacing = [dx,dy]
    ds.SliceThickness = dz
    ds.SpacingBetweenSlices = dz
    ds.ImageOrientationPatient = [1,0,0,0,1,0]
    ds.RescaleSlope = 1/scale_factor
    # Set other things
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.PixelRepresentation = 0
    ds.ReconstructionMethod = recon_name
    # Create all slices
    for i in range(pixel_data.shape[0]):
        # Load existing DICOM file
        ds_i = ds.copy()
        ds_i.InstanceNumber = i+1
        ds_i.ImagePositionPatient = [Sx,Sy,Sz+i*dz]
        # Create SOP Instance UID unique to slice
        SOP_instance_UID_slice = f'{ds_i.SOPInstanceUID[:-3]}{i+1:03d}'
        ds_i.SOPInstanceUID = SOP_instance_UID_slice 
        ds_i.file_meta.MediaStorageSOPInstanceUID = (SOP_instance_UID_slice)
        # Set the pixel data
        ds_i.PixelData = pixel_data[i].tobytes()
        ds_i.save_as(os.path.join(save_path, f'{ds.SOPInstanceUID}.dcm'))
        
def open_SPECT_file(
    files_SPECT: Sequence[str],
    ) -> np.array:
    """Given a list of seperate SPECT DICOM files, opens them up and stacks them together into a single SPECT image. 

    Args:
        files_SPECT (Sequence[str]): List of SPECT DICOM filepaths corresponding to different z slices of the same scan.

    Returns:
        np.array: SPECT scan with units of vendor.
    """
    SPECT_slices = []
    slice_locs = []
    for file in files_SPECT:
        ds = pydicom.read_file(file)
        SPECT_slices.append(ds.RescaleSlope * ds.pixel_array + ds.RescaleIntercept)
        slice_locs.append(float(ds.ImagePositionPatient[2]))
    SPECT_image = np.transpose(np.array(SPECT_slices)[np.argsort(slice_locs)], (2,1,0)).astype(np.float32)
    return SPECT_image