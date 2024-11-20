from __future__ import annotations
import warnings
import copy
import os
from functools import partial
import collections.abc
from collections.abc import Sequence, Callable
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
from rt_utils import RTStructBuilder
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta, StarGuideProjMeta
import nibabel as nib
import pandas as pd
from pytomography.utils import (
    compute_EW_scatter,
    get_mu_from_spectrum_interp,
)
from ..shared import (
    open_multifile,
    open_singlefile,
    _get_affine_multifile,
    _get_affine_single_file,
    create_ds,
    align_images_affine
)
from .attenuation_map import get_HU2mu_conversion as get_HU2mu_conversion_old

def parse_projection_dataset(
    ds: Dataset,
) -> Sequence[torch.Tensor, np.array, np.array, dict]:
    """Gets projections with corresponding radii and angles corresponding to projection data from a DICOM file.

    Args:
        ds (Dataset): pydicom dataset object.

    Returns:
        (torch.tensor[EWindows, TimeWindows, Ltheta, Lr, Lz], np.array, np.array): Returns (i) projection data (ii) angles (iii) radii and (iv) flags for whether or not multiple energy windows/time slots were detected.
    """
    flags = {"multi_energy_window": False, "multi_time_slot": False}
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
    if len(np.unique(energy_window_vector)) > 1:
        flags["multi_energy_window"] = True
    if len(np.unique(time_slot_vector)) > 1:
        flags["multi_time_slot"] = True
    # Get radii and angles
    detectors = np.array(ds.DetectorVector)
    radii = np.array([])
    angles = np.array([])
    for detector in np.unique(detectors):
        n_angles = ds.RotationInformationSequence[0].NumberOfFramesInRotation
        delta_angle = ds.RotationInformationSequence[0].AngularStep
        try:
            start_angle = ds.DetectorInformationSequence[detector - 1].StartAngle
        except:
            start_angle = ds.RotationInformationSequence[0].StartAngle
        rotation_direction = ds.RotationInformationSequence[0].RotationDirection
        if rotation_direction == "CC" or rotation_direction == "CCW":
            angles = np.concatenate(
                [angles, start_angle + delta_angle * np.arange(n_angles)]
            )
        else:
            angles = np.concatenate(
                [angles, start_angle - delta_angle * np.arange(n_angles)]
            )
        try:
            radial_positions_detector = ds.DetectorInformationSequence[
                detector - 1
            ].RadialPosition
        except AttributeError:
            radial_positions_detector = ds.RotationInformationSequence[
                detector - 1
            ].RadialPosition
        if not isinstance(radial_positions_detector, collections.abc.Sequence):
            radial_positions_detector = n_angles * [radial_positions_detector]
        radii = np.concatenate([radii, radial_positions_detector])
    radii /= 10 # convert to cm
    # Try to access GE Xeleris information if it exists
    try:
        radii_offset = np.array(ds[0x0055,0x1022][0][0x0013,0x101e].value).reshape(-1,3)[:,-1] / 10
        radii += radii_offset
    except:
        pass
    projections = []
    for energy_window in np.unique(energy_window_vector):
        t_slot_projections = []
        for time_slot in np.unique(time_slot_vector):
            pixel_array_i = pixel_array[
                (time_slot_vector == time_slot)
                * (energy_window_vector == energy_window)
            ]
            t_slot_projections.append(pixel_array_i)
        projections.append(t_slot_projections)
    projections = np.array(projections)

    angles = (angles + 180) % 360  # to detector angle convention
    sorted_idxs = np.argsort(angles)
    projections = np.transpose(
        projections[:, :, sorted_idxs, ::-1], (0, 1, 2, 4, 3)
    ).astype(np.float32)
    projections = (
        torch.tensor(projections.copy()).to(pytomography.dtype).to(pytomography.device)
    )
    return (projections, angles[sorted_idxs], radii[sorted_idxs], flags)


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
    ds = pydicom.dcmread(file, force=True)
    dx = ds.PixelSpacing[0] / 10
    dz = ds.PixelSpacing[1] / 10
    dr = (dx, dx, dz)
    projections, angles, radii, _ = parse_projection_dataset(ds)
    shape_proj = (projections.shape[-3], projections.shape[-2], projections.shape[-1])
    shape_obj = (shape_proj[1], shape_proj[1], shape_proj[2])
    object_meta = SPECTObjectMeta(dr, shape_obj)
    proj_meta = SPECTProjMeta((shape_proj[1], shape_proj[2]), (dx, dz), angles, radii)
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
    ds = pydicom.dcmread(file, force=True)
    projections, _, _, flags = parse_projection_dataset(ds)
    if index_peak is not None:
        projections = projections[index_peak].unsqueeze(dim=0)
        flags["multi_energy_window"] = False
    if index_time is not None:
        projections = projections[:, index_time].unsqueeze(dim=1)
        flags["multi_time_slot"] = False
    projections = projections.squeeze()
    dimension_list = ["Ltheta", "Lr", "Lz"]
    if flags["multi_time_slot"]:
        dimension_list = ["N_timeslots"] + dimension_list
        if pytomography.verbose:
            print("Multiple time slots found")
    if flags["multi_energy_window"]:
        dimension_list = ["N_energywindows"] + dimension_list
        if pytomography.verbose:
            print("Multiple energy windows found")
    if pytomography.verbose:
        print(f'Returned projections have dimensions ({" ".join(dimension_list)})')
    return projections

def get_energy_window_bounds(file_NM: str, idx: int) -> tuple[float, float]:
    """Get the energy window bounds from a DICOM file corresponding to energy window index idx.

    Args:
        file_NM (str): File to get energy window bounds from.
        idx (int): Index of the energy window

    Returns:
        tuple[float, float]: Lower and upper bounds
    """
    ds = pydicom.dcmread(file_NM)
    energy_window = ds.EnergyWindowInformationSequence[idx]
    window_lower = energy_window.EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
    window_upper = energy_window.EnergyWindowRangeSequence[0].EnergyWindowUpperLimit
    return window_lower, window_upper

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

def get_energy_window_scatter_estimate(
    file: str,
    index_peak: int,
    index_lower: int,
    index_upper: int | None = None,
    weighting_lower: float = 0.5,
    weighting_upper: float = 0.5,
    proj_meta = None,
    sigma_theta: float = 0,
    sigma_r: float = 0,
    sigma_z: float = 0,
    N_sigmas: int = 3,
    return_scatter_variance_estimate: bool = False
) -> torch.Tensor:
    """Gets an estimate of scatter projection data from a DICOM file using either the dual energy window (`index_upper=None`) or triple energy window method.

    Args:
        file (str): Filepath of the DICOM file
        index_peak (int): Index of the ``EnergyWindowInformationSequence`` DICOM attribute corresponding to the photopeak.
        index_lower (int): Index of the ``EnergyWindowInformationSequence`` DICOM attribute corresponding to lower scatter window.
        index_upper (int): Index of the ``EnergyWindowInformationSequence`` DICOM attribute corresponding to upper scatter window. Defaults to None (dual energy window).
        weighting_lower (float): Weighting of the lower scatter window. Defaults to 0.5.
        weighting_upper (float): Weighting of the upper scatter window. Defaults to 0.5.
        return_scatter_variance_estimate (bool): If true, then also return the variance estimate of the scatter. Defaults to False.
    Returns:
        torch.Tensor[Ltheta,Lr,Lz]: Tensor corresponding to the scatter estimate.
    """
    projections_all = get_projections(file).to(pytomography.device)
    return get_energy_window_scatter_estimate_projections(file, projections_all, index_peak, index_lower, index_upper, weighting_lower, weighting_upper, proj_meta, sigma_theta, sigma_r, sigma_z, N_sigmas, return_scatter_variance_estimate)

def get_energy_window_scatter_estimate_projections(
    file: str,
    projections: torch.Tensor,
    index_peak: int,
    index_lower: int,
    index_upper: int | None = None,
    weighting_lower: float = 0.5,
    weighting_upper: float = 0.5,
    proj_meta = None,
    sigma_theta: float = 0,
    sigma_r: float = 0,
    sigma_z: float = 0,
    N_sigmas: int = 3,
    return_scatter_variance_estimate: bool = False
) -> torch.Tensor:
    """Gets an estimate of scatter projection data from a DICOM file using either the dual energy window (`index_upper=None`) or triple energy window method. This is seperate from ``get_energy_window_scatter_estimate`` as it allows a user to input projecitons that are already loaded/modified. This is useful for when projection data gets mixed for reconstructing multiple bed positions.

    Args:
        file (str): Filepath of the DICOM file
        projections (torch.Tensor): Loaded projection data
        index_peak (int): Index of the ``EnergyWindowInformationSequence`` DICOM attribute corresponding to the photopeak.
        index_lower (int): Index of the ``EnergyWindowInformationSequence`` DICOM attribute corresponding to lower scatter window.
        index_upper (int): Index of the ``EnergyWindowInformationSequence`` DICOM attribute corresponding to upper scatter window.
        weighting_lower (float): Weighting of the lower scatter window. Defaults to 0.5.
        weighting_upper (float): Weighting of the upper scatter window. Defaults to 0.5.
        return_scatter_variance_estimate (bool): If true, then also return the variance estimate of the scatter. Defaults to False.
    Returns:
        torch.Tensor[Ltheta,Lr,Lz]: Tensor corresponding to the scatter estimate.
    """
    ds = pydicom.dcmread(file, force=True)
    ww_peak = get_window_width(ds, index_peak)
    ww_lower = get_window_width(ds, index_lower)
    ww_upper = get_window_width(ds, index_upper) if index_upper is not None else None
    projections_lower = projections[index_lower]
    projections_upper = projections[index_upper] if index_upper is not None else None
    scatter = compute_EW_scatter(
        projections_lower,
        projections_upper,
        ww_lower,
        ww_upper,
        ww_peak,
        weighting_lower,
        weighting_upper,
        proj_meta,
        sigma_theta,
        sigma_r,
        sigma_z,
        N_sigmas,
        return_scatter_variance_estimate
    )
    return scatter

def get_attenuation_map_from_file(file_AM: str) -> torch.Tensor:
    """Gets an attenuation map from a DICOM file. This data is usually provided by the manufacturer of the SPECT scanner.

    Args:
        file_AM (str): File name of attenuation map

    Returns:
        torch.Tensor: Tensor of shape [batch_size, Lx, Ly, Lz] corresponding to the atteunation map in units of cm:math:`^{-1}`
    """
    ds = pydicom.dcmread(file_AM, force=True)
    # DICOM header for scale factor that shows up sometimes
    if (0x033, 0x1038) in ds:
        scale_factor = 1 / ds[0x033, 0x1038].value
    elif (0x0028, 0x1053) in ds:
        scale_factor = ds[0x0028, 0x1053].value
    else:
        scale_factor = 1.0
    attenuation_map = ds.pixel_array.astype(np.float32) * scale_factor
    return torch.tensor(np.transpose(attenuation_map, (2, 1, 0))).to(pytomography.dtype).to(pytomography.device)


def get_psfmeta_from_scanner_params(
    collimator_name: str,
    energy_keV: float,
    min_sigmas: float = 3,
    material: str = 'lead',
    intrinsic_resolution: float = 0,
    intrinsic_resolution_140keV: float | None = None,
    shape: str = 'gaussian'
    ) -> SPECTPSFMeta:
    """Obtains SPECT PSF metadata given a unique collimator code and photopeak energy of radionuclide. For more information on collimator codes, see the "external data" section of the readthedocs page.

    Args:
        collimator_name (str): Code for the collimator used.
        energy_keV (float): Energy of the photopeak
        min_sigmas (float): Minimum size of the blurring kernel used. Fixes the convolutional kernel size so that all locations have at least ``min_sigmas`` in dimensions (some will be greater)
        material (str): Material of the collimator.
        intrinsic_resolution (float): Intrinsic resolution (FWHM) of the scintillator crystals. Note that most scanners provide the intrinsic resolution at 140keV only; if you only have access to this, you should use the ``intrinsic_resolution_140keV`` argument of this function. Defaults to 0.
        intrinsic_resolution_140keV (float | None): Intrinsic resolution (FWHM) of the scintillator crystals at an energy of 140keV. The true intrinsic resolution is calculated assuming the resolution is proportional to E^(-1/2). If provided, then ``intrinsic_resolution`` is ignored. Defaults to None.
        shape (str, optional): Shape of the PSF. Defaults to 'gaussian', in which case sigma is the sigma of the Gaussian. Can also be 'square' for square collimators, in this case sigma is half the diameter of the bore.

    Returns:
        SPECTPSFMeta: PSF metadata.
    """

    module_path = os.path.dirname(os.path.abspath(__file__))
    collimator_filepath = os.path.join(module_path, "../../data/collim.col")
    with open(collimator_filepath) as f:
        collimator_data = f.readlines()
    collimator_data = np.array(collimator_data)
    try:
        line = collimator_data[np.char.find(collimator_data, collimator_name) >= 0][0]
    except:
        Exception(
            f"Cannot find data for collimator name {collimator_name}. For a list of available collimator names, run `from pytomography.utils import print_collimator_parameters` and then `print_collimator_parameters()`."
        )
    hole_length = float(line.split()[3])
    hole_diameter = float(line.split()[1])
    lead_attenuation = get_mu_from_spectrum_interp(os.path.join(module_path, f'../../data/NIST_attenuation_data/{material}.csv'), energy_keV)
    collimator_slope = hole_diameter/(hole_length - (2/lead_attenuation))
    collimator_intercept = hole_diameter
    if shape=='gaussian':
        FWHM2sigma = 1/(2*np.sqrt(2*np.log(2)))
        collimator_slope *= FWHM2sigma
        collimator_intercept *= FWHM2sigma
        if intrinsic_resolution_140keV is not None:
            intrinsic_resolution = intrinsic_resolution_140keV * (energy_keV/140)**(-1/2) * FWHM2sigma
        else:
            intrinsic_resolution = intrinsic_resolution * FWHM2sigma
    elif shape=='box':
        collimator_slope /= 2 # half the diameter
        collimator_intercept /= 2
        intrinsic_resolution = 0 # dont include for square
    sigma_fit = lambda r, a, b, c: np.sqrt((a*r+b)**2+c**2)
    sigma_fit_params = [collimator_slope, collimator_intercept, intrinsic_resolution]
    
    return SPECTPSFMeta(
        sigma_fit_params=sigma_fit_params,
        sigma_fit=sigma_fit,
        min_sigmas=min_sigmas,
        shape=shape
    )

def CT_to_mumap(
    CT: torch.tensor,
    files_CT: Sequence[str],
    file_NM: str,
    index_peak: int = 0,
    technique: str | Callable ='from_table',
    E_SPECT: float | None = None
) -> torch.tensor:
    """Converts a CT image to a mu-map given SPECT projection data. The CT data must be aligned with the projection data already; this is a helper function for ``get_attenuation_map_from_CT_slices``.

    Args:
        CT (torch.tensor): CT object in units of HU
        files_CT (Sequence[str]): Filepaths of all CT slices
        file_NM (str): Filepath of SPECT projectio ndata
        index_peak (int, optional): Index of EnergyInformationSequence corresponding to the photopeak. Defaults to 0.
        technique (str, optional): Technique to convert HU to attenuation coefficients. The default, 'from_table', uses a table of coefficients for bilinear curves obtained for a variety of common radionuclides. The technique 'from_cortical_bone_fit' looks for a cortical bone peak in the scan and uses that to obtain the bilinear coefficients. For phantom scans where the attenuation coefficient is always significantly less than bone, the cortical bone technique will still work, since the first part of the bilinear curve (in the air to water range) does not depend on the cortical bone fit. Alternatively, one can provide an arbitrary function here which takes in a 3D scan with units of HU and converts to mu.
        E_SPECT (float): Energy of the photopeak in SPECT scan; this overrides the energy in the DICOM file, so should only be used if the DICOM file is incorrect. If None, then the energy is obtained from the DICOM file.

    Returns:
        torch.tensor: Attenuation map in units of 1/cm
    """
    ds_NM = pydicom.dcmread(file_NM)
    window_upper = (
        ds_NM.EnergyWindowInformationSequence[index_peak]
        .EnergyWindowRangeSequence[0]
        .EnergyWindowUpperLimit
    )
    window_lower = (
        ds_NM.EnergyWindowInformationSequence[index_peak]
        .EnergyWindowRangeSequence[0]
        .EnergyWindowLowerLimit
    )
    if E_SPECT is None:
        E_SPECT = (window_lower + window_upper) / 2 # assumes in center
    if technique=='from_table':
        HU2mu_conversion = get_HU2mu_conversion(files_CT, E_SPECT)
        return HU2mu_conversion(CT)
    elif technique=='from_cortical_bone_fit':
        KVP = pydicom.dcmread(files_CT[0]).KVP
        HU2mu_conversion = get_HU2mu_conversion_old(files_CT, KVP, E_SPECT)
        return HU2mu_conversion(CT)
    elif callable(technique):
        return technique(CT)
    else:
        print('Invalid technique')


def bilinear_transform(
    HU: float,
    a1: float,
    a2: float,
    b1: float,
    b2: float
    ) -> float:
    r"""Function used to convert between Hounsfield Units at an effective CT energy and linear attenuation coefficient at a given SPECT radionuclide energy. It consists of two distinct linear curves in regions :math:`HU<0` and :math:`HU \geq 0`.

    Args:
        HU (float): Hounsfield units at CT energy
        a1 (float): Fit parameter 1
        a2 (float): Fit parameter 2
        b1 (float): Fit parameter 3
        b2 (float): Fit parameter 4

    Returns:
        float: Linear attenuation coefficient at SPECT energy
    """
    output =  np.piecewise(
        HU,
        [HU < 0, HU >= 0],
        [lambda x: a1*x + b1,
        lambda x: a2*x + b2])
    output[output<0] = 0
    return output

def get_HU2mu_conversion(
    files_CT: Sequence[str],
    E_SPECT: float
    ) -> function:
    """Obtains the HU to mu conversion function that converts CT data to the required linear attenuation value in units of 1/cm required for attenuation correction in SPECT/PET imaging.

    Args:
        files_CT (Sequence[str]): CT data files
        CT_kvp (float): kVp value for CT scan
        E_SPECT (float): Energy of photopeak in SPECT scan

    Returns:
        function: Conversion function from HU to mu.
    """
    module_path = os.path.dirname(os.path.abspath(__file__))
    E_CT = pydicom.dcmread(files_CT[0]).KVP
    model_name = pydicom.dcmread(files_CT[0]).ManufacturerModelName.replace(' ','').replace('-', '').lower()
    df = pd.read_csv(os.path.join(module_path, f'../../data/ct_table.csv'))
    df['Model Name'] = df['Model Name'].apply(lambda s: s.replace(' ','').replace('-', ''))
    if model_name in df['Model Name'].values:
        df = df[df['Model Name'] == model_name]
    else:
        Warning('Scanner model not found in database. Using scanner with similar parameters instead')
    df['abs_diff_1'] = (df['Energy'] - E_SPECT).abs()
    df['abs_diff_2'] = (df['CT Energy'] - E_CT).abs()
    df_sorted = df.sort_values(by=['abs_diff_1', 'abs_diff_2'])
    print(f'Given photopeak energy {E_SPECT} keV and CT energy {E_CT} keV from the CT DICOM header, the HU->mu conversion from the following configuration is used: {df_sorted.iloc[0].Energy} keV SPECT energy, {df_sorted.iloc[0]["CT Energy"]} keV CT energy, and scanner model {df_sorted.iloc[0]["Model Name"]}')
    a1opt, b1opt, a2opt, b2opt = df_sorted.iloc[0,[3,4,5,6]].values
    return partial(bilinear_transform, a1=a1opt, a2=a2opt, b1=b1opt, b2=b2opt)

def get_attenuation_map_from_CT_slices(
    files_CT: Sequence[str],
    file_NM: str | None = None,
    index_peak: int = 0,
    mode: str = "constant",
    HU2mu_technique: str | Callable = "from_table",
    E_SPECT: float | None = None,
    output_shape = None,
) -> torch.Tensor:
    """Converts a sequence of DICOM CT files (corresponding to a single scan) into a torch.Tensor object usable as an attenuation map in PyTomography.

    Args:
        files_CT (Sequence[str]): List of all files corresponding to an individual CT scan
        file_NM (str): File corresponding to raw PET/SPECT data (required to align CT with projections). If None, then no alignment is done. Defaults to None.
        index_peak (int, optional): Index corresponding to photopeak in projection data. Defaults to 0.
        mode (str): Mode for affine transformation interpolation
        HU2mu_technique (str): Technique to convert HU to attenuation coefficients. The default, 'from_table', uses a table of coefficients for bilinear curves obtained for a variety of common radionuclides. The technique 'from_cortical_bone_fit' looks for a cortical bone peak in the scan and uses that to obtain the bilinear coefficients. For phantom scans where the attenuation coefficient is always significantly less than bone, the cortical bone technique will still work, since the first part of the bilinear curve (in the air to water range) does not depend on the cortical bone fit. Alternatively, one can provide an arbitrary function here which takes in a 3D scan with units of HU and converts to mu.
        E_SPECT (float): Energy of the photopeak in SPECT scan; this overrides the energy in the DICOM file, so should only be used if the DICOM file is incorrect. Defaults to None.
        output_shape (tuple): Shape of the output attenuation map. If None, then the shape is determined by the NM file.

    Returns:
        torch.Tensor: Tensor of shape [Lx, Ly, Lz] corresponding to attenuation map.
    """

    CT = open_multifile(files_CT).cpu().numpy()
    CT = CT_to_mumap(CT, files_CT, file_NM, index_peak, technique=HU2mu_technique, E_SPECT=E_SPECT)
    # Get affine matrix for alignment:
    M_CT = _get_affine_multifile(files_CT)
    M_NM = _get_affine_spect_projections(file_NM)
    M = npl.inv(M_CT) @ M_NM
    # Apply affine
    ds_NM = pydicom.dcmread(file_NM)
    if output_shape is None:
        output_shape = (ds_NM.Rows, ds_NM.Rows, ds_NM.Columns)
    CT = affine_transform(
        CT, M, output_shape=output_shape, mode=mode, cval=0, order=1
    )
    CT = torch.tensor(CT).to(pytomography.dtype).to(pytomography.device)
    return CT

def _get_affine_spect_projections(filename: str) -> np.array:
    """Computes an affine matrix corresponding the coordinate system of a SPECT DICOM file of projections.

    Args:
        ds (Dataset): DICOM dataset of projection data

    Returns:
        np.array: Affine matrix
    """
    # Note: per DICOM convention z actually decreases as the z-index increases (initial z slices start with the head)
    ds = pydicom.dcmread(filename)
    Sx, Sy, Sz = ds.DetectorInformationSequence[0].ImagePositionPatient
    dx = dy = ds.PixelSpacing[0]
    dz = float(ds.PixelSpacing[1])
    if Sy == 0:
        Sx -= (ds.Rows-1) / 2 * dx
        Sy -= (ds.Rows-2) / 2 * dy
        Sy -= ds.RotationInformationSequence[0].TableHeight
    Sz -= (ds.Rows-1) * dz # location of bottom pixel
    # Difference between Siemens and GE
    # if ds.Manufacturer=='GE MEDICAL SYSTEMS':
    #Sz -= ds.RotationInformationSequence[0].TableTraverse
    M = np.zeros((4, 4))
    M[0] = np.array([dx, 0, 0, Sx])
    M[1] = np.array([0, dy, 0, Sy])
    M[2] = np.array([0, 0, dz, Sz])
    M[3] = np.array([0, 0, 0, 1])
    return M

def load_multibed_projections(
    files_NM: str,
) -> torch.Tensor:
    """This function loads projection data from each of the files in files_NM; for locations outside the FOV in each projection, it appends the data from the adjacent projection (it uses the midway point between the projection overlap).

    Args:
        files_NM (str): Filespaths for each of the projections

    Returns:
        torch.Tensor: Tensor of shape ``[N_bed_positions, N_energy_windows, Ltheta, Lr, Lz]``.
    """
    projectionss = torch.stack([get_projections(file_NM) for file_NM in files_NM])
    dss = np.array([pydicom.dcmread(file_NM) for file_NM in files_NM])
    zs = torch.tensor(
        [ds.DetectorInformationSequence[0].ImagePositionPatient[-1] for ds in dss]
    )
    # Sort by increasing z-position
    order = torch.argsort(zs)
    dss = dss[order.cpu().numpy()]
    zs = zs[order]
    zs = torch.round((zs - zs[0]) / dss[0].PixelSpacing[1]).to(torch.long)
    projectionss = projectionss[order]
    z_voxels = projectionss[0].shape[-1]
    projectionss_combined = torch.stack([p for p in projectionss])
    for i in range(len(projectionss)):
        if i>0: # Set lower part
            dz = zs[i] - zs[i-1]
            index_midway = int((z_voxels - dz)/2)
            # Assumes the projections overlap slightly
            projectionss_combined[i][...,:index_midway] = projectionss[i-1][...,dz:dz+index_midway]
        if i<len(projectionss)-1: # Set upper part
            dz = zs[i+1] - zs[i]
            index_midway = int((z_voxels - dz)/2)
            # Assumes the projections overlap slightly
            projectionss_combined[i][...,dz+index_midway:] = projectionss[i+1][...,index_midway:z_voxels-dz]
    # Return back in original order of files_NM
    return projectionss_combined[torch.argsort(order)]

def stitch_multibed(
    recons: torch.Tensor,
    files_NM: Sequence[str],
    return_stitching_weights: bool = False
) -> torch.Tensor:
    """Stitches together multiple reconstructed objects corresponding to different bed positions.

    Args:
        recons (torch.Tensor[n_beds, Lx, Ly, Lz]): Reconstructed objects. The first index of the tensor corresponds to different bed positions
        files_NM (list): List of length ``n_beds`` corresponding to the DICOM file of each reconstruction
        return_stitching_weights (bool): If true, instead of returning stitched reconstruction, instead returns the stitching weights (and z location in the stitched image) for each bed position (this is used as a tool for uncertainty estimation in multi bed positions). Defaults to False

    Returns:
        torch.Tensor[Lx, Ly, Lz']: Stitched together DICOM file. Note the new z-dimension size :math:`L_z'`.
    """
    dss = np.array([pydicom.dcmread(file_NM) for file_NM in files_NM])
    zs = np.array(
        [ds.DetectorInformationSequence[0].ImagePositionPatient[-1] for ds in dss]
    )
    # Sort by increasing z-position
    order = np.argsort(zs)
    dss = dss[order]
    zs = zs[order]
    recons = recons[order]
    # convert to voxel height
    zs = np.round((zs - zs[0]) / dss[0].PixelSpacing[1]).astype(int)
    original_z_height = recons.shape[-1]
    new_z_height = zs[-1] + original_z_height
    recon_aligned = torch.zeros((dss[0].Rows, dss[0].Rows, new_z_height)).to(
        pytomography.device
    )
    blank_below, blank_above = 1, recons.shape[-1]-1
    # Apply stitching method
    stitching_weights = []
    for i in range(len(recons)):
        stitching_weights_i = torch.zeros(recons.shape[1:]).to(pytomography.device)
        stitching_weights_i[:,:,blank_below:blank_above] = 1
        stitching_weights.append(stitching_weights_i)
    for i in range(len(recons)):
        # stitching from above
        if i!=len(recons)-1:
            overlap_lower = zs[i+1] - zs[i] + blank_below
            overlap_upper = blank_above
            delta = overlap_upper - overlap_lower
            # Only offer midslice stitch now because TEM messes with uncertainty estimation
            half = round(delta / 2)
            stitching_weights[i][:,:,overlap_lower+half:overlap_lower+delta] = 0
            stitching_weights[i+1][:,:,blank_below:blank_below+half] = 0
    for i in range(len(zs)):
        recon_aligned[:, :, zs[i] : zs[i] + original_z_height] += recons[i]  * stitching_weights[i]
    if return_stitching_weights:
        # put back in original order
        return torch.cat(stitching_weights)[np.argsort(order)], zs[np.argsort(order)]
    else:
        return recon_aligned

def get_aligned_rtstruct(
    file_RT: str,
    file_NM: str,
    dicom_series_path: str,
    rt_struct_name: str,
    cutoff_value = 0.5,
    shape = None
):
    """Loads an RT struct file and aligns it with SPECT projection data corresponding to ``file_NM``. 

    Args:
        file_RT (str): Filepath of the RT Struct file
        file_NM (str): Filepath of the NM file (used to align the RT struct)
        dicom_series_path (str): Filepath of the DICOM series linked to the RTStruct file (required for loading RTStructs).
        rt_struct_name (str): Name of the desired RT struct.
        cutoff_value (float, optional): After interpolation is performed to align the mask in the new frame, mask voxels with values less than this are excluded. Defaults to 0.5.

    Returns:
        torch.Tensor: RTStruct mask aligned with SPECT data.
    """
    if shape is None:
        object_meta, _ = get_metadata(file_NM)
        shape = object_meta.shape
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=dicom_series_path, 
        rt_struct_path=file_RT
    )
    files_CT = [os.path.join(dicom_series_path, file) for file in os.listdir(dicom_series_path)]
    mask = rtstruct.get_roi_mask_by_name(rt_struct_name).astype(float)
    M_CT = _get_affine_multifile(files_CT)
    M_NM = _get_affine_spect_projections(file_NM)
    M = npl.inv(M_CT) @ M_NM
    mask_aligned = affine_transform(mask.transpose((1,0,2)), M, output_shape=shape, mode='constant', cval=0, order=1)
    if cutoff_value is None:
        return torch.tensor(mask_aligned.copy()).to(pytomography.device)
    else:
        return torch.tensor(mask_aligned>cutoff_value).to(pytomography.device)

def get_aligned_nifti_mask(
    file_nifti: str,
    file_NM: str,
    dicom_series_path: str,
    mask_idx: float,
    cutoff_value = 0.5,
    shape = None
):
    """Loads an RT struct file and aligns it with SPECT projection data corresponding to ``file_NM``. 

    Args:
        file_nifti (str): Filepath of the nifti file containing the reconstruction mask
        file_NM (str): Filepath of the NM file (used to align the RT struct)
        dicom_series_path (str): Filepath of the DICOM series linked to the RTStruct file (required for loading RTStructs).
        mask_idx (str): Integer in nifti mask corresponding to ROI.
        cutoff_value (float, optional): After interpolation is performed to align the mask in the new frame, mask voxels with values less than this are excluded. Defaults to 0.5.

    Returns:
        torch.Tensor: RTStruct mask aligned with SPECT data.
    """
    if shape is None:
        object_meta, _ = get_metadata(file_NM)
        shape = object_meta.shape
    mask = (nib.load(file_nifti).get_fdata().transpose((1,0,2))[::-1]==mask_idx).astype(float)
    files_CT = [os.path.join(dicom_series_path, file) for file in os.listdir(dicom_series_path)]
    M_CT = _get_affine_multifile(files_CT)
    M_NM = _get_affine_spect_projections(file_NM)
    M = npl.inv(M_CT) @ M_NM
    mask_aligned = affine_transform(mask.transpose((1,0,2))[:,:,::-1], M, output_shape=shape, mode='constant', cval=0, order=1)[:,:,::-1]
    return torch.tensor(mask_aligned>cutoff_value).to(pytomography.device)

def get_FOV_mask_from_projections(file_NM, contraction=1):
    projections = get_projections(file_NM)
    dims = len(projections.shape)
    x = projections.sum(dim=tuple([i for i in range(dims-2)]))
    r_valid = (x.sum(dim=1)>0).to(torch.int)
    z_valid = (x.sum(dim=0)>0).to(torch.int)
    r_min = r_valid.argmax()
    r_max = r_valid.shape[0] - r_valid.flip(dims=(0,)).argmax() - 1
    z_min = z_valid.argmax()
    z_max = z_valid.shape[0] - z_valid.flip(dims=(0,)).argmax() - 1
    # Adjust to ignore outer boundaries in case less sensitivity
    r_min +=contraction; z_min +=contraction; r_max -=contraction; z_max -=contraction
    blank_mask = torch.zeros_like(x)
    blank_mask[r_min:r_max+1, z_min:z_max+1] = 1
    return blank_mask

def get_mean_stray_radiation_counts(file_blank, file_NM, index_peak=None):
    ds_blank = pydicom.dcmread(file_blank)
    # Get acquisition times
    ds_NM = pydicom.dcmread(file_NM)
    dT_blank = ds_blank.RotationInformationSequence[0].ActualFrameDuration / 1000
    dT_NM = ds_NM.RotationInformationSequence[0].ActualFrameDuration / 1000
    # Get mask
    projections_blank = get_projections(file_blank)
    blank_mask = get_FOV_mask_from_projections(file_blank)
    N_angles = projections_blank.shape[-3]
    # Get mean stray radiation counts
    mean_stray_counts = (projections_blank*blank_mask).sum(dim=(-3,-2,-1)) / (N_angles * blank_mask.sum()) * dT_NM / dT_blank
    if index_peak is None:
        return mean_stray_counts
    else:
        return mean_stray_counts[index_peak].item()

def get_mean_stray_radiation_counts_MEW_scatter(file_blank, file_NM, index_peak, index_lower, index_upper=None, weighting_lower=0.5, weighting_upper=0.5):
    mean_stray_counts = get_mean_stray_radiation_counts(file_blank, file_NM)
    ds_NM = pydicom.dcmread(file_NM)
    stray_lower = mean_stray_counts[index_lower]
    stray_upper = mean_stray_counts[index_upper] if index_upper is not None else 0
    width_peak = get_window_width(ds_NM, index_peak)
    width_lower = get_window_width(ds_NM, index_lower)
    width_upper = get_window_width(ds_NM, index_upper) if index_upper is not None else None
    return compute_EW_scatter(stray_lower, stray_upper, width_lower, width_upper, width_peak, weighting_lower, weighting_upper).item()
        
def save_dcm(
    save_path: str,
    object: torch.Tensor,
    file_NM: str,
    recon_name: str = 'pytomo_recon',
    return_ds: bool = False,
    single_dicom_file: bool = False,
    scale_by_number_projections: bool = False
    ) -> None:
    """Saves the reconstructed object `object` to a series of DICOM files in the folder given by `save_path`. Requires the filepath of the projection data `file_NM` to get Study information.

    Args:
        object (torch.Tensor): Reconstructed object of shape [Lx,Ly,Lz].
        save_path (str): Location of folder where to save the DICOM output files.
        file_NM (str): File path of the projection data corresponding to the reconstruction.
        recon_name (str): Type of reconstruction performed. Obtained from the `recon_method_str` attribute of a reconstruction algorithm class.
        return_ds (bool): If true, returns the DICOM dataset objects instead of saving to file. Defaults to False.
    """
    if not return_ds:
        try:
            Path(save_path).resolve().mkdir(parents=True, exist_ok=False)
        except:
            raise Exception(
                f"Folder {save_path} already exists; new folder name is required."
            )
    # Convert tensor image to numpy array
    ds_NM = pydicom.dcmread(file_NM)
    SOP_instance_UID = generate_uid()
    if single_dicom_file:
        SOP_class_UID = '1.2.840.10008.5.1.4.1.1.20'
        modality = 'NM'
        imagetype = "['ORIGINAL', 'PRIMARY', 'RECON TOMO', 'EMISSION']"
    else:
        SOP_class_UID = "1.2.840.10008.5.1.4.1.1.128"  # SPECT storage
        modality = 'PT'
        imagetype = None
    ds = create_ds(ds_NM, SOP_instance_UID, SOP_class_UID, modality, imagetype)
    pixel_data = torch.permute(object,(2,1,0)).cpu().numpy()
    if scale_by_number_projections:
        scale_factor = get_metadata(file_NM)[1].num_projections
        ds.RescaleSlope = 1
    else:
        scale_factor = (2**16 - 1) / pixel_data.max()
        ds.RescaleSlope = 1/scale_factor
    pixel_data *= scale_factor #maximum dynamic range
    pixel_data = pixel_data.round().astype(np.uint16)
    # Affine
    Sx, Sy, Sz = ds_NM.DetectorInformationSequence[0].ImagePositionPatient
    dx = dy = ds_NM.PixelSpacing[0]
    dz = ds_NM.PixelSpacing[1]
    if Sy == 0:
        Sx -= (ds_NM.Rows-1) / 2 * dx
        Sy -= (ds_NM.Rows-1) / 2 * dy
        # Y-Origin point at tableheight=0
        Sy -= ds_NM.RotationInformationSequence[0].TableHeight
    # Sz now refers to location of lowest slice
    Sz -= (pixel_data.shape[0] - 1) * dz
    ds.Rows, ds.Columns = pixel_data.shape[1:]
    ds.SeriesNumber = 1
    if single_dicom_file:
        ds.NumberOfFrames = pixel_data.shape[0]
    else:
        ds.NumberOfSlices = pixel_data.shape[0]
    ds.PixelSpacing = [dx, dy]
    ds.SliceThickness = dz
    ds.SpacingBetweenSlices = dz
    ds.ImageOrientationPatient = [1,0,0,0,1,0]
    # Set other things
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.ReconstructionMethod = recon_name
    if single_dicom_file:
        ds.InstanceNumber = 1
        ds.ImagePositionPatient = [Sx, Sy, Sz]
        ds.PixelData = pixel_data.tobytes()
    # Add all study data/time information if available
    for attr in ['StudyDate', 'StudyTime', 'SeriesDate', 'SeriesTime', 'AcquisitionDate', 'AcquisitionTime', 'ContentDate', 'ContentTime', 'PatientSex', 'PatientAge', 'Manufacturer', 'PatientWeight', 'PatientHeight']:
        if hasattr(ds_NM, attr):
            ds[attr] = ds_NM[attr]
    ds.SeriesDescription = f'{ds_NM.SeriesDescription}: {recon_name}'
    # Create all slices
    if not single_dicom_file:
        dss = []
        for i in range(pixel_data.shape[0]):
            # Load existing DICOM file
            ds_i = copy.deepcopy(ds)
            ds_i.InstanceNumber = i + 1
            ds_i.ImagePositionPatient = [Sx, Sy, Sz + i * dz]
            # Create SOP Instance UID unique to slice
            ds_i.SOPInstanceUID = f"{ds.SOPInstanceUID[:-3]}{i+1:03d}"
            ds_i.file_meta.MediaStorageSOPInstanceUID = ds_i.SOPInstanceUID
            # Set the pixel data
            ds_i.PixelData = pixel_data[i].tobytes()
            dss.append(ds_i)      
    if return_ds:
        if single_dicom_file:
            return ds
        else:
            return dss
    else:
        if single_dicom_file:
            # If single dicom file, will overwrite any file that is there
            ds.save_as(os.path.join(save_path, f'{ds.SOPInstanceUID}.dcm'))
        else:
            for ds_i in dss:
                ds_i.save_as(os.path.join(save_path, f'{ds_i.SOPInstanceUID}.dcm'))
                   
# ---------------------------------------
# Imaging System Specific Functions
# ---------------------------------------

def get_starguide_projections(files_NM: Sequence[str], index_peak: int | None = None):
    """Obtain projections from the sequence of files corresponding to a single starguide acquisition; there should be 12 files (one for each head position). The projections are sorted by energy window.

    Args:
        files_NM (Sequence[str]): Sequence of files corresponding to the acquisition (one file for each head)
        index_peak (int | None, optional): Photopeak index; if None then returns all energy peaks. Defaults to None.

    Returns:
        torch.Tensor: StarGuide projeciton data
    """
    projections = []
    energy_window_vector = []
    for file in files_NM:
        try:
            ds = pydicom.dcmread(file)
            energy_window_vector += ds.EnergyWindowVector
            projections += list(ds.pixel_array * ds[0x0011, 0x103b].value)
        except:
            continue
    energy_window_vector = torch.tensor(energy_window_vector)
    unique_idxs = torch.unique(energy_window_vector)
    projections_all = []
    for idx in unique_idxs:
        idx = energy_window_vector==idx
        projections_all.append(torch.tensor(projections)[idx].swapaxes(1,2).to(pytomography.dtype).to(pytomography.device))
    if index_peak is not None:
        return projections_all[index_peak]
    else:
        return torch.stack(projections_all)
    
def get_starguide_metadata(files_NM: Sequence[str], index_peak: int = 0, nearest_theta: float = 1.0):
    """Obtains the metadata for a Starguide SPECT acquisition.

    Args:
        files_NM (Sequence[str]): Sequence of NM files for a StarGuide acqusition
        index_peak (int, optional): Photopeak index for reconstruction. Defaults to 0.
        nearest_theta (float, optional): Nearest theta to round angles to. Defaults to 1.0.

    Returns:
        Sequence: Object meta and projection data for the acquisition.
    """
    angles = []
    radii = []
    offsets = []
    times = []
    energy_window_vector = []
    for i, file in enumerate(files_NM):
        try:
            ds = pydicom.dcmread(file)
            t = np.array(ds[0x0009,0x1003].value)
            x = np.array(ds[0x0099,0x01051].value)
            y = np.array(ds[0x0099,0x01052].value)
            thetas = np.array(ds[0x0099,0x01053].value)
            r = x*np.sin(thetas*np.pi/180) + y * np.cos(thetas*np.pi/180)
            l = x*np.cos(thetas*np.pi/180) - y * np.sin(thetas*np.pi/180)
            angle = np.round(thetas/nearest_theta) * nearest_theta # round to nearest degrree if nearest_theta=1.0
            angles += list(angle)
            radii += list(r)
            offsets += list(l)
            times += list(t)
            energy_window_vector += ds.EnergyWindowVector
        except:
            print(f'File at index {i} failed')
            continue
    idx = torch.tensor(energy_window_vector)== index_peak + 1
    radii = np.array(radii)[idx]
    offsets = torch.tensor(offsets)[idx].to(pytomography.dtype).to(pytomography.device)
    times = torch.tensor(times)[idx].to(pytomography.dtype).to(pytomography.device)
    angles = torch.tensor(angles)[idx].to(pytomography.dtype).to(pytomography.device)
    projections = get_starguide_projections(files_NM, index_peak)
    dx = ds.PixelSpacing[0] / 10 # to cm
    proj_meta = StarGuideProjMeta(projections.shape, angles, times, offsets, radii)
    object_meta = SPECTObjectMeta(dr=(dx, dx, dx), shape=(196,196,112)) # 196 is what GE uses
    return object_meta, proj_meta

def get_starguide_affine_CT(files_CT: Sequence[str]):
    """Obtain the affine matrix for a Starguide CT acquisition.

    Args:
        files_CT (Sequence[str]): Files corresponding to the CT acquisition

    Returns:
        np.array: Affine matrix for the CT acquisition
    """
    ds = pydicom.dcmread(files_CT[0])
    dx = dy = ds.PixelSpacing[0] / 10
    dz = ds.SliceThickness / 10
    shape = [*ds.pixel_array.shape, len(files_CT)]
    Sx_CT = - (shape[0]-1) * dx / 2
    Sy_CT = - (shape[1]-1) * dy / 2
    Sz_CT = - (shape[2]-1) * dz / 2
    affine_CT = np.array([[dx, 0, 0, Sx_CT],
                        [0, dy, 0, Sy_CT],
                        [0, 0, dz, Sz_CT],
                        [0, 0, 0, 1]])
    return affine_CT

def get_starguide_affine_NM(files_NM: Sequence[str]):
    """Obtain the affine matrix for a Starguide NM acquisition.

    Args:
        files_NM (Sequence[str]): Files corresponding to the NM acquisition

    Returns:
        np.array: Affine matrix for the NM acquisition
    """
    object_meta, _ = get_starguide_metadata(files_NM)
    dx_NM = dy_NM = dz_NM = object_meta.dr[0]
    Sx_NM = - (object_meta.shape[0]-1) * dx_NM / 2
    Sy_NM = - (object_meta.shape[1]-1) * dy_NM / 2
    Sz_NM = - (object_meta.shape[2]-1) * dz_NM / 2
    affine_NM = np.array([[dx_NM, 0,0,Sx_NM],
                         [0, dy_NM, 0, Sy_NM],
                         [0, 0, dz_NM, Sz_NM],
                         [0, 0, 0, 1]])
    return affine_NM

def get_starguide_attenuation_map_from_CT_slices(
    files_CT: Sequence[str],
    files_NM: Sequence[str],
    index_peak: int = 0,
    mode: str = "constant",
    E_SPECT: float | None = None,
):  
    """Obtain the attenuation map for a Starguide SPECT acquisition from a sequence of CT files.

    Args:
        files_CT (Sequence[str]): CT files corresponding to the acquisition
        files_NM (Sequence[str]): NM files corresponding to the acquisition
        index_peak (int, optional): Index corresponding to photopeak. Defaults to 0.
        mode (str, optional): Mode for the affine matrix. Defaults to "constant".
        E_SPECT (float | None, optional): Energy of SPECT; this overrights the energy from index_peak if provided. Defaults to None.

    Returns:
       torch.Tensor: Attenuation map in units of 1/cm
    """
    object_meta, _ = get_starguide_metadata(files_NM, index_peak)
    CT = open_multifile(files_CT).cpu().numpy()
    CT = CT_to_mumap(CT, files_CT, files_NM[0], index_peak=index_peak, technique='from_cortical_bone_fit', E_SPECT=E_SPECT)
    affine_CT = get_starguide_affine_CT(files_CT)
    affine_NM = get_starguide_affine_NM(files_NM)
    M = npl.inv(affine_CT) @ affine_NM
    CT = affine_transform(
            CT, M, output_shape=object_meta.shape, mode=mode, cval=0, order=1
    )
    CT = torch.tensor(CT).to(pytomography.dtype).to(pytomography.device)
    CT = torch.flip(CT, [2])
    return CT