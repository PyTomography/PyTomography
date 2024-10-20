from __future__ import annotations
from typing import Sequence
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import os
import pytomography
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta
from pytomography.utils import get_mu_from_spectrum_interp, compute_EW_scatter
from ..shared import get_header_value

relation_dict = {'unsignedinteger': 'int',
                 'shortfloat': 'float',
                 'int': 'int'}

def get_metadata(headerfile: str, distance: str = 'cm', corrfile: str | None = None):
    """Obtains required metadata from a SIMIND header file.

    Args:
        headerfile (str): Path to the header file
        distance (str, optional): The units of measurements in the SIMIND file (this is required as input, since SIMIND uses mm/cm but doesn't specify). Defaults to 'cm'.
        corrfile (str, optional): .cor file used in SIMIND to specify radial positions for non-circular orbits. This needs to be provided for non-standard orbits.

    Returns:
        (SPECTObjectMeta, SPECTProjMeta, torch.Tensor[1, Ltheta, Lr, Lz]): Required information for reconstruction in PyTomography.
    """
    if distance=='mm':
        scale_factor = 1/10
    elif distance=='cm':
        scale_factor = 1    
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    proj_dim1 = get_header_value(headerdata, 'matrix size [1]', int)
    proj_dim2 = get_header_value(headerdata, 'matrix size [2]', int)
    dx = get_header_value(headerdata, 'scaling factor (mm/pixel) [1]', np.float32) / 10 # to mm
    dz = get_header_value(headerdata, 'scaling factor (mm/pixel) [2]', np.float32) / 10 # to mm
    dr = (dx, dx, dz)
    extent_of_rotation = get_header_value(headerdata, 'extent of rotation', np.float32)
    number_of_projections = get_header_value(headerdata, 'number of projections', int)
    start_angle = get_header_value(headerdata, 'start angle', np.float32)
    direction = get_header_value(headerdata, 'direction of rotation', str)
    angles = np.linspace(-start_angle, -start_angle+extent_of_rotation, number_of_projections, endpoint=False)
    if direction=='CW':
        angles = -angles % 360
    # Get radial positions
    if corrfile is not None:
        radii = np.loadtxt(corrfile) * scale_factor
    else:
        radius = get_header_value(headerdata, 'Radius', np.float32) * scale_factor
        radii = np.ones(len(angles))*radius
    shape_obj = (proj_dim1, proj_dim1, proj_dim2)
    object_meta = SPECTObjectMeta(dr,shape_obj)
    proj_meta = SPECTProjMeta((proj_dim1, proj_dim2), (dx,dz), angles, radii)
    return object_meta, proj_meta

def _get_projections_from_single_file(headerfile: str):
    """Gets projection data from a SIMIND header file.

    Args:
        headerfile (str): Path to the header file
        distance (str, optional): The units of measurements in the SIMIND file (this is required as input, since SIMIND uses mm/cm but doesn't specify). Defaults to 'cm'.

    Returns:
        (torch.Tensor[1, Ltheta, Lr, Lz]): Simulated SPECT projection data.
    """
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    num_proj = get_header_value(headerdata, 'total number of images', int)
    proj_dim1 = get_header_value(headerdata, 'matrix size [1]', int)
    proj_dim2 = get_header_value(headerdata, 'matrix size [2]', int)
    number_format = get_header_value(headerdata, 'number format', str)
    number_format= relation_dict[number_format]
    num_bytes_per_pixel = get_header_value(headerdata, 'number of bytes per pixel', int)
    imagefile = get_header_value(headerdata, 'name of data file', str)
    dtype = eval(f'np.{number_format}{num_bytes_per_pixel*8}')
    projections = np.fromfile(os.path.join(str(Path(headerfile).parent), imagefile), dtype=dtype)
    projections = np.transpose(projections.reshape((num_proj,proj_dim2,proj_dim1))[:,::-1], (0,2,1))
    projections = torch.tensor(projections.copy()).to(pytomography.device)
    return projections

def get_projections(headerfiles: str | Sequence[str], weights: float = None):
    """Gets projection data from a SIMIND header file.

    Args:
        headerfile (str): Path to the header file
        distance (str, optional): The units of measurements in the SIMIND file (this is required as input, since SIMIND uses mm/cm but doesn't specify). Defaults to 'cm'.

    Returns:
        (torch.Tensor[1, Ltheta, Lr, Lz]): Simulated SPECT projection data.
    """
    if type(headerfiles) is str:
        return _get_projections_from_single_file(headerfiles)
    elif isinstance(headerfiles[0], Sequence) and not isinstance(headerfiles[0], str):
        projections = []
        for i, headerfiles_window_i in enumerate(headerfiles):
            projections_window_i = []
            for j, headerfiles_window_i_organ_j in enumerate(headerfiles_window_i):
                projections_window_i_organ_j = get_projections(headerfiles_window_i_organ_j)
                if weights is not None: projections_window_i_organ_j *= weights[j]
                projections_window_i.append(projections_window_i_organ_j)
            projections_window_i = torch.sum(torch.stack(projections_window_i), dim=0)
            projections.append(projections_window_i)
        projections = torch.stack(projections)
        return projections.squeeze()
    else:
        projections = []
        for i, headerfile in enumerate(headerfiles):
            projections.append(get_projections(headerfile))
        projections = torch.stack(projections)
    return projections.squeeze()

def get_energy_window_bounds(headerfile: str) -> tuple[float, float]:
    """Computes the lower and upper bounds of the energy window from a SIMIND header file

    Args:
        headerfile (str): SIMIND header file

    Returns:
        tuple[float, float]: Lower and upper energies
    """
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    lwr = get_header_value(headerdata, 'energy window lower level', np.float32)
    upr = get_header_value(headerdata, 'energy window upper level', np.float32)
    return lwr, upr

def get_energy_window_width(headerfile: str) -> float:
    """Computes the energy window width from a SIMIND header file

    Args:
        headerfile (str): Headerfile corresponding to SIMIND data

    Returns:
        float: Energy window width
    """
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    lwr = get_header_value(headerdata, 'energy window lower level', np.float32)
    upr = get_header_value(headerdata, 'energy window upper level', np.float32)
    return upr - lwr

def combine_projection_data(
    headerfiles: Sequence[str],
    weights: Sequence[float]
    ):
    """Takes in a list of SIMIND headerfiles corresponding to different simulated regions and adds the projection data together based on the `weights`.

    Args:
        headerfiles (Sequence[str]): List of filepaths corresponding to the SIMIND header files of different simulated regions
        weights (Sequence[str]): Amount by which to weight each projection relative.

    Returns:
        (SPECTObjectMeta, SPECTProjMeta, torch.Tensor): Returns necessary object/projections metadata along with the projection data
    """
    projections = 0 
    for headerfile, weight in zip(headerfiles, weights):
        projections_i = get_projections(headerfile)
        projections += projections_i * weight
    return projections

def get_attenuation_map(headerfile: str, smi_index_22: int = 3):
    """Opens attenuation data from SIMIND output

    Args:
        headerfile (str): Path to header file
        smi_index_22 (int, optional): Value of provided in the simind simulation tag: " in:x22,<idx>x " where <idx> is 3 (mu) or 5 (mu-castor). You can check what value this is by default (if you did not provide it) by looking at simind.ini in the simind/smc_dir folder. Defaults to 3.

    Returns:
        torch.Tensor[batch_size, Lx, Ly, Lz]: Tensor containing attenuation map required for attenuation correction in SPECT/PET imaging.
    """
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    matrix_size_1 = get_header_value(headerdata, 'matrix size [1]', int)
    matrix_size_2 = get_header_value(headerdata, 'matrix size [2]', int)
    matrix_size_3 = get_header_value(headerdata, 'matrix size [3]', int)
    shape = (matrix_size_3, matrix_size_2, matrix_size_1)
    imagefile = get_header_value(headerdata, 'name of data file', str)
    CT = np.fromfile(os.path.join(str(Path(headerfile).parent), imagefile), dtype=np.float32)
    # Flip "Z" ("X" in SIMIND) b/c "first density image located at +X" according to SIMIND manual
    # Flip "Y" ("Z" in SIMIND) b/c axis convention is opposite for x22,5x (mu-castor format)
    CT = np.transpose(CT.reshape(shape), (2,1,0))
    if smi_index_22==5: # mu-castor format
        CT = CT[:,::-1,::-1]
    elif smi_index_22==3: # mu format
        CT = CT[:,:,::-1]
    CT = torch.tensor(CT.copy())
    return CT.to(pytomography.device)

def get_psfmeta_from_header(headerfile: str, min_sigmas=3):
    """Obtains the SPECTPSFMeta data corresponding to a SIMIND simulation scan from the headerfile

    Args:
        headerfile (str): SIMIND headerfile.

    Returns:
        SPECTPSFMeta: SPECT PSF metadata required for PSF modeling in reconstruction.
    """
    # Newer version of simind have collimator dimensions in mm
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    if float(get_header_value(headerdata, 'program version ', str)[1:2])>=8:
        to_cm = 0.1
    else:
        to_cm = 1
    
    FWHM2sigma = 1/(2*np.sqrt(2*np.log(2)))
    module_path = os.path.dirname(os.path.abspath(__file__))
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    hole_diameter = get_header_value(headerdata, 'Collimator hole diameter', np.float32) * to_cm
    hole_length = get_header_value(headerdata, 'Collimator thickness', np.float32) * to_cm
    energy_keV = get_header_value(headerdata, 'Photon Energy', np.float32)
    lead_attenuation = get_mu_from_spectrum_interp(os.path.join(module_path, '../../data/NIST_attenuation_data/lead.csv'), energy_keV)
    collimator_slope = hole_diameter/(hole_length - (2/lead_attenuation)) * 1/(2*np.sqrt(2*np.log(2)))
    try:
        intrinsic_resolution_140keV = get_header_value(headerdata, 'Intrinsic FWHM for the camera', np.float32) * FWHM2sigma * to_cm
        intrinsic_resolution = intrinsic_resolution_140keV * (140/energy_keV)**0.5
    except:
        intrinsic_resolution = 0
    collimator_intercept = hole_diameter * FWHM2sigma
    sigma_fit = lambda r, a, b, c: np.sqrt((a*r+b)**2+c**2)
    sigma_fit_params = [collimator_slope, collimator_intercept, intrinsic_resolution]
    return SPECTPSFMeta(
        sigma_fit_params=sigma_fit_params,
        sigma_fit=sigma_fit,
        min_sigmas=min_sigmas)