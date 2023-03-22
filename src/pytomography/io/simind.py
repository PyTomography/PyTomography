from __future__ import annotations
from typing import Sequence
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import os
from pytomography.metadata import ObjectMeta, ImageMeta
from pytomography.projections import ForwardProjectionNet, BackProjectionNet
from pytomography.mappings import SPECTAttenuationNet, SPECTPSFNet
from pytomography.priors import Prior
from pytomography.callbacks import CallBack
from pytomography.metadata import PSFMeta
from pytomography.algorithms import OSEMOSL

relation_dict = {'unsignedinteger': 'int',
                 'shortfloat': 'float',
                 'int': 'int'}

def find_first_entry_containing_header(
    list_of_attributes: list[str],
    header: str,
    dtype: type = np.float32
    ) -> float|str|int:
    """Finds the first entry in a SIMIND Interfile output corresponding to the header (header).

    Args:
        list_of_attributes (list[str]): Simind data file, as a list of lines.
        header (str): The header looked for
        dtype (type, optional): The data type to be returned corresponding to the value of the header. Defaults to np.float32.

    Returns:
        float|str|int: The value corresponding to the header (header).
    """
    line = list_of_attributes[np.char.find(list_of_attributes, header)>=0][0]
    if dtype == np.float32:
        return np.float32(line.replace('\n', '').split(':=')[-1])
    elif dtype == str:
        return (line.replace('\n', '').split(':=')[-1].replace(' ', ''))
    elif dtype == int:
        return int(line.replace('\n', '').split(':=')[-1].replace(' ', ''))

def simind_projections_to_data(headerfile: str, distance: str = 'cm'):
    """Obtains ObjectMeta, ImageMeta, and projections from a SIMIND header file.

    Args:
        headerfile (str): Path to the header file
        distance (str, optional): The units of measurements in the SIMIND file (this is required as input, since SIMIND uses mm/cm but doesn't specify). Defaults to 'cm'.

    Returns:
        (ObjectMeta, ImageMeta, torch.Tensor[1, Ltheta, Lr, Lz]): Required information for reconstruction in PyTomography.
    """
    if distance=='mm':
        scale_factor = 1/10
    elif distance=='cm':
        scale_factor = 1    
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    num_proj = find_first_entry_containing_header(headerdata, 'total number of images', int)
    proj_dim1 = find_first_entry_containing_header(headerdata, 'matrix size [1]', int)
    proj_dim2 = find_first_entry_containing_header(headerdata, 'matrix size [2]', int)
    dx = find_first_entry_containing_header(headerdata, 'scaling factor (mm/pixel) [1]', np.float32) / 10 # to mm
    dz = find_first_entry_containing_header(headerdata, 'scaling factor (mm/pixel) [2]', np.float32) / 10 # to mm
    dr = (dx, dx, dz)
    number_format = find_first_entry_containing_header(headerdata, 'number format', str)
    number_format= relation_dict[number_format]
    num_bytes_per_pixel = find_first_entry_containing_header(headerdata, 'number of bytes per pixel', int)
    extent_of_rotation = find_first_entry_containing_header(headerdata, 'extent of rotation', np.float32)
    number_of_projections = find_first_entry_containing_header(headerdata, 'number of projections', int)
    start_angle = find_first_entry_containing_header(headerdata, 'start angle', np.float32)
    angles = np.linspace(start_angle, extent_of_rotation, number_of_projections, endpoint=False)
    radius = find_first_entry_containing_header(headerdata, 'Radius', np.float32) *scale_factor
    imagefile = find_first_entry_containing_header(headerdata, 'name of data file', str)
    shape_proj= (num_proj, proj_dim1, proj_dim2)
    shape_obj = (proj_dim1, proj_dim1, proj_dim2)
    object_meta = ObjectMeta(dr,shape_obj)
    image_meta = ImageMeta(object_meta, angles, np.ones(len(angles))*radius)
    dtype = eval(f'np.{number_format}{num_bytes_per_pixel*8}')
    projections = np.fromfile(os.path.join(str(Path(headerfile).parent), imagefile), dtype=dtype)
    projections = np.transpose(projections.reshape((num_proj,proj_dim2,proj_dim1))[:,::-1], (0,2,1))
    projections = torch.tensor(projections.copy()).unsqueeze(dim=0)
    return object_meta, image_meta, projections

def simind_MEW_to_data(headerfiles: list[str], distance: str = 'cm'):
    """Opens multiple projection files corresponding to the primary, lower scatter, and upper scatter windows

    Args:
        headerfiles (list[str]): List of file paths to required files. Must be in order of: 1. Primary, 2. Lower Scatter, 3. Upper scatter
        distance (str, optional): The units of measurements in the SIMIND file (this is required as input, since SIMIND uses mm/cm but doesn't specify). Defaults to 'cm'.

    Returns:
        (ObjectMeta, ImageMeta, torch.Tensor[1, Ltheta, Lr, Lz], torch.Tensor[1, Ltheta, Lr, Lz]): Required information for reconstruction in PyTomography. First returned tensor contains primary data, and second returned tensor returns estimated scatter using the triple energy window method.
    """
    # assumes all three energy windows have same metadata
    projectionss = []
    window_widths = []
    for headerfile in headerfiles:
        object_meta, image_meta, projections = simind_projections_to_data(headerfile, distance)
        with open(headerfile) as f:
            headerdata = f.readlines()
        headerdata = np.array(headerdata)
        lwr_window = find_first_entry_containing_header(headerdata, 'energy window lower level', np.float32)
        upr_window = find_first_entry_containing_header(headerdata, 'energy window upper level', np.float32)
        window_widths.append(upr_window - lwr_window)
        projectionss.append(projections)
    projections_scatter = (projectionss[1]/window_widths[1] + projectionss[2]/window_widths[2])* window_widths[0] / 2
    return object_meta, image_meta, projectionss[0], projections_scatter

def simind_CT_to_data(headerfile: str):
    """Opens attenuation data from SIMIND output

    Args:
        headerfile (str): Path to header file

    Returns:
        torch.tensor[Lx,Ly,Lz]: Tensor containing CT data.
    """
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    matrix_size_1 = find_first_entry_containing_header(headerdata, 'matrix size [1]', int)
    matrix_size_2 = find_first_entry_containing_header(headerdata, 'matrix size [2]', int)
    matrix_size_3 = find_first_entry_containing_header(headerdata, 'matrix size [3]', int)
    shape = (matrix_size_3, matrix_size_2, matrix_size_1)
    imagefile = find_first_entry_containing_header(headerdata, 'name of data file', str)
    CT = np.fromfile(os.path.join(str(Path(headerfile).parent), imagefile), dtype=np.float32)
    CT = np.transpose(CT.reshape(shape)[::-1,::-1], (2,1,0))
    CT = torch.tensor(CT.copy())
    return CT

def get_SPECT_recon_algorithm_simind(
    projections_header: str,
    scatter_headers: Sequence[str] | None = None,
    CT_header: str = None,
    psf_meta: PSFMeta = None,
    prior: Prior = None,
    object_initial: torch.Tensor | None = None,
    recon_algorithm_class: nn.Module = OSEMOSL,
    device: str = 'cpu'
) -> nn.Module:
    
    if scatter_headers is None:
        object_meta, image_meta, projections = simind_projections_to_data(projections_header)
        projections_scatter = 0 # equivalent to 0 estimated scatter everywhere
    else:
        object_meta, image_meta, projections, projections_scatter = simind_MEW_to_data([projections_header, *scatter_headers])
        projections_scatter.to(device)
    projections.to(device)
    if CT_header is not None:
        CT = simind_CT_to_data(CT_header)
    object_correction_nets = []
    image_correction_nets = []
    if CT_header is not None:
        CT_net = SPECTAttenuationNet(CT.unsqueeze(dim=0).to(device), device=device)
        object_correction_nets.append(CT_net)
    if psf_meta is not None:
        psf_net = SPECTPSFNet(psf_meta, device=device)
        object_correction_nets.append(psf_net)
    fp_net = ForwardProjectionNet(object_correction_nets, image_correction_nets, object_meta, image_meta, device=device)
    bp_net = BackProjectionNet(object_correction_nets, image_correction_nets, object_meta, image_meta, device=device)
    if prior is not None:
        prior.set_device(device)
    recon_algorithm = recon_algorithm_class(projections, fp_net, bp_net, object_initial, projections_scatter, prior)
    return recon_algorithm
