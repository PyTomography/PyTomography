from __future__ import annotations
import pytomography
from pytomography.metadata import ObjectMeta
import torch
import numpy as np
import numpy.linalg as npl
from scipy.ndimage import affine_transform
from ..shared import get_header_value, get_attenuation_map_interfile

def get_aligned_attenuation_map_GATE(
    headerfile: str,
    object_meta: ObjectMeta
    ):
    """Returns an aligned attenuation map in units of inverse mm for use in reconstruction of PETSIRD listmode data.

    Args:
        headerfile (str): Filepath to the header file of the attenuation map
        object_meta (ObjectMeta): Object metadata providing spatial information about the reconstructed dimensions.

    Returns:
        torch.Tensor: Aligned attenuation map
    """
    amap = get_attenuation_map_interfile(headerfile)[0].cpu().numpy()
    # Load metadata
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    dx = get_header_value(headerdata, 'scaling factor (mm/pixel) [1]', np.float32)
    dy = get_header_value(headerdata, 'scaling factor (mm/pixel) [2]', np.float32)
    dz = get_header_value(headerdata, 'scaling factor (mm/pixel) [3]', np.float32)
    dr_amap = (dx, dy, dz)
    shape_amap = amap.shape
    object_origin_amap = (- np.array(shape_amap) / 2 + 0.5) * (np.array(dr_amap))
    dr = object_meta.dr
    shape = object_meta.shape
    object_origin = object_origin = (- np.array(shape) / 2 + 0.5) * (np.array(dr))
    M_PET = np.array([[dr[0],0,0,object_origin[0]],
                 [0,dr[1],0,object_origin[1]],
                 [0,0,dr[2],object_origin[2]],
                 [0,0,0,1]])
    M_CT = np.array([[dr_amap[0],0,0,object_origin_amap[0]],
                 [0,dr_amap[1],0,object_origin_amap[1]],
                 [0,0,dr_amap[2],object_origin_amap[2]],
                 [0,0,0,1]])
    amap = affine_transform(amap, npl.inv(M_CT)@M_PET, output_shape = shape, order=1)
    amap = torch.tensor(amap, device=pytomography.device).unsqueeze(0) / 10
    return amap


def get_scanner_LUT(path, init_volume_name='crystal', final_volume_name='world', mean_interaction_depth=0):
    with open(path) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    # Recursively get names of all volumes
    volume = init_volume_name
    parents = [volume]
    while volume!=final_volume_name:
        print(volume)
        volume = get_header_value(headerdata, f'/daughters/name.*{volume}', split_substr='/', split_idx=2, dtype=str)
        if not(volume):
            return None
        parents.append(volume)
    # Get initial positions of crystal
    positions = []
    for parent in parents:
        try:
            x = get_header_value(headerdata, f'/gate/{parent}/placement/setTranslation', split_substr=None, split_idx=1)
            y = get_header_value(headerdata, f'/gate/{parent}/placement/setTranslation', split_substr=None, split_idx=2)
            z = get_header_value(headerdata, f'/gate/{parent}/placement/setTranslation', split_substr=None, split_idx=3)
        except:
            x = y = z = 0
        positions.append([x,y,z])
    positions = np.array(positions)
    x_crystal, y_crystal, z_crystal = positions.sum(axis=0)
    x_crystal = np.array([x_crystal])
    y_crystal = np.array([y_crystal])
    z_crystal = np.array([z_crystal])
    # Get edges of crystal (assume original in +X) TODO: fix
    x_crystal = x_crystal - get_header_value(headerdata, f'/gate/{init_volume_name}/geometry/setXLength', split_substr=None, split_idx=1) / 2 + mean_interaction_depth
    # Generate positions of all crystals using repeaters
    #parents.reverse()
    print(parents)
    for parent in parents:
        repeaters = get_header_value(headerdata, f'/gate/{parent}/repeaters/insert', split_substr=None, split_idx=1, dtype=str, return_all=True)
        if not(repeaters):
            continue
        for repeater in repeaters:
            if repeater=='cubicArray':
                repeat_numbers = [get_header_value(headerdata, f'/gate/{parent}/{repeater}/setRepeatNumber{coord}', split_substr=None, split_idx=1) for coord in ['X', 'Y', 'Z']]
                repeat_vector = [get_header_value(headerdata, f'/gate/{parent}/{repeater}/setRepeatVector', split_substr=None, split_idx=i) for i in range(1,4)]
                # GATE convention is that z changes first, so meshgrid is done like this
                xv, zv, yv = np.meshgrid(
                    repeat_vector[0] * (np.arange(0,repeat_numbers[0]) - (repeat_numbers[0]-1)/2),
                    repeat_vector[2] * (np.arange(0,repeat_numbers[2]) - (repeat_numbers[2]-1)/2),
                    repeat_vector[1] * (np.arange(0,repeat_numbers[1]) - (repeat_numbers[1]-1)/2),
                        )
                len_repeat = xv.ravel().shape[0]
                x_crystal_repeated = np.repeat(x_crystal[:,np.newaxis], len_repeat, axis=1)
                y_crystal_repeated = np.repeat(y_crystal[:,np.newaxis], len_repeat, axis=1)
                z_crystal_repeated = np.repeat(z_crystal[:,np.newaxis], len_repeat, axis=1)
                x_crystal = (x_crystal_repeated+xv.ravel()).ravel()
                y_crystal = (y_crystal_repeated+yv.ravel()).ravel()
                z_crystal = (z_crystal_repeated+zv.ravel()).ravel()
            elif repeater=='linear':
                repeat_number = get_header_value(headerdata, f'/gate/{parent}/{repeater}/setRepeatNumber', split_substr=None, split_idx=1)
                repeat_vector = [get_header_value(headerdata, f'/gate/{parent}/{repeater}/setRepeatVector', split_substr=None, split_idx=i) for i in range(1,4)]
                xr = repeat_vector[0] * (np.arange(0,repeat_number) - (repeat_number-1)/2)
                yr = repeat_vector[1] * (np.arange(0,repeat_number) - (repeat_number-1)/2)
                zr = repeat_vector[2] * (np.arange(0,repeat_number) - (repeat_number-1)/2)
                len_repeat = xr.shape[0]
                x_crystal_repeated = np.repeat(x_crystal[:,np.newaxis], len_repeat, axis=1)
                y_crystal_repeated = np.repeat(y_crystal[:,np.newaxis], len_repeat, axis=1)
                z_crystal_repeated = np.repeat(z_crystal[:,np.newaxis], len_repeat, axis=1)
                x_crystal = (x_crystal_repeated+xr).ravel()
                y_crystal = (y_crystal_repeated+yr).ravel()
                z_crystal = (z_crystal_repeated+zr).ravel() 
            elif repeater=='ring':
                repeat_number = get_header_value(headerdata, f'/gate/{parent}/{repeater}/setRepeatNumber', split_substr=None, split_idx=1)
                first_angle = get_header_value(headerdata, f'/gate/{parent}/{repeater}/setFirstAngle', split_substr=None, split_idx=1) * np.pi / 180
                if not first_angle:
                    first_angle = 0
                phi = np.linspace(first_angle, first_angle+2*np.pi, int(repeat_number), endpoint=False)
                x_crystal_repeated = np.repeat(x_crystal[:,np.newaxis], len(phi), axis=1)
                y_crystal_repeated = np.repeat(y_crystal[:,np.newaxis], len(phi), axis=1)
                z_crystal_repeated = np.repeat(z_crystal[:,np.newaxis], len(phi), axis=1)
                x_crystal = (np.cos(phi)*x_crystal_repeated - np.sin(phi)*y_crystal_repeated).ravel()
                y_crystal = (np.sin(phi)*x_crystal_repeated + np.cos(phi)*y_crystal_repeated).ravel()
                z_crystal = (z_crystal_repeated).ravel()
    return np.vstack((x_crystal,y_crystal,z_crystal)).T