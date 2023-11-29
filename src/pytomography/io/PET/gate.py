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