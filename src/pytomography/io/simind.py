import numpy as np
import torch
import os
from pytomography.metadata import ObjectMeta, ImageMeta
from pathlib import Path


def find_first_entry_containing_substring(list_of_attributes, substring, dtype=np.float32):
    line = list_of_attributes[np.char.find(list_of_attributes, substring)>=0][0]
    if dtype == np.float32:
        return np.float32(line.replace('\n', '').split(':=')[-1])
    elif dtype == str:
        return (line.replace('\n', '').split(':=')[-1].replace(' ', ''))
    elif dtype == int:
        return int(line.replace('\n', '').split(':=')[-1].replace(' ', ''))

def simind_projections_to_data(headerfile):    
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    num_proj = find_first_entry_containing_substring(headerdata, 'total number of images', int)
    proj_dim1 = find_first_entry_containing_substring(headerdata, 'matrix size [1]', int)
    proj_dim2 = find_first_entry_containing_substring(headerdata, 'matrix size [2]', int)
    dx = find_first_entry_containing_substring(headerdata, 'scaling factor (mm/pixel) [1]', np.float32) / 10 # to mm
    number_format = find_first_entry_containing_substring(headerdata, 'number format', str)
    num_bytes_per_pixel = find_first_entry_containing_substring(headerdata, 'number of bytes per pixel', np.float32)
    extent_of_rotation = find_first_entry_containing_substring(headerdata, 'extent of rotation', np.float32)
    number_of_projections = find_first_entry_containing_substring(headerdata, 'number of projections', int)
    start_angle = find_first_entry_containing_substring(headerdata, 'start angle', np.float32)
    angles = np.linspace(start_angle, extent_of_rotation, number_of_projections, endpoint=False)
    radius = find_first_entry_containing_substring(headerdata, 'Radius', np.float32) / 10
    imagefile = find_first_entry_containing_substring(headerdata, 'name of data file', str)
    shape_proj= (num_proj, proj_dim1, proj_dim2)
    shape_obj = (proj_dim1, proj_dim1, proj_dim2)
    object_meta = ObjectMeta(dx, shape_obj)
    image_meta = ImageMeta(object_meta, angles, np.ones(len(angles))*radius)
    projections = np.fromfile(os.path.join(str(Path(headerfile).parent), imagefile), dtype=np.float32)
    projections = np.transpose(projections.reshape((num_proj,proj_dim2,proj_dim1))[:,::-1], (0,2,1))
    projections = torch.tensor(projections.copy()).unsqueeze(dim=0)
    return object_meta, image_meta, projections

def simind_CT_to_data(headerfile):    
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    matrix_size_1 = find_first_entry_containing_substring(headerdata, 'matrix size [1]', int)
    matrix_size_2 = find_first_entry_containing_substring(headerdata, 'matrix size [2]', int)
    matrix_size_3 = find_first_entry_containing_substring(headerdata, 'matrix size [3]', int)
    shape = (matrix_size_3, matrix_size_2, matrix_size_1)
    imagefile = find_first_entry_containing_substring(headerdata, 'name of data file', str)
    CT = np.fromfile(os.path.join(str(Path(headerfile).parent), imagefile), dtype=np.float32)
    CT = np.transpose(CT.reshape(shape)[::-1,::-1], (2,1,0))
    CT = torch.tensor(CT.copy())
    return CT
