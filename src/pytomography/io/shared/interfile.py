from __future__ import annotations
from pathlib import Path
import numpy as np
import os
import torch
import pytomography

def get_header_value(
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
    
def get_attenuation_map_interfile(headerfile: str):
    """Opens attenuation data from SIMIND output

    Args:
        headerfile (str): Path to header file

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
    amap = np.fromfile(os.path.join(str(Path(headerfile).parent), imagefile), dtype=np.float32)
    # Flip "Z" ("X" in SIMIND) b/c "first density image located at +X" according to SIMIND manual
    # Flip "Y" ("Z" in SIMIND) b/c axis convention is opposite for x22,5x (mu-castor format)
    amap = np.transpose(amap.reshape(shape), (2,1,0))[:,::-1,::-1]
    amap = torch.tensor(amap.copy()).unsqueeze(dim=0)
    return amap.to(pytomography.device)