from __future__ import annotations
import torch
from torch.nn.functional import pad
import numpy as np
import os
from torch.nn import Conv1d

def rev_cumsum(x: torch.Tensor):
    """Reverse cumulative sum along the first axis of a tensor of shape [Lx, Ly, Lz].
    since this is used with SPECT attenuation correction, the initial voxel only contributes 1/2.

    Args:
        x (torch.tensor[Lx,Ly,Lz]): Tensor to be summed

    Returns:
        torch.tensor[Lx, Ly, Lz]: The cumulatively summed tensor.
    """
    return torch.cumsum(x.flip(dims=(0,)), dim=0).flip(dims=(0,)) - x/2


def get_distance(Lx: int, r: float, dx: float):
    """Given the radial distance to center of object space from the scanner, computes the distance between each parallel plane (i.e. (y-z plane)) and a detector located at +x. This function is used for SPECT PSF modeling where the amount of blurring depends on thedistance from the detector.

    Args:
        Lx (int): The number of y-z planes to compute the distance of
        r (float): The radial distance between the central y-z plane and the detector at +x.
        dx (float): The spacing between y-z planes in Euclidean distance.

    Returns:
        np.array[Lx]: An array of distances for each y-z plane to the detector.
    """
    if Lx%2==0:
        d = r + (Lx/2 - np.arange(Lx) - 1/2) * dx
    else:
        d = r + ((Lx-1)/2 - np.arange(Lx) ) * dx
    # Correction for if radius of scanner is inside the the bounds
    d[d<0] = 0
    return d

def get_object_nearest_neighbour(object: torch.Tensor, shifts: list[int], mode='replicate'):
    """Given an object tensor, finds the nearest neighbour (corresponding to ``shifts``) for each voxel (done by shifting object by i,j,k)

    Args:
        object (torch.Tensor): Original object
        shifts (list[int]): List of three integers [i,j,k] corresponding to neighbour location

    Returns:
        torch.Tensor: Shifted object whereby each voxel corresponding to neighbour [i,j,k] of the ``object``.
    """
    shift_max = max(np.abs(shifts))
    if shift_max==0: # no shift
        return object
    else:
        shifts = [-shift for shift in shifts]
        neighbour = pad(object.unsqueeze(0), 6*[shift_max], mode=mode).squeeze()
        neighbour = torch.roll(neighbour, shifts=shifts, dims=(0,1,2)) 
        return neighbour[shift_max:-shift_max,shift_max:-shift_max,shift_max:-shift_max]

def print_collimator_parameters():
    """Prints all the available SPECT collimator parameters
    """
    module_path = os.path.dirname(os.path.abspath(__file__))
    collimator_filepath = os.path.join(module_path, '../data/collim.col')
    with open(collimator_filepath) as f:
        for line in f.readlines():
            print(line)
            
def check_if_class_contains_method(instance, method_name):
    """Checks if class corresponding to instance implements the method ``method_name``

    Args:
        instance (Object): A python object
        method_name (str): Name of the method of the object being checked
    """
    if not (hasattr(instance, method_name) and callable(getattr(instance, method_name))):
        raise Exception(f'"{instance.__class__.__name__}" must implement "{method_name}"')

def get_1d_gaussian_kernel(sigma: float, kernel_size: int, padding_mode='zeros') -> Conv1d:
    """Returns a 1D gaussian blurring kernel

    Args:
        sigma (float): Sigma (in pixels) of blurring pixels
        kernel_size (int): Size of kernel used
        padding_mode (str, optional): Type of padding. Defaults to 'zeros'.

    Returns:
        Conv1d: Torch Conv1d layer corresponding to the gaussian kernel
    """
    x = torch.arange(-int(kernel_size//2), int(kernel_size//2)+1)
    k = torch.exp(-x**2/(2*sigma**2)).reshape(1,1,-1)
    k = k/k.sum()
    layer = Conv1d(1,1,kernel_size, padding='same', padding_mode=padding_mode, bias=False)
    layer.weight.data = k
    return layer





    
