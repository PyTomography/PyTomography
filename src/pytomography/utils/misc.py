from __future__ import annotations
import torch
from torch.nn.functional import pad
import numpy as np
import os

def rev_cumsum(x: torch.Tensor):
    """Reverse cumulative sum along the first axis of a tensor of shape [batch_size, Lx, Ly, Lz].
    since this is used with SPECT attenuation correction, the initial voxel only contributes 1/2.

    Args:
        x (torch.tensor[batch_size,Lx,Ly,Lz]): Tensor to be summed

    Returns:
        torch.tensor[batch_size, Lx, Ly, Lz]: The cumulatively summed tensor.
    """
    return torch.cumsum(x.flip(dims=(1,)), dim=1).flip(dims=(1,)) - x/2


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

def get_object_nearest_neighbour(object: torch.Tensor, shifts: list[int]):
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
        neighbour = pad(object, 6*[shift_max])
        neighbour = torch.roll(neighbour, shifts=shifts, dims=(1,2,3)) 
        return neighbour[:,shift_max:-shift_max,shift_max:-shift_max,shift_max:-shift_max]

def get_blank_below_above(proj: torch.tensor):
    """Obtains the number of blank z-slices at the sup (``blank_above``) and inf (``blank_below``) of the projection data. This method is entirely empircal, and looks for z slices where there are zero detected counts.

    Args:
        proj (torch.tensor): Projection data from a scanner

    Returns:
        Sequence[int]: A tuple of two elements corresponding to the number of blank slices at the inf, and the number of blank slices at the sup.
    """
    greater_than_zero = (proj[0].cpu().numpy() > 0).sum(axis=(0,1))>0
    blank_below = np.argmax(greater_than_zero)
    blank_above = proj[0].cpu().numpy().shape[-1] - np.argmax(greater_than_zero[::-1])
    return blank_below, blank_above

def print_collimator_parameters():
    """Prints all the available SPECT collimator parameters
    """
    module_path = os.path.dirname(os.path.abspath(__file__))
    collimator_filepath = os.path.join(module_path, '../data/collim.col')
    with open(collimator_filepath) as f:
        for line in f.readlines():
            print(line)






    
