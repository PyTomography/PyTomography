"""Small helpers used across PyTomography: cumulative sums, distances to a detector, neighbouring voxels, kernels."""
from __future__ import annotations
import torch
from torch.nn.functional import pad
import numpy as np
import os
from torch.nn import Conv1d

def rev_cumsum(x: torch.Tensor):
    r"""Reverse cumulative sum along the first axis, in which each voxel counts only half towards its own sum.

    The result is :math:`y_i = \sum_{j \ge i} x_j - x_i / 2`. SPECT attenuation modelling uses it to integrate the
    attenuation from the centre of each voxel to the detector, which is at +x (the end of the first axis) once the
    object is rotated to the view.

    Args:
        x (torch.Tensor): Tensor to sum, e.g. [Lx, Ly, Lz] (summed along x).

    Returns:
        torch.Tensor: The sums, of the shape of ``x``.
    """
    return torch.cumsum(x.flip(dims=(0,)), dim=0).flip(dims=(0,)) - x/2


def get_distance(Lx: int, r: float, dx: float):
    r"""Distance from each y-z plane of an object to a detector at +x, for distance-dependent SPECT PSF modelling.

    Plane :math:`i` lies :math:`(i - (L_x - 1)/2)\,dx` from the centre of the object along x, so its distance to the
    detector is :math:`r - (i - (L_x - 1)/2)\,dx`; planes beyond the detector get 0.

    Args:
        Lx (int): Number of y-z planes.
        r (float): Distance from the central plane (the centre of the object) to the detector, in mm.
        dx (float): Spacing of the planes, in mm.

    Returns:
        np.ndarray: The distances [Lx], in mm, largest for the first plane.
    """
    if Lx%2==0:
        d = r + (Lx/2 - np.arange(Lx) - 1/2) * dx
    else:
        d = r + ((Lx-1)/2 - np.arange(Lx) ) * dx
    # Correction for if radius of scanner is inside the the bounds
    d[d<0] = 0
    return d

def get_object_nearest_neighbour(object: torch.Tensor, shifts: list[int], mode='replicate'):
    r"""The value of each voxel's neighbour at a given offset, i.e. ``object`` shifted by ``-shifts``.

    The result at voxel :math:`(i, j, k)` is ``object`` at :math:`(i + s_0, j + s_1, k + s_2)`; neighbours outside the
    object are taken from ``object`` padded with ``mode``. With every shift 0, ``object`` itself is returned.

    Args:
        object (torch.Tensor): Image [Lx, Ly, Lz].
        shifts (list[int]): Offset :math:`(s_0, s_1, s_2)` of the neighbour, in voxels.
        mode (str, optional): Mode of ``torch.nn.functional.pad`` for the neighbours outside the object
            (``'replicate'``: the nearest voxel on the edge). Defaults to ``'replicate'``.

    Returns:
        torch.Tensor: The neighbours [Lx, Ly, Lz].
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
    """Prints PyTomography's table of SPECT collimators, which comes from SIMIND's collimator database.

    Each record starts with ``*`` and gives the collimator's code (as :func:`pytomography.io.SPECT.collimator_psf` takes
    it), then its hole diameter, septal thickness and hole length in cm (SIMIND's unit), its hole shape and type, and a
    description; the other lines are comments.
    """
    module_path = os.path.dirname(os.path.abspath(__file__))
    collimator_filepath = os.path.join(module_path, '../data/collim.col')
    with open(collimator_filepath) as f:
        for line in f.readlines():
            print(line)
            
def check_if_class_contains_method(instance, method_name):
    """Checks that an object has a method called ``method_name``, and raises an exception if it does not.

    Args:
        instance (object): Any object.
        method_name (str): Name of the method.

    Raises:
        Exception: ``instance`` has no callable attribute ``method_name``.
    """
    if not (hasattr(instance, method_name) and callable(getattr(instance, method_name))):
        raise Exception(f'"{instance.__class__.__name__}" must implement "{method_name}"')

def get_1d_gaussian_kernel(sigma: float, kernel_size: int, padding_mode='zeros') -> Conv1d:
    """A 1D Gaussian blur, normalised to sum to 1, as a ``torch.nn.Conv1d`` layer with ``'same'`` padding.

    Args:
        sigma (float): Standard deviation of the Gaussian, in samples.
        kernel_size (int): Number of taps of the kernel; odd.
        padding_mode (str, optional): Padding at the edges, as for ``torch.nn.Conv1d`` (``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``). Defaults to ``'zeros'``.

    Returns:
        Conv1d: The layer (on the CPU), applied to tensors [N, 1, L]; its output has the length of its input.
    """
    x = torch.arange(-int(kernel_size//2), int(kernel_size//2)+1)
    k = torch.exp(-x**2/(2*sigma**2)).reshape(1,1,-1)
    k = k/k.sum()
    layer = Conv1d(1,1,kernel_size, padding='same', padding_mode=padding_mode, bias=False)
    layer.weight.data = k
    return layer





    
