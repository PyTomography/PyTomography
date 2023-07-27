from __future__ import annotations
from typing import Sequence
import pytomography
from pytomography.utils import compute_pad_size
import torch
import inspect

class ObjectMeta():
    """Metadata for object space

    Args:
        dr (list[float]): List of 3 elements specifying voxel dimensions in cm.
        shape (list[int]): List of 3 elements [Lx, Ly, Lz] specifying the length of each dimension.
    """
    def __init__(self, dr: list[float], shape: list[int]) -> None:
        self.dr = dr
        self.dx = dr[0]
        self.dy = dr[1]
        self.dz = dr[2]
        self.shape = shape
        self.compute_padded_shape()

    def compute_padded_shape(self) -> list:
        """Computes the padded shape of an object required when rotating the object (to avoid anything getting cut off).
        """
        self.pad_size = compute_pad_size(self.shape[0])
        x_padded = self.shape[0] + 2*self.pad_size
        y_padded = self.shape[1] + 2*self.pad_size
        z_padded = self.shape[2]
        self.padded_shape = (int(x_padded), int(y_padded), int(z_padded)) 
    
    def __repr__(self):
        return f"Shape: {self.shape}, Spacing: {self.dr}cm"


class ImageMeta():
    """Metadata for image space

    Args:
        object_meta (ObjectMeta): Corresponding object space metadata
        angles (list): Specifies the detector angles for all projections in image space
        radii (list, optional): Specifies the radial distance of the detector corresponding to each angle in `angles`; only required in certain cases (i.e. PSF correction). Defaults to None.
    """
    def __init__(
        self,
        object_meta: ObjectMeta,
        angles: Sequence,
        radii=None
    ) -> None:
        self.object_meta = object_meta
        self.angles = torch.tensor(angles).to(pytomography.device).to(pytomography.dtype)
        self.radii = radii
        self.num_projections = len(angles)
        self.shape = (self.num_projections, object_meta.shape[1], object_meta.shape[2])
        self.compute_padded_shape()
        
    def compute_padded_shape(self) -> list:
        """Computes the padded shape of an object required when rotating the object (to avoid anything getting cut off).
        """
        self.pad_size = compute_pad_size(self.shape[1])
        theta_padded = self.shape[0]
        r_padded = self.shape[1] + 2*self.pad_size
        z_padded = self.shape[2]
        self.padded_shape =  (int(theta_padded), int(r_padded), int(z_padded)) 
    
    def __repr__(self):
        return f"Shape: {self.shape}\nAngles: {self.angles}\nRadii: {self.radii}cm\nObjectMeta: {self.object_meta}"


class PSFMeta():
    r"""Metadata for PSF correction. PSF blurring is implemented using Gaussian blurring with :math:`\sigma(r) = f(r,p)` where :math:`r` is the distance from the detector, :math`\sigma` is the width of the Gaussian blurring at that location, and :math:`f(r,p)` is the `sigma_fit` function which takes in additional parameters :math:`p` called `sigma_fit_params`. (By default, `sigma_fit` is a linear curve). As such, :math:`\frac{1}{\sigma\sqrt{2\pi}}e^{-r^2/(2\sigma(r)^2)}` is the point spread function. Blurring is implemented using convolutions with a specified kernel size. 

     Args:
        sigma_fit_params (float): Parameters to the sigma fit function
        sigma_fit (function): Function used to model blurring as a function of radial distance. Defaults to a 2 parameter linear model.
        kernel_dimensions (str): If '1D', blurring is done seperately in each axial plane (so only a 1 dimensional convolution is used). If '2D', blurring is mixed between axial planes (so a 2D convolution is used). Defaults to '2D'.
        min_sigmas (float, optional): This is the number of sigmas to consider in PSF correction. PSF are modelled by Gaussian functions whose extension is infinite, so we need to crop the Gaussian when computing this operation numerically. Note that the blurring width is depth dependent, but the kernel size used for PSF blurring is constant. As such, this parameter is used to fix the kernel size such that all locations have at least ``min_sigmas`` of a kernel size.
    """
    def __init__(
        self,
        sigma_fit_params: Sequence[float, float],
        sigma_fit : function = lambda r, a, b: a*r+b,
        kernel_dimensions: str = '2D',
        min_sigmas: float = 3
    ) -> None:
        self.sigma_fit_params = sigma_fit_params
        self.sigma_fit = sigma_fit
        self.kernel_dimensions = kernel_dimensions
        self.min_sigmas = min_sigmas
        
    def __repr__(self):
        return f"Function: {inspect.getsource(self.sigma_fit)}\nParameters: {self.sigma_fit_params}\nDimensions: {self.kernel_dimensions}\nMaximum sigmas: {self.min_sigmas}"
        