from __future__ import annotations
from typing import Sequence
import pytomography
from pytomography.utils import compute_pad_size
import torch
from..metadata import ObjectMeta, ProjMeta

class SPECTObjectMeta(ObjectMeta):
    """Metadata for object space in SPECT imaging. Required for padding of object space during the rotate+sum method

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
        self.recon_method = None
        self.units = 'counts'

    def compute_padded_shape(self) -> list:
        """Computes the padded shape of an object required when rotating the object (to avoid anything getting cut off).
        """
        self.pad_size = compute_pad_size(self.shape[0])
        x_padded = self.shape[0] + 2*self.pad_size
        y_padded = self.shape[1] + 2*self.pad_size
        z_padded = self.shape[2]
        self.padded_shape = (int(x_padded), int(y_padded), int(z_padded)) 

class SPECTProjMeta(ProjMeta):
    """Metadata for projection space in SPECT imaging

    Args:
        projection_shape (Sequence): 2D shape of each projection
        dr (Sequence): Pixel dimensions of projection data in cm
        angles (Sequence): The angles for each 2D projection
        radii (Sequence, optional): Specifies the radial distance (in cm) of the detector corresponding to each angle in `angles`; only required in certain cases (i.e. PSF correction). Defaults to None.
    """
    def __init__(
        self,
        projection_shape: Sequence,
        dr: list[float],
        angles: Sequence,
        radii=None
    ) -> None:
        self.angles = torch.tensor(angles).to(pytomography.dtype).to(pytomography.device)
        self.dr = dr
        self.radii = radii
        self.num_projections = len(angles)
        self.shape = (self.num_projections, projection_shape[0], projection_shape[1])
        self.compute_padded_shape()
        
    def compute_padded_shape(self) -> list:
        """Computes the padded shape of an object required when rotating the object (to avoid anything getting cut off).
        """
        self.pad_size = compute_pad_size(self.shape[1])
        theta_padded = self.shape[0]
        r_padded = self.shape[1] + 2*self.pad_size
        z_padded = self.shape[2]
        self.padded_shape =  (int(theta_padded), int(r_padded), int(z_padded)) 

class SPECTPSFMeta():
    r"""Metadata for PSF correction. PSF blurring is implemented using Gaussian blurring with :math:`\sigma(r) = f(r,p)` where :math:`r` is the distance from the detector, :math:`\sigma` is the width of the Gaussian blurring at that location, and :math:`f(r,p)` is the ``sigma_fit`` function which takes in additional parameters :math:`p` called ``sigma_fit_params``. (By default, ``sigma_fit`` is a linear curve). As such, :math:`\frac{1}{\sigma\sqrt{2\pi}}e^{-r^2/(2\sigma(r)^2)}` is the point spread function. Blurring is implemented using convolutions with a specified kernel size. 

     Args:
        sigma_fit_params (float): Parameters to the sigma fit function
        sigma_fit (function): Function used to model blurring as a function of radial distance. Defaults to a 2 parameter linear model.
        kernel_dimensions (str): If '1D', blurring is done seperately in each axial plane (so only a 1 dimensional convolution is used). If '2D', blurring is mixed between axial planes (so a 2D convolution is used). Defaults to '2D'.
        min_sigmas (float, optional): This is the number of sigmas to consider in PSF correction. PSF are modelled by Gaussian functions whose extension is infinite, so we need to crop the Gaussian when computing this operation numerically. Note that the blurring width is depth dependent, but the kernel size used for PSF blurring is constant. As such, this parameter is used to fix the kernel size such that all locations have at least ``min_sigmas`` of a kernel size.
        shape (str, optional): Shape of the PSF. Defaults to 'gaussian', in which case sigma is the sigma of the Gaussian. Can also be 'square' for square collimators, in this case sigma is half the diameter of the bore.
    """
    def __init__(
        self,
        sigma_fit_params: Sequence[float, float],
        sigma_fit : function = lambda r, a, b: a*r+b,
        kernel_dimensions: str = '2D',
        min_sigmas: float | None = 3,
        shape: str = 'gaussian'
    ) -> None:
        self.sigma_fit_params = sigma_fit_params
        self.sigma_fit = sigma_fit
        self.kernel_dimensions = kernel_dimensions
        if shape is 'square':
            self.min_sigmas = 1 # will include whole PSF
        else:
            self.min_sigmas = min_sigmas 
        self.shape = shape
        
    def __repr__(self):
        attributes = [f"{attr} = {getattr(self, attr)}\n" for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        return ''.join(attributes)