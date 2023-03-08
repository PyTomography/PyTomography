from __future__ import annotations
from pytomography.utils import compute_pad_size
import numpy as np

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
        self.pad_size = compute_pad_size(self.shape[0])
        self.padded_shape = self.compute_padded_shape()

    def compute_padded_shape(self) -> list:
        """Computes the padded shape of an object required when rotating the object (to avoid anything getting cut off).

        Returns:
            list: Padded dimensions of the object.
        """
        x_padded = self.shape[0] + 2*self.pad_size
        y_padded = self.shape[1] + 2*self.pad_size
        z_padded = self.shape[2]
        return (int(x_padded), int(y_padded), int(z_padded)) 


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
        angles: np.array,
        radii=None
    ) -> None:
        self.object_meta = object_meta
        self.angles = angles
        self.radii = radii
        self.num_projections = len(angles)
        self.shape = (self.num_projections, object_meta.shape[1], object_meta.shape[2])
        self.pad_size = compute_pad_size(self.shape[1])
        self.padded_shape = self.compute_padded_shape()
        
    def compute_padded_shape(self) -> list:
        """Computes the padded shape of an object required when rotating the object (to avoid anything getting cut off).

        Returns:
            list: Padded dimensions of the object.
        """
        theta_padded = self.shape[0]
        r_padded = self.shape[1] + 2*self.pad_size
        z_padded = self.shape[2]
        return (int(theta_padded), int(r_padded), int(z_padded)) 


class PSFMeta():
    r"""Metadata for PSF correction. PSF blurring is implemented using Gaussian blurring with
     :math:`\sigma(d) = ad + b` where :math:`a` is the collimator slope, :math:`b` is the collimator intercept, and :math:`d` is the distance from a plane in object space to a detector aligned parallel with the plane: as such, :math:`\frac{1}{\sigma\sqrt{2\pi}}e^{-r^2/(2\sigma(d)^2)}` is the point spread function where :math:`r` is the radial distance between some point in image space and the corresponding point in object space. Blurring is implemented using convolutions with a specified kernel size. 

     Args:
        collimator_slope (float): The collimator slope used for blurring (dimensionless units)
        collimator_intercept (float): The collimator intercept used for blurring. Should be in units of cm.
        kernel_dimensions (str): If '1D', blurring is done seperately in each axial plane (so only a 1 dimensional convolution is used). If '2D', blurring is mixed between axial planes (so a 2D convolution is used). Defaults to '2D'.
        max_sigmas (float, optional): This is the number of sigmas to consider in PSF correction. PSF are modelled by Gaussian functions whose extension is infinite, so we need to crop the Gaussian when computing this operation numerically. Note that the blurring width is depth dependent, but the kernel size used for PSF blurring is constant. As such, this parameter is used to fix the kernel size such that all locations have at least ``max_sigmas`` of a kernel size.
    """
    def __init__(
        self,
        collimator_slope: float,
        collimator_intercept: float,
        kernel_dimensions: str = '2D',
        max_sigmas: float = 3
    ) -> None:
        self.collimator_slope = collimator_slope
        self.collimator_intercept = collimator_intercept
        self.kernel_dimensions = kernel_dimensions
        self.max_sigmas = max_sigmas