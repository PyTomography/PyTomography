import torch
import numpy as np
from pytomography.utils import compute_pad_size
from torch.nn.functional import pad

class ObjectMeta():
    """Metadata for object space
    """
    def __init__(self, dr, shape):
        """Initializer 

        Args:
            dr (listlike): List of 3 elements specifying voxel dimensions in cm
            shape (listlike): List of 3 elements [Lx, Ly, Lz] specifying the length of each dimension
        """
        self.dr = dr
        self.dx = dr[0]
        self.dy = dr[1]
        self.dz = dr[2]
        self.shape = shape
        self.pad_size = compute_pad_size(self.shape[0])
        self.padded_shape = self.compute_padded_shape()

    def compute_padded_shape(self):
        x_padded = self.shape[0] + 2*self.pad_size
        y_padded = self.shape[1] + 2*self.pad_size
        z_padded = self.shape[2]
        return (int(x_padded), int(y_padded), int(z_padded)) 


class ImageMeta():
    """Metadata for image space
    """
    def __init__(self, object_meta, angles, radii=None):
        """Initializer

        Args:
            object_meta (ObjectMeta): Corresponding object space metadata
            angles (list): Specifies the detector angles for all projections in image space
            radii (list, optional): Specifies the radial distance of the detector corresponding to each
            angle in `angles`; only required in certain cases (i.e. PSF correction). Defaults to None.
        """
        self.object_meta = object_meta
        self.angles = angles
        self.radii = radii
        self.num_projections = len(angles)
        self.shape = (self.num_projections, object_meta.shape[1], object_meta.shape[2])

class ObjectTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        t = torch.as_tensor(x).as_subclass(cls)
        if len(t.shape)!=4:
            raise Exception('Tensor should be 4 dimensional')
        return t
    def __init__(self, x, dr):
        super().__init__()
        self.dr = dr
        self.padded = False
        self.original_shape = x.shape
    def pad(self):
        if self.padded:
            raise Exception('Tensor already padded for rotation')
        width = self.data.shape[-1]
        pad_size = int(np.ceil((np.sqrt(2)*width - width)/2))
        self.data = pad(self.data, [0,0,pad_size,pad_size,pad_size,pad_size])
        self.padded = True
    def unpad(self):
        if not self.padded:
            raise Exception('Tensor is not padded for rotation')
        pad_size = (self.data.shape[-2] - self.original_shape[-2])//2
        self.data = self.data[:,pad_size:-pad_size,pad_size:-pad_size,:]
        self.padded = False

class ImageTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        t = torch.as_tensor(x).as_subclass(cls)
        if len(t.shape)!=4:
            raise Exception('Tensor should be 4 dimensional')
        return t
    def __init__(self, x, dr):
        super().__init__()
        self.dr = dr
        self.padded = False
        self.original_shape = x.shape
    def pad(self):
        if self.padded:
            raise Exception('Tensor already padded for rotation')
        width = self.data.shape[-1]
        pad_size = int(np.ceil((np.sqrt(2)*width - width)/2))
        self.data = pad(self.data, [0,0,pad_size,pad_size,pad_size,pad_size])
        self.padded = True
    def unpad(self):
        if not self.padded:
            raise Exception('Tensor is not padded for rotation')
        pad_size = (self.data.shape[-2] - self.original_shape[-2])//2
        self.data = self.data[:,pad_size:-pad_size,pad_size:-pad_size,:]
        self.padded = False

class PSFMeta():
    """Metadata for PSF correction. PSF blurring is implemented using Gaussian blurring with
     :math:`\sigma(d) = ad + b` where :math:`a` is the collimator slope, :math:`b` is the collimator intercept, and :math:`d` is the distance from a plane in object space to a detector aligned parallel with the plane. As such, :math:`sigma(d)` is the point spread function. Blurring is implemented using convolutions with a specified kernel size. 
    """
    def __init__(self, collimator_slope, collimator_intercept, kernel_dimensions='2D', kernel_size=61):
        """Initializer

        Args:
            collimator_slope (float): The collimator slope used for blurring
            collimator_intercept (float): The collimator intercept used for blurring.
            kernel_dimensions (str): If '1D', blurring is done seperately in each axial plane (so
            only a 1 dimensional convolution is used). If '2D', blurring is mixed between axial planes
            (so a 2D convolution is used). Defaults to '2D'.
            kernel_size (int, optional): Size of kernel used for blurring. Defaults to 61.
        """
        self.collimator_slope = collimator_slope
        self.collimator_intercept = collimator_intercept
        self.kernel_dimensions = kernel_dimensions
        self.kernel_size = kernel_size