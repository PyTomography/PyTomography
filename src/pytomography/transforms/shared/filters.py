import torch
import numpy as np
from torch.nn.functional import conv1d
import pytomography
from pytomography.metadata import ObjectMeta, ProjMeta
from pytomography.transforms import Transform

class GaussianFilter(Transform):
    """Applies a Gaussian smoothing filter to the reconstructed object with the specified full-width-half-max (FWHM)

    Args:
        FWHM (float): Specifies the width of the gaussian
        n_sigmas (float): Number of sigmas to include before truncating the kernel.
    """
    def __init__(self, FWHM: float, n_sigmas: float = 3):
        self.sigma = FWHM / (2*np.sqrt(2*np.log(2))) 
        self.n_sigmas = n_sigmas
        
    def configure(self, object_meta: ObjectMeta, proj_meta: ProjMeta) -> None:
        """Configures the transform to the object/proj metadata. This is done after creating the network so that it can be adjusted to the system matrix.

        Args:
            object_meta (ObjectMeta): Object metadata.
            proj_meta (ProjMeta): Projections metadata.
        """
        self.object_meta = object_meta
        self.proj_meta = proj_meta
        self._get_kernels()
        
    def _get_kernels(self):
        """Obtains required kernels for smoothing
        """
        self.kernels = []
        for i in range(3):
            dx = self.object_meta.dr[i]
            kernel_size = int(2*np.ceil(self.n_sigmas*self.sigma/dx)+1)
            x = torch.arange(-int(kernel_size//2), int(kernel_size//2)+1).to(pytomography.device)*dx
            k = torch.exp(-x**2/(2*self.sigma**2)).reshape(1,1,-1)
            self.kernels.append(k/k.sum())
            
    def __call__(self, object):
        """Alternative way to call"""
        return self.forward(object)
            
    def forward(self, object):
        """Applies the Gaussian smoothing

        Args:
            object (torch.tensor): Object to smooth

        Returns:
            torch.tensor: Smoothed object
        """
        for i in [0,1,2]:
            object = object.swapaxes(i,2)
            new_shape = object.shape
            object = object.reshape(-1,1,new_shape[-1])
            object = conv1d(object, self.kernels[i], padding='same')
            object = object.reshape(new_shape)
            object= object.swapaxes(i,2)
        return object
            
    def backward(self, object, norm_constant=None):
        """Applies Gaussian smoothing in back projection. Because the operation is symmetric, it is the same as the forward projection.

        Args:
            object (torch.tensor): Object to smooth
            norm_constant (torch.tensor, optional): Normalization constant used in iterative algorithms. Defaults to None.

        Returns:
            torch.tensor: Smoothed object
        """

        object = self.forward(object)
        if norm_constant is not None:
            norm_constant = self.forward(norm_constant)
            return object, norm_constant
        else:
            return object

    