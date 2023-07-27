from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn
import numpy as np
import pytomography
from pytomography.utils import get_distance, compute_pad_size, pad_object, pad_object_z, unpad_object_z, rotate_detector_z, unpad_object
from pytomography.transforms import Transform
from pytomography.metadata import ObjectMeta, ImageMeta, PSFMeta
import time
    
class GaussianBlurNet(nn.Module):
    def __init__(self, layer_r, layer_z=None):
        super(GaussianBlurNet, self).__init__()
        self.layer_r = layer_r
        self.layer_z = layer_z

    def forward(self, input):
        output = self.layer_r(torch.permute(input[0],(1,0,2)))
        # If 2D blurring
        if self.layer_z:
            output = self.layer_z(torch.permute(output,(2,1,0)))
        return torch.permute(output,(1,2,0)).unsqueeze(0)

def get_1D_PSF_layer(
    sigmas: np.array,
    kernel_size: int,
    ) -> torch.nn.Conv1d:
    """Creates a 1D convolutional layer that is used for PSF modeling.

    Args:
        sigmas (array): Array of length Lx corresponding to blurring (sigma of Gaussian) as a function of distance from scanner
        kernel_size (int): Size of the kernel used in each layer. Needs to be large enough to cover most of Gaussian
        
    Returns:
        torch.nn.Conv2d: Convolutional neural network layer used to apply blurring to objects of shape [batch_size, Lx, Ly, Lz]
    """
    N = len(sigmas)
    layer = nn.Conv1d(N, N, kernel_size, groups=N, padding='same',
                    padding_mode='zeros', bias=0, device=pytomography.device)
    x = torch.arange(-int(kernel_size//2), int(kernel_size//2)+1).to(pytomography.device).unsqueeze(0).unsqueeze(0).repeat((N,1,1))
    sigmas = torch.tensor(sigmas).to(pytomography.device).to(pytomography.dtype).reshape((N,1,1))
    kernel = torch.exp(-x**2 / (2*sigmas**2 + pytomography.delta))
    kernel = kernel / kernel.sum(axis=-1).unsqueeze(-1)
    layer.weight.data = kernel.to(torch.float32)
    return layer

class SPECTPSFTransform(Transform):
    """obj2obj transform used to model the effects of PSF blurring in SPECT. The smoothing kernel used to apply PSF modeling uses a Gaussian kernel with width :math:`\sigma` dependent on the distance of the point to the detector; that information is specified in the ``PSFMeta`` parameter. 

    Args:
        psf_meta (PSFMeta): Metadata corresponding to the parameters of PSF blurring
    """
    def __init__(
        self,
        psf_meta: PSFMeta, 
    ) -> None:
        """Initializer that sets corresponding psf parameters"""
        super(SPECTPSFTransform, self).__init__()
        self.psf_meta = psf_meta

    def configure(
        self,
        object_meta: ObjectMeta,
        image_meta: ImageMeta
    ) -> None:
        """Function used to initalize the transform using corresponding object and image metadata

        Args:
            object_meta (ObjectMeta): Object metadata.
            image_meta (ImageMeta): Image metadata.
        """
        super(SPECTPSFTransform, self).configure(object_meta, image_meta)
        self.layers = {}
        for radius in np.unique(image_meta.radii):
            kernel_size_r = self.compute_kernel_size(radius, axis=0)
            kernel_size_z = self.compute_kernel_size(radius, axis=2)
            # Compute sigmas and normalize to pixel units
            sigma_r = self.get_sigma(radius)/object_meta.dx
            sigma_z = self.get_sigma(radius)/object_meta.dz
            layer_r = get_1D_PSF_layer(sigma_r, kernel_size_r)
            layer_z = get_1D_PSF_layer(sigma_z, kernel_size_z)
            if self.psf_meta.kernel_dimensions=='2D':
                self.layers[radius] = GaussianBlurNet(layer_r, layer_z)
            else: # 1D blurring
                self.layers[radius] = GaussianBlurNet(layer_r)
        
    def compute_kernel_size(self, radius, axis) -> int:
        """Function used to compute the kernel size used for PSF blurring. In particular, uses the ``min_sigmas`` attribute of ``PSFMeta`` to determine what the kernel size should be such that the kernel encompasses at least ``min_sigmas`` at all points in the object. 

        Returns:
            int: The corresponding kernel size used for PSF blurring.
        """
        sigma_max = max(self.get_sigma(radius))
        sigma_max /= self.object_meta.dr[axis]
        return (np.ceil(sigma_max * self.psf_meta.min_sigmas)*2 + 1).astype(int)
    
    def get_sigma(
        self,
        radius: float,
    ) -> np.array:
        """Uses PSF Meta data information to get blurring :math:`\sigma` as a function of distance from detector.

        Args:
            radius (float): The distance from the detector.

        Returns:
            array: An array of length Lx corresponding to blurring at each point along the 1st axis in object space
        """
        dim = self.object_meta.shape[0] + 2*compute_pad_size(self.object_meta.shape[0])
        distances = get_distance(dim, radius, self.object_meta.dx)
        sigma = self.psf_meta.sigma_fit(distances, *self.psf_meta.sigma_fit_params)
        return sigma
    
    def apply_psf(self, object, ang_idx):
        object_return = []
        for i in range(len(ang_idx)):
            object_temp = object[i].unsqueeze(0)
            object_temp = self.layers[self.image_meta.radii[ang_idx[i]]](object_temp) 
            object_return.append(object_temp)
        return torch.vstack(object_return)
    
    @torch.no_grad()
    def forward(
		self,
		object_i: torch.Tensor,
		ang_idx: int, 
	) -> torch.tensor:
        r"""Applies the PSF transform :math:`A:\mathbb{U} \to \mathbb{U}` for the situation where an object is being detector by a detector at the :math:`+x` axis.

        Args:
            object_i (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] being projected along its first axis
            ang_idx (int): The projection indices: used to find the corresponding angle in image space corresponding to each projection angle in ``object_i``.

        Returns:
            torch.tensor: Tensor of size [batch_size, Lx, Ly, Lz] such that projection of this tensor along the first axis corresponds to n PSF corrected projection.
        """
        return self.apply_psf(object_i, ang_idx)
        
    @torch.no_grad()
    def backward(
		self,
		object_i: torch.Tensor,
		ang_idx: int, 
		norm_constant: torch.Tensor | None = None,
	) -> torch.tensor:
        r"""Applies the transpose of the PSF transform :math:`A^T:\mathbb{U} \to \mathbb{U}` for the situation where an object is being detector by a detector at the :math:`+x` axis. Since the PSF transform is a symmetric matrix, its implemtation is the same as the ``forward`` method.

        Args:
            object_i (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] being projected along its first axis
            ang_idx (int): The projection indices: used to find the corresponding angle in image space corresponding to each projection angle in ``object_i``.
            norm_constant (torch.tensor, optional): A tensor used to normalize the output during back projection. Defaults to None.

        Returns:
            torch.tensor: Tensor of size [batch_size, Lx, Ly, Lz] such that projection of this tensor along the first axis corresponds to n PSF corrected projection.
        """
        if norm_constant is not None:
            object_i = self.apply_psf(object_i, ang_idx)
            norm_constant = self.apply_psf(norm_constant, ang_idx)
            return object_i, norm_constant
        else:
            return self.apply_psf(object_i, ang_idx)