from __future__ import annotations
from collections.abc import Callable
from typing import Sequence
import torch
import torch.nn as nn
import numpy as np
from fft_conv_pytorch import FFTConv2d
from torch.nn import Conv2d
import pytomography
from pytomography.utils import get_distance, compute_pad_size
from pytomography.transforms import Transform
from pytomography.metadata import SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta
    
class GaussianBlurNet(nn.Module):
    def __init__(self, layer_r, layer_z=None):
        super(GaussianBlurNet, self).__init__()
        self.layer_r = layer_r
        self.layer_z = layer_z

    def forward(self, input):
        output = self.layer_r(torch.permute(input[0],(2,0,1)))
        # If 2D blurring
        if self.layer_z:
            output = self.layer_z(torch.permute(output,(2,1,0)))
            output = torch.permute(output,(2,1,0))
        return torch.permute(output,(1,2,0)).unsqueeze(0)
    
class ArbitraryPSFNet(nn.Module):
    def __init__(self, kernel_f, distances, kernel_size, dr):
        super(ArbitraryPSFNet, self).__init__()
        self.kernel_f = kernel_f
        self.kernel_size = kernel_size
        self.distances = distances
        self.x_eval = np.arange(-(kernel_size-1)/2, (kernel_size+1)/2, 1) * dr[0]
        self.y_eval = np.arange(-(kernel_size-1)/2, (kernel_size+1)/2, 1) * dr[1]

    @torch.no_grad()
    def forward(self, input):
        groups = input.shape[1]
        kernel = torch.tensor(np.array([self.kernel_f(self.x_eval, self.y_eval, d) for d in self.distances])).unsqueeze(1).to(pytomography.device).to(pytomography.dtype)
        net = FFTConv2d(groups, groups, self.kernel_size, padding=int((self.kernel_size-1)/2), groups=groups, bias=False).to(pytomography.device)
        net.weight = torch.nn.Parameter(kernel)
        return net(input)

def get_1D_PSF_layer(
    sigmas: np.array,
    kernel_size: int,
    ) -> torch.nn.Conv1d:
    """Creates a 1D convolutional layer that is used for PSF modeling.

    Args:
        sigmas (array): Array of length Lx corresponding to blurring (sigma of Gaussian) as a function of distance from scanner
        kernel_size (int): Size of the kernel used in each layer. Needs to be large enough to cover most of Gaussian
        
    Returns:
        torch.nn.Conv2d: Convolutional neural network layer used to apply blurring to objects of shape [Lx, L1, L2] where Lx is treated as a batch size, L1 as the channel (or group index) and L2 is the axis being blurred over 
    """
    N = len(sigmas)
    layer = nn.Conv1d(N, N, kernel_size, groups=N, padding='same',
                    padding_mode='zeros', bias=0, device=pytomography.device)
    x = torch.arange(-int(kernel_size//2), int(kernel_size//2)+1).to(pytomography.device).unsqueeze(0).unsqueeze(0).repeat((N,1,1))
    sigmas = torch.tensor(sigmas).to(pytomography.device).to(pytomography.dtype).reshape((N,1,1))
    kernel = torch.exp(-x**2 / (2*sigmas**2 + pytomography.delta))
    kernel = kernel / kernel.sum(axis=-1).unsqueeze(-1)
    layer.weight.data = kernel.to(pytomography.dtype)
    return layer

class SPECTPSFTransform(Transform):
    r"""obj2obj transform used to model the effects of PSF blurring in SPECT. The smoothing kernel used to apply PSF modeling uses a Gaussian kernel with width :math:`\sigma` dependent on the distance of the point to the detector; that information is specified in the ``SPECTPSFMeta`` parameter. 

    Args:
        psf_meta (SPECTPSFMeta): Metadata corresponding to the parameters of PSF blurring
        kernel_f (Callable): Function :math:`PSF(x,y,d)` that gives PSF at every source-detector distance :math:`d`.
    """
    def __init__(
        self,
        psf_meta: SPECTPSFMeta | None = None,
        kernel_f: Callable | None = None
    ) -> None:
        """Initializer that sets corresponding psf parameters"""
        super(SPECTPSFTransform, self).__init__()
        if (kernel_f is None)*(psf_meta is None):
            Exception(f'Must give at least one of (i) PSF metadata to obtain parameters for Gaussian fit or (ii) kernel function')
        self.psf_meta = psf_meta
        self.kernel_f = kernel_f
        
    def _configure_gaussian_model(self):
        self.layers = {}
        for radius in np.unique(self.proj_meta.radii):
            kernel_size_r = self._compute_kernel_size(radius, axis=0)
            kernel_size_z = self._compute_kernel_size(radius, axis=2)
            # Compute sigmas and normalize to pixel units
            sigma_r = self._get_sigma(radius)/self.object_meta.dx
            sigma_z = self._get_sigma(radius)/self.object_meta.dz
            layer_r = get_1D_PSF_layer(sigma_r, kernel_size_r)
            layer_z = get_1D_PSF_layer(sigma_z, kernel_size_z)
            if self.psf_meta.kernel_dimensions=='2D':
                self.layers[radius] = GaussianBlurNet(layer_r, layer_z)
            else: # 1D blurring
                self.layers[radius] = GaussianBlurNet(layer_r)
                
    def _configure_kernel_model(self):
        self.layers = {}
        kernel_size = self.object_meta.shape[0] - 1
        #kernel_size = int(self.object_meta.shape[0]/2) - 1
        for radius in np.unique(self.proj_meta.radii):
            dim = self.object_meta.shape[0] + 2*compute_pad_size(self.object_meta.shape[0])
            distances = get_distance(dim, radius, self.object_meta.dx)
            self.layers[radius] = ArbitraryPSFNet(self.kernel_f, distances, kernel_size, (self.object_meta.dx, self.object_meta.dz))
        
    def configure(
        self,
        object_meta: SPECTObjectMeta,
        proj_meta: SPECTProjMeta
    ) -> None:
        """Function used to initalize the transform using corresponding object and projection metadata

        Args:
            object_meta (SPECTObjectMeta): Object metadata.
            proj_meta (SPECTProjMeta): Projections metadata.
        """
        super(SPECTPSFTransform, self).configure(object_meta, proj_meta)
        if self.kernel_f is not None:
            self._configure_kernel_model()
        else:
            self._configure_gaussian_model()
        
    def _compute_kernel_size(self, radius, axis) -> int:
        """Function used to compute the kernel size used for PSF blurring. In particular, uses the ``min_sigmas`` attribute of ``SPECTPSFMeta`` to determine what the kernel size should be such that the kernel encompasses at least ``min_sigmas`` at all points in the object. 

        Returns:
            int: The corresponding kernel size used for PSF blurring.
        """
        sigma_max = max(self._get_sigma(radius))
        sigma_max /= self.object_meta.dr[axis]
        return (np.ceil(sigma_max * self.psf_meta.min_sigmas)*2 + 1).astype(int)
    
    def _get_sigma(
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
    
    def _apply_psf(
        self,
        object: torch.tensor,
        ang_idx: Sequence[int]
        ) -> torch.tensor:
        """Applies PSF modeling to an object with corresponding angle indices

        Args:
            object (torch.tensor): Tensor of shape ``[batch_size, Lx, Ly, Lz]`` corresponding to object rotated at different angles
            ang_idx (Sequence[int]): List of length ``batch_size`` corresponding to angle of each object in the batch

        Returns:
            torch.tensor: Object with PSF modeling applied
        """
        object_return = []
        for i in range(len(ang_idx)):
            object_temp = object[i].unsqueeze(0)
            object_temp = self.layers[self.proj_meta.radii[ang_idx[i]]](object_temp) 
            object_return.append(object_temp)
        return torch.vstack(object_return)
    
    @torch.no_grad()
    def forward(
		self,
		object_i: torch.Tensor,
		ang_idx: Sequence[int], 
	) -> torch.tensor:
        r"""Applies the PSF transform :math:`A:\mathbb{U} \to \mathbb{U}` for the situation where an object is being detector by a detector at the :math:`+x` axis.

        Args:
            object_i (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] being projected along its first axis
            ang_idx (int): The projection indices: used to find the corresponding angle in projection space corresponding to each projection angle in ``object_i``.

        Returns:
            torch.tensor: Tensor of size [batch_size, Lx, Ly, Lz] such that projection of this tensor along the first axis corresponds to n PSF corrected projection.
        """
        return self._apply_psf(object_i, ang_idx)
        
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
            ang_idx (int): The projection indices: used to find the corresponding angle in projection space corresponding to each projection angle in ``object_i``.
            norm_constant (torch.tensor, optional): A tensor used to normalize the output during back projection. Defaults to None.

        Returns:
            torch.tensor: Tensor of size [batch_size, Lx, Ly, Lz] such that projection of this tensor along the first axis corresponds to n PSF corrected projection.
        """
        if norm_constant is not None:
            object_i = self._apply_psf(object_i, ang_idx)
            norm_constant = self._apply_psf(norm_constant, ang_idx)
            return object_i, norm_constant
        else:
            return self._apply_psf(object_i, ang_idx)