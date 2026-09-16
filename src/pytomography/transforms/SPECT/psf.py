from __future__ import annotations
from collections.abc import Callable
from typing import Sequence
import torch
import torch.nn as nn
import numpy as np
from fft_conv_pytorch import FFTConv2d
from torch.nn import Conv2d, Conv1d
import copy
import pytomography
from pytomography.utils import get_distance, compute_pad_size
from pytomography.transforms import Transform
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta
    
class Seperable1DBlurNet(nn.Module):
    """Network used to apply Gaussian blurring to each plane parallel to the detector head. The typical network used for low/medium energy SPECT PSF modeling.

    Args:
        layer_r (nn.Conv1d): Kernel used for blurring in radial direction
        layer_z (nn.Conv1d | None): Kernel used for blurring in sup/inf direction.
    """
    def __init__(self, layer_r: Conv1d, layer_z: Conv1d | None = None):
        super(Seperable1DBlurNet, self).__init__()
        self.layer_r = layer_r
        self.layer_z = layer_z

    def forward(self, input):
        """Applies PSF blurring to `input`. Each X-plane gets a different blurring kernel applied, depending on detector distance.

        Args:
            input (torch.tensor): Object to apply Gaussian blurring to

        Returns:
            torch.tensor: Blurred object, adjusted such that subsequent summation along the x-axis models the CDR
        """
        output = self.layer_r(torch.permute(input,(2,0,1)))
        # If 2D blurring
        if self.layer_z:
            output = self.layer_z(torch.permute(output,(2,1,0)))
            output = torch.permute(output,(2,1,0))
        return torch.permute(output,(1,2,0))
    
class PSF2D:
    def __init__(
        self,
        psf_operator,
        distances,
        kernel_size,
        dr,
        ) -> None:
        self.psf_operator = psf_operator
        self.kernel_size = kernel_size
        self.distances = torch.tensor(distances).to(pytomography.dtype).to(pytomography.device)
        x = torch.arange(-(self.kernel_size-1)/2, (self.kernel_size+1)/2, 1).to(pytomography.device) * dr[0]
        y = torch.arange(-(self.kernel_size-1)/2, (self.kernel_size+1)/2, 1).to(pytomography.device) * dr[1]
        self.xv, self.yv = torch.meshgrid(x, y, indexing='xy')
        
    @torch.no_grad()
    def __call__(self, input):
        return self.psf_operator(input,self.xv,self.yv,self.distances,normalize=True)

def get_1D_PSF_layer_gaussian(
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
    sigmas = torch.tensor(sigmas).to(pytomography.dtype).to(pytomography.device).reshape((N,1,1))
    kernel = torch.exp(-x**2 / (2*sigmas**2 + pytomography.delta))
    kernel = kernel / kernel.sum(axis=-1).unsqueeze(-1)
    layer.weight.data = kernel.to(pytomography.dtype)
    return layer

def get_1D_PSF_layer_square(
    sigmas: np.array,
    kernel_size: int,
    ) -> torch.nn.Conv1d:
    
    N = len(sigmas)
    layer = nn.Conv1d(N, N, kernel_size, groups=N, padding='same',
                    padding_mode='zeros', bias=0, device=pytomography.device)
    x = torch.arange(-int(kernel_size//2), int(kernel_size//2)+1).to(pytomography.device).unsqueeze(0).unsqueeze(0).repeat((N,1,1))
    sigmas = torch.tensor(sigmas).to(pytomography.dtype).to(pytomography.device).reshape((N,1,1))
    kernel = 1 - torch.abs(x) / sigmas
    kernel[kernel < 0] = 0
    kernel = kernel / kernel.sum(axis=-1).unsqueeze(-1)
    layer.weight.data = kernel.to(pytomography.dtype)
    return layer

class SPECTPSFTransform(Transform):
    r"""obj2obj transform used to model the effects of PSF blurring in SPECT. The smoothing kernel used to apply PSF modeling uses a Gaussian kernel with width :math:`\sigma` dependent on the distance of the point to the detector; that information is specified in the ``SPECTPSFMeta`` parameter. There are a few potential arguments to initialize this transform (i) `psf_meta`, which contains relevant collimator information to obtain a Gaussian PSF model that works for low/medium energy SPECT (ii) `kernel_f`, an callable function that gives the kernel at any source-detector distance :math:`d`, or (iii) `psf_operator`, a network configured to automatically apply full PSF modeling to a given object :math:`f` at all source-detector distances. Only one of the arguments should be given.

    Args:
        psf_meta (SPECTPSFMeta): Metadata corresponding to the parameters of PSF blurring. In most cases (low/medium energy SPECT), this should be the only given argument.
        kernel_f (Callable): Function :math:`PSF(x,y,d)` that gives PSF at every source-detector distance :math:`d`. It should be able to take in 1D numpy arrays as its first two arguments, and a single argument for the final argument :math:`d`. The function should return a corresponding 2D PSF kernel.
        psf_operator (Callable): Network that takes in an object :math:`f` and applies all necessary PSF correction to return a new object :math:`\tilde{f}` that is PSF corrected, such that subsequent summation along the x-axis accurately models the collimator detector response.
    """
    def __init__(
        self,
        psf_meta: SPECTPSFMeta | None = None,
        psf_operator: Callable | None = None,
        assume_padded: bool = True,
    ) -> None:
        """Initializer that sets corresponding psf parameters"""
        super(SPECTPSFTransform, self).__init__()
        if sum(arg is not None for arg in [psf_meta, psf_operator]) != 1:
            Exception(f'Exactly one of the arguments for initialization should be given.')
        self.psf_meta = psf_meta
        self.psf_operator = psf_operator
        self.assume_padded = assume_padded
        
    def _configure_simple_model(self):
        """Internal function to configure Gaussian modeling. This is called when `psf_meta` is given in initialization
        """
        self.layers = {}
        for radius in np.unique(self.proj_meta.radii):
            kernel_size_r = self._compute_kernel_size(radius, axis=0)
            kernel_size_z = self._compute_kernel_size(radius, axis=2)
            # Compute sigmas and normalize to pixel units
            sigma_r = self._get_sigma(radius)/self.object_meta.dx
            sigma_z = self._get_sigma(radius)/self.object_meta.dz
            if self.psf_meta.shape=='gaussian':
                layer_r = get_1D_PSF_layer_gaussian(sigma_r, kernel_size_r)
                layer_z = get_1D_PSF_layer_gaussian(sigma_z, kernel_size_z)
            elif self.psf_meta.shape=='square':
                layer_r = get_1D_PSF_layer_square(sigma_r, kernel_size_r)
                layer_z = get_1D_PSF_layer_square(sigma_z, kernel_size_z)
            if self.psf_meta.kernel_dimensions=='2D':
                self.layers[radius] = Seperable1DBlurNet(layer_r, layer_z)
            else: # 1D blurring
                self.layers[radius] = Seperable1DBlurNet(layer_r)
            
    def _configure_manual_net(self):
        """Internal function to configure the PSF net. This is called when `psf_operator` is given in initialization
        """
        self.layers = {}
        for radius in np.unique(self.proj_meta.radii):
            dim = self.object_meta.shape[0] + 2*compute_pad_size(self.object_meta.shape[0])
            distances = get_distance(dim, radius, self.object_meta.dx)
            self.layers[radius] = PSF2D(self.psf_operator, distances, self.object_meta.shape[0]-1, self.object_meta.dr)
        
    def configure(
        self,
        object_meta: SPECTObjectMeta,
        proj_meta: SPECTProjMeta
    ) -> None:
        r"""Function used to initalize the transform using corresponding object and projection metadata

        Args:
            object_meta (SPECTObjectMeta): Object metadata.
            proj_meta (SPECTProjMeta): Projections metadata.
        """
        super(SPECTPSFTransform, self).configure(object_meta, proj_meta)
        if self.psf_operator is not None:
            self._configure_manual_net()
        else:
            self._configure_simple_model()
        
    def _compute_kernel_size(self, radius, axis) -> int:
        r"""Function used to compute the kernel size used for PSF blurring. In particular, uses the ``min_sigmas`` attribute of ``SPECTPSFMeta`` to determine what the kernel size should be such that the kernel encompasses at least ``min_sigmas`` at all points in the object. 

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
        r"""Uses PSF Meta data information to get blurring :math:`\sigma` as a function of distance from detector.

        Args:
            radius (float): The distance from the detector.

        Returns:
            array: An array of length Lx corresponding to blurring at each point along the 1st axis in object space
        """
        dim = self.object_meta.shape[0]
        if self.assume_padded:
            dim += 2*compute_pad_size(self.object_meta.shape[0])
        distances = get_distance(dim, radius, self.object_meta.dx)
        sigma = self.psf_meta.sigma_fit(distances, *self.psf_meta.sigma_fit_params)
        return sigma
    
    @torch.no_grad()
    def forward(
		self,
		object: torch.Tensor,
		ang_idx: Sequence[int], 
	) -> torch.tensor:
        r"""Applies the PSF transform :math:`A:\mathbb{U} \to \mathbb{U}` for the situation where an object is being detector by a detector at the :math:`+x` axis.

        Args:
            object_i (torch.tensor): Tensor of size [Lx, Ly, Lz] being projected along its first axis
            ang_idx (int): The projection indices: used to find the corresponding angle in projection space corresponding to each projection angle in ``object_i``.

        Returns:
            torch.tensor: Tensor of size [Lx, Ly, Lz] such that projection of this tensor along the first axis corresponds to n PSF corrected projection.
        """
        return self.layers[self.proj_meta.radii[ang_idx]](object) 
        
    @torch.no_grad()
    def backward(
		self,
		object: torch.Tensor,
		ang_idx: Sequence[int],
	) -> torch.tensor:
        r"""Applies the transpose of the PSF transform :math:`A^T:\mathbb{U} \to \mathbb{U}` for the situation where an object is being detector by a detector at the :math:`+x` axis. Since the PSF transform is a symmetric matrix, its implemtation is the same as the ``forward`` method.

        Args:
            object_i (torch.tensor): Tensor of size [Lx, Ly, Lz] being projected along its first axis
            ang_idx (int): The projection index

        Returns:
            torch.tensor: Tensor of size [Lx, Ly, Lz] such that projection of this tensor along the first axis corresponds to n PSF corrected projection.
        """
        return self.layers[self.proj_meta.radii[ang_idx]](object) 