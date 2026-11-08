"""The SPECT collimator-detector response: a blur of each plane parallel to the detector that grows with distance."""
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Conv1d
import pytomography
from pytomography.utils import get_distance, compute_pad_size
from pytomography.transforms import Transform
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta

class Seperable1DBlurNet(nn.Module):
    """Blurs each plane parallel to the detector (each x) with its own 1D kernels along r and along z.

    :class:`SPECTPSFTransform` builds one per detector radius from a :class:`~pytomography.metadata.SPECT.SPECTPSFMeta`.

    Args:
        layer_r (nn.Conv1d): The kernels along r (y in the rotated object), one per x.
        layer_z (nn.Conv1d | None, optional): The kernels along z, one per x. Defaults to None: no blur along z.
    """
    def __init__(self, layer_r: Conv1d, layer_z: Conv1d | None = None):
        super(Seperable1DBlurNet, self).__init__()
        self.layer_r = layer_r
        self.layer_z = layer_z

    def forward(self, input):
        """Blurs every plane of constant x with its own kernels (which depend on its distance from the detector).

        Args:
            input (torch.Tensor): The rotated object, [Lx, Ly, Lz].

        Returns:
            torch.Tensor: The blurred object, [Lx, Ly, Lz]; its sum along x models the collimator-detector response.
        """
        output = self.layer_r(torch.permute(input,(2,0,1)))
        # If 2D blurring
        if self.layer_z:
            output = self.layer_z(torch.permute(output,(2,1,0)))
            output = torch.permute(output,(2,1,0))
        return torch.permute(output,(1,2,0))
    
def get_1D_PSF_layer_gaussian(
    sigmas: np.array,
    kernel_size: int,
    ) -> torch.nn.Conv1d:
    """A 1D convolution layer with a normalised Gaussian kernel per plane, for PSF modelling.

    Args:
        sigmas (np.ndarray): The Gaussian sigma of each of the Lx planes, in voxels (it depends on the plane's distance
            from the detector).
        kernel_size (int): Size of the kernels, in voxels; large enough to cover most of the widest Gaussian.

    Returns:
        torch.nn.Conv1d: A grouped convolution (one group per plane) that blurs the last axis of objects of shape
        [L, Lx, L2], each plane with its own kernel, on ``pytomography.device``.
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
    """A 1D convolution layer with a normalised triangular kernel per plane, for square-hole collimators.

    Args:
        sigmas (np.ndarray): Where each of the Lx planes' triangle reaches zero (its full width at half maximum), in
            voxels.
        kernel_size (int): Size of the kernels, in voxels.

    Returns:
        torch.nn.Conv1d: A grouped convolution, as :func:`get_1D_PSF_layer_gaussian` gives.
    """
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
    r"""Object-to-object transform that models the SPECT collimator-detector response (the distance-dependent PSF).

    For each view, every plane of the rotated object parallel to the detector is blurred by the PSF at its distance
    from the detector, so that the sum along x is the blurred projection. The PSF is given by ``psf_meta``, a
    :class:`~pytomography.metadata.SPECT.SPECTPSFMeta`, e.g. from :func:`pytomography.io.SPECT.collimator_psf`:
    separable 1D Gaussian (or triangular, for square holes) blurs along r and z whose width grows with distance. The
    usual choice for low and medium energy SPECT. For any other PSF, such as the kernels spectarf makes from angular
    response functions (septal penetration at high energy), use
    :class:`~pytomography.transforms.SPECT.SPECTKernelPSFTransform`.

    The blur is symmetric, so :meth:`backward` is :meth:`forward`. The kernels are made in :meth:`configure`, one set
    per distinct radius in ``proj_meta.radii``.

    Args:
        psf_meta (SPECTPSFMeta): The PSF's width as a function of the distance.
        assume_padded (bool, optional): The objects given to :meth:`forward` and :meth:`backward` are padded for
            rotation, as they are inside :class:`~pytomography.projectors.SPECT.SPECTSystemMatrix`. Defaults to True.

    Raises:
        TypeError: If ``psf_meta`` is not a :class:`~pytomography.metadata.SPECT.SPECTPSFMeta` (v3's ``psf_operator``,
            SPECTPSFToolbox's operators, was removed in v4: use :class:`SPECTKernelPSFTransform`).
    """
    def __init__(
        self,
        psf_meta: SPECTPSFMeta,
        assume_padded: bool = True,
    ) -> None:
        super(SPECTPSFTransform, self).__init__()
        if not isinstance(psf_meta, SPECTPSFMeta):
            raise TypeError('SPECTPSFTransform takes a SPECTPSFMeta (e.g. from io.SPECT.collimator_psf); for other PSFs '
                            "(v3's psf_operator, or spectarf's kernels) use SPECTKernelPSFTransform")
        self.psf_meta = psf_meta
        self.assume_padded = assume_padded

    def _configure_simple_model(self):
        """Builds the separable 1D layers of every distinct radius from ``psf_meta``."""
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
            
    def configure(
        self,
        object_meta: SPECTObjectMeta,
        proj_meta: SPECTProjMeta
    ) -> None:
        r"""Builds the blurring kernels for the system matrix's grid and detector radii; the system matrix calls it.

        Args:
            object_meta (SPECTObjectMeta): The object grid.
            proj_meta (SPECTProjMeta): The projection space, with the radius of every view.
        """
        super(SPECTPSFTransform, self).configure(object_meta, proj_meta)
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
        r"""Applies the PSF :math:`A:\mathbb{U} \to \mathbb{U}` to the object of one view (detector at +x).

        Args:
            object (torch.Tensor): The object rotated for the view, [Lx, Ly, Lz] (padded), projected along x.
            ang_idx (int): Index of the view in ``proj_meta``: its radius chooses the kernels.

        Returns:
            torch.Tensor: The blurred object, [Lx, Ly, Lz]; its sum along x is the blurred projection.
        """
        return self.layers[self.proj_meta.radii[ang_idx]](object) 
        
    @torch.no_grad()
    def backward(
		self,
		object: torch.Tensor,
		ang_idx: Sequence[int],
	) -> torch.tensor:
        r"""Applies the transpose :math:`A^T:\mathbb{U} \to \mathbb{U}`, which is :meth:`forward` (the blur is symmetric).

        Args:
            object (torch.Tensor): A back projected view, [Lx, Ly, Lz] (padded), in the view's rotated frame.
            ang_idx (int): Index of the view in ``proj_meta``.

        Returns:
            torch.Tensor: The blurred object, [Lx, Ly, Lz].
        """
        return self.layers[self.proj_meta.radii[ang_idx]](object) 