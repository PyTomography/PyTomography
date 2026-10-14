"""The SPECT collimator-detector response: a blur of each plane parallel to the detector that grows with distance."""
from __future__ import annotations
from collections.abc import Callable
from typing import Sequence
import torch
import torch.nn as nn
import numpy as np
from fft_conv_pytorch import FFTConv2d
from torch.nn import Conv2d, Conv1d
from torch.nn.functional import pad
import copy
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
    
class PSF2D:
    r"""Applies a 2D PSF operator (e.g. from the SPECTPSFToolbox) to each plane of an object.

    The operator provides the 2D kernel of every plane; this class convolves the object with it. When the operator
    exposes its kernel (a ``_get_kernel`` method), the convolution is performed here as a depthwise FFT convolution:
    each plane is blurred by its own kernel, so in Fourier space this is an elementwise product of the object and
    kernel spectra. Operators otherwise convolve through ``fft_conv_pytorch``, whose grouped ``complex_matmul``
    expresses that product as a batched matrix multiplication of one-by-one matrices, one per plane and frequency,
    which costs far more than the transforms themselves. The kernel is obtained from the operator either way, so the
    modeled PSF is unchanged; the fast path is used only after its output has been checked against the operator's own
    output on the first call. :class:`SPECTPSFTransform` builds one per detector radius when given a ``psf_operator``.

    Args:
        psf_operator (Callable): Operator that models the collimator-detector response.
        distances (Sequence): Source-detector distance of each plane of the object, in mm.
        kernel_size (int): Size of the (square) kernel of each plane, in voxels.
        dr (Sequence[float]): Voxel size, in mm.
        cache_kernel (bool, optional): If True, the Fourier transform of the kernel is computed once and kept on the
            device, instead of being recomputed at every projection angle. The kernel is the same for every angle at a
            given radius, so this saves time (a further ~1.5x on top of the depthwise convolution) at the cost of
            memory: one complex spectrum per distinct radius in ``proj_meta.radii``, e.g. 57 MB for a 128x128x128
            object with a 127x127 kernel. Defaults to False.
    """
    def __init__(
        self,
        psf_operator,
        distances,
        kernel_size,
        dr,
        cache_kernel: bool = False,
        ) -> None:
        self.psf_operator = psf_operator
        self.kernel_size = kernel_size
        self.distances = torch.tensor(distances).to(pytomography.dtype).to(pytomography.device)
        x = torch.arange(-(self.kernel_size-1)/2, (self.kernel_size+1)/2, 1).to(pytomography.device) * dr[0]
        y = torch.arange(-(self.kernel_size-1)/2, (self.kernel_size+1)/2, 1).to(pytomography.device) * dr[1]
        self.xv, self.yv = torch.meshgrid(x, y, indexing='xy')
        self.cache_kernel = cache_kernel
        self._use_fast_conv = None      # None until checked against the operator on the first call
        self._conv_geometry = None
        self._kernel_spectrum = None

    def _geometry(self, kernel_shape: Sequence[int], input_shape: Sequence[int]) -> tuple:
        """Padded sizes and crop used to convolve an object of shape ``input_shape`` with kernels of shape ``kernel_shape``, matching the ``padding='same'`` convention of ``fft_conv_pytorch``."""
        kernel_height, kernel_width = kernel_shape[-2], kernel_shape[-1]
        pad_height, pad_width = (kernel_height-1)/2, (kernel_width-1)/2
        pad_top, pad_bottom = int(np.floor(pad_height)), int(np.ceil(pad_height))
        pad_left, pad_right = int(np.floor(pad_width)), int(np.ceil(pad_width))
        height = input_shape[1] + pad_top + pad_bottom
        width = input_shape[2] + pad_left + pad_right
        # the one sided FFT requires an even final dimension
        width_fft = width + width % 2
        return (tuple(input_shape), height, width, width_fft, kernel_height, kernel_width, pad_top, pad_bottom, pad_left, pad_right)

    def _get_kernel_spectrum(self, geometry: tuple) -> torch.Tensor:
        """Conjugated Fourier transform of the zero padded kernel of every plane, computed once per ``geometry`` when ``cache_kernel`` is set."""
        if self._kernel_spectrum is not None:
            return self._kernel_spectrum
        _, height, _, width_fft, kernel_height, kernel_width = geometry[:6]
        kernel = self.psf_operator._get_kernel(self.xv, self.yv, self.distances)
        kernel = pad(kernel.unsqueeze(1), [0, width_fft-kernel_width, 0, height-kernel_height])
        spectrum = torch.fft.rfftn(kernel.to(torch.float32), dim=(2,3)).conj().squeeze(1).unsqueeze(0)
        if self.cache_kernel:
            self._kernel_spectrum = spectrum
        return spectrum

    def _fast_convolution(self, input: torch.Tensor, geometry: tuple) -> torch.Tensor:
        """Convolves each plane of ``input`` with the kernel of that plane, as an elementwise product of spectra."""
        _, height, width, _, kernel_height, kernel_width, pad_top, pad_bottom, pad_left, pad_right = geometry
        signal = pad(input.unsqueeze(0), [pad_left, pad_right, pad_top, pad_bottom])
        if signal.shape[-1] % 2:
            signal = pad(signal, [0,1])
        signal = torch.fft.rfftn(signal.to(torch.float32), dim=(2,3))
        output = torch.fft.irfftn(signal * self._get_kernel_spectrum(geometry), dim=(2,3))
        return output[:,:,0:(height-kernel_height+1),0:(width-kernel_width+1)].squeeze()

    def _configure_fast_conv(self, input: torch.Tensor) -> None:
        """Enables the fast convolution only if the operator exposes its kernel and the result agrees with the operator's own output."""
        self._use_fast_conv = False
        self._kernel_spectrum = None
        if not hasattr(self.psf_operator, '_get_kernel'):
            return
        try:
            kernel = self.psf_operator._get_kernel(self.xv, self.yv, self.distances)
            geometry = self._geometry(kernel.shape, input.shape)
            output = self._fast_convolution(input, geometry)
            reference = self.psf_operator(input, self.xv, self.yv, self.distances, normalize=True)
            scale = reference.abs().max()
            agrees = bool(scale == 0) or bool((output-reference).abs().max() <= 1e-4*scale)
        except Exception:
            agrees = False
        if agrees:
            self._use_fast_conv = True
            self._conv_geometry = geometry
        else:
            self._conv_geometry = None

    @torch.no_grad()
    def __call__(self, input):
        if self._use_fast_conv is None or (self._use_fast_conv and self._conv_geometry[0] != tuple(input.shape)):
            self._kernel_spectrum = None
            self._configure_fast_conv(input)
        if self._use_fast_conv:
            return self._fast_convolution(input, self._conv_geometry)
        return self.psf_operator(input,self.xv,self.yv,self.distances,normalize=True)

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
    from the detector, so that the sum along x is the blurred projection. The PSF comes from one of:

    * ``psf_meta``: a :class:`~pytomography.metadata.SPECT.SPECTPSFMeta`, e.g. from
      :func:`pytomography.io.SPECT.collimator_psf`: separable 1D Gaussian (or triangular) blurs along r and z whose
      width grows with distance. The usual choice for low and medium energy SPECT.
    * ``psf_operator``: an operator that blurs a whole object at given distances (e.g. a model of septal penetration
      and scatter from the SPECTPSFToolbox), applied plane by plane through :class:`PSF2D`.

    The blur is symmetric, so :meth:`backward` is :meth:`forward`. Give exactly one of ``psf_meta`` and
    ``psf_operator``; the kernels are made in :meth:`configure`, one set per distinct radius in ``proj_meta.radii``.

    Args:
        psf_meta (SPECTPSFMeta, optional): The PSF's width as a function of the distance. Defaults to None.
        psf_operator (Callable, optional): An operator that takes an object :math:`f` and the distances and returns the
            blurred object, such that its sum along x models the collimator-detector response. Defaults to None.
        assume_padded (bool, optional): The objects given to :meth:`forward` and :meth:`backward` are padded for
            rotation, as they are inside :class:`~pytomography.projectors.SPECT.SPECTSystemMatrix`. Defaults to True.
        cache_kernel (bool, optional): Only used with ``psf_operator``: compute the Fourier transform of the operator's
            kernel once per distinct radius and keep it on the device, rather than recomputing it at every view. See
            :class:`PSF2D` for the memory this uses. Defaults to False.

    Raises:
        ValueError: If both or neither of ``psf_meta`` and ``psf_operator`` are given.
    """
    def __init__(
        self,
        psf_meta: SPECTPSFMeta | None = None,
        psf_operator: Callable | None = None,
        assume_padded: bool = True,
        cache_kernel: bool = False,
    ) -> None:
        super(SPECTPSFTransform, self).__init__()
        if sum(arg is not None for arg in [psf_meta, psf_operator]) != 1:
            raise ValueError('SPECTPSFTransform takes exactly one of psf_meta and psf_operator')
        self.psf_meta = psf_meta
        self.psf_operator = psf_operator
        self.assume_padded = assume_padded
        self.cache_kernel = cache_kernel
        
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
            self.layers[radius] = PSF2D(self.psf_operator, distances, self.object_meta.shape[0]-1, self.object_meta.dr, cache_kernel=self.cache_kernel)
        
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