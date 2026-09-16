"""This module contains classes that implement filtered back projection reconstruction algorithms.
"""
from __future__ import annotations
import pytomography
import torch
from pytomography.projectors import SystemMatrix
from pytomography.utils import RampFilter

class FilteredBackProjection:
    r"""Implementation of filtered back projection reconstruction :math:`\hat{f} = \frac{\pi}{N_{\text{proj}}} \mathcal{R}^{-1}\mathcal{F}^{-1}\Pi\mathcal{F} g` where :math:`N_{\text{proj}}` is the number of projections, :math:`\mathcal{R}` is the 3D radon transform, :math:`\mathcal{F}` is the 2D Fourier transform (applied to each projection seperately), and :math:`\Pi` is the filter applied in Fourier space, which is by default the ramp filter.

        Args:
            projections (torch.Tensor): projection data :math:`g` to be reconstructed
            system_matrix (SystemMatrix): system matrix for the imaging system. In FBP, phenomena such as attenuation and PSF should not be implemented in the system matrix
            filter (Callable, optional): Additional Fourier space filter (applied after Ramp Filter) used during reconstruction.
    """
    def __init__(
        self,
        projections: torch.Tensor,
        system_matrix: SystemMatrix,
        filter=RampFilter
        ) -> None:
        self.system_matrix = system_matrix
        self.filter = filter
        # Random transform equivalent to SPECT System matrix
    def __call__(self, projections):
        """Applies reconstruction

        Returns:
            torch.tensor: Reconstructed object prediction
        """
        freq_fft = torch.fft.fftfreq(projections.shape[-2]).reshape((-1,1)).to(pytomography.device) # only works for SPECT
        filter_total = self.filter()(freq_fft)
        proj_fft = torch.fft.fft(self.proj, axis=-2)
        proj_fft = proj_fft* filter_total
        proj_filtered = torch.fft.ifft(proj_fft, axis=-2).real
        object_prediction = self.system_matrix.backward(proj_filtered) * torch.pi / len(self.system_matrix.proj_meta.shape[0]) # assumes the "angle" index is the first of the system matrix
        return object_prediction
            
    