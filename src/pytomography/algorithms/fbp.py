"""This module contains classes that implement filtered back projection reconstruction algorithms.
"""
import pytomography
import torch
from pytomography.metadata import SPECTObjectMeta, SPECTProjMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.utils import RampFilter

class FilteredBackProjection:
    r"""Implementation of filtered back projection reconstruction :math:`\hat{f} = \frac{\pi}{N_{\text{proj}}} \mathcal{R}^{-1}\mathcal{F}^{-1}\Pi\mathcal{F} g` where :math:`N_{\text{proj}}` is the number of projections, :math:`\mathcal{R}` is the 3D radon transform, :math:`\mathcal{F}` is the 2D Fourier transform (applied to each projection seperately), and :math:`\Pi` is the filter applied in Fourier space, which is by default the ramp filter.

        Args:
            proj (torch.Tensor): projection data :math:`g` to be reconstructed
            angles (Sequence): Angles corresponding to each projection
            filter (Callable, optional): Additional Fourier space filter (applied after Ramp Filter) used during reconstruction.
    """
    def __init__(self, proj, angles, filter=None):
        self.proj = proj
        self.object_meta = SPECTObjectMeta(dr=(1,1,1),shape=(proj.shape[2], proj.shape[2], proj.shape[3]))
        self.proj_meta = SPECTProjMeta(projection_shape=proj.shape[2:],angles=angles)
        self.filter = filter
        # Random transform equivalent to SPECT System matrix
        self.system_matrix = SPECTSystemMatrix(
            obj2obj_transforms=[],
            proj2proj_transforms=[],
            object_meta=self.object_meta,
            proj_meta=self.proj_meta)
    def __call__(self):
        """Applies reconstruction

        Returns:
            torch.tensor: Reconstructed object prediction
        """
        freq_fft = torch.fft.fftfreq(self.proj.shape[-2]).reshape((-1,1)).to(pytomography.device)
        filter_total = RampFilter()(freq_fft)
        if self.filter is not None:
            filter_total *= self.filter(freq_fft)
        proj_fft = torch.fft.fft(self.proj, axis=-2)
        proj_fft = proj_fft* filter_total
        proj_filtered = torch.fft.ifft(proj_fft, axis=-2).real
        object_prediction = self.system_matrix.backward(proj_filtered) * torch.pi / len(self.proj_meta.angles)
        return object_prediction
            
    