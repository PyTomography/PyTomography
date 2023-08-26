"""This module contains classes that implement filtered back projection reconstruction algorithms.
"""
import pytomography
import torch
from pytomography.metadata import SPECTObjectMeta, SPECTImageMeta
from pytomography.projections.SPECT import SPECTSystemMatrix
from pytomography.utils import RampFilter

class FilteredBackProjection:
    r"""Implementation of filtered back projection reconstruction :math:`\hat{f} = \frac{\pi}{N_{\text{proj}}} \mathcal{R}^{-1}\mathcal{F}^{-1}\Pi\mathcal{F} g` where :math:`N_{\text{proj}}` is the number of projections, :math:`\mathcal{R}` is the 3D radon transform, :math:`\mathcal{F}` is the 2D Fourier transform (applied to each projection seperately), and :math:`\Pi` is the filter applied in Fourier space, which is by default the ramp filter.

        Args:
            image (torch.Tensor): image data :math:`g` to be reconstructed
            angles (Sequence): Angles corresponding to each projection in the image/
            filter (Callable, optional): Additional Fourier space filter (applied after Ramp Filter) used during reconstruction.
    """
    def __init__(self, image, angles, filter=None):
        self.image = image
        self.object_meta = SPECTObjectMeta(dr=(1,1,1),shape=(image.shape[2], image.shape[2], image.shape[3]))
        self.image_meta = SPECTImageMeta(projection_shape=image.shape[2:],angles=angles)
        self.filter = filter
        # Random transform equivalent to SPECT System matrix
        self.system_matrix = SPECTSystemMatrix(
            obj2obj_transforms=[],
            im2im_transforms=[],
            object_meta=self.object_meta,
            image_meta=self.image_meta)
    def __call__(self):
        """Applies reconstruction

        Returns:
            torch.tensor: Reconstructed object prediction
        """
        freq_fft = torch.fft.fftfreq(self.image.shape[-2]).reshape((-1,1)).to(pytomography.device)
        filter_total = RampFilter()(freq_fft)
        if self.filter is not None:
            filter_total *= self.filter(freq_fft)
        image_fft = torch.fft.fft(self.image, axis=-2)
        image_fft = image_fft* filter_total
        image_filtered = torch.fft.ifft(image_fft, axis=-2).real
        object_prediction = self.system_matrix.backward(image_filtered) * torch.pi / len(self.image_meta.angles)
        return object_prediction
            
    