import torch
import torch.nn as nn
import numpy as np
from pytomography.utils import get_distance

def get_PSF_transform(sigma, kernel_size, delta=1e-9, device='cpu', kernel_dimensions='2D'):
    N = len(sigma)
    layer = torch.nn.Conv2d(N, N, kernel_size, groups=N, padding='same',
                            padding_mode='replicate', bias=0, device=device)
    x_grid, y_grid = torch.meshgrid(2*[torch.arange(-int(kernel_size//2), int(kernel_size//2)+1)],
                                    indexing='ij')
    x_grid = x_grid.unsqueeze(dim=0).repeat((N,1,1))
    y_grid = y_grid.unsqueeze(dim=0).repeat((N,1,1))
    sigma = torch.tensor(sigma, dtype=torch.float32).reshape((N,1,1))
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2*sigma**2 + delta))
    if kernel_dimensions=='1D':
        kernel[y_grid!=0] = 0
    kernel = kernel / kernel.sum(axis=(1,2)).reshape(N,1,1)
    layer.weight.data = kernel.unsqueeze(dim=1).to(device)
    return layer

class PSFCorrectionNet(nn.Module):
    def __init__(self, object_meta, image_meta, collimator_slope,
                 collimator_intercept, kernel_size=21, kernel_dimensions='2D', device='cpu'):
        super(PSFCorrectionNet, self).__init__()
        self.device = device
        self.object_meta = object_meta
        self.image_meta = image_meta
        self.layers = {}
        for radius in np.unique(image_meta.radii):
            sigma = self.get_sigma(radius, object_meta.dx, object_meta.shape, collimator_slope, collimator_intercept)
            self.layers[radius] = get_PSF_transform(sigma/object_meta.dx, kernel_size, kernel_dimensions=kernel_dimensions, device=self.device)
    def get_sigma(self, radius, dx, shape, collimator_slope, collimator_intercept):
        distances = get_distance(shape[0], radius, dx)
        sigma = collimator_slope * distances + collimator_intercept
        return sigma
    @torch.no_grad()
    def forward(self, object_i, i, norm_constant=None):
        return self.layers[self.image_meta.radii[i]](object_i)