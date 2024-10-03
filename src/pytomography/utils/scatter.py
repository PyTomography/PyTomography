from __future__ import annotations
from collections.abc import Sequence
from . import get_1d_gaussian_kernel
from pytomography.metadata import ProjMeta
import numpy as np
import pytomography
import torch
from pytomography.utils import get_1d_gaussian_kernel

@torch.no_grad()
def get_smoothed_scatter(
    scatter: torch.Tensor,
    proj_meta: ProjMeta,
    sigma_theta: float = 0,
    sigma_r: float = 0,
    sigma_z: float = 0,
    N_sigmas: int = 3
) -> torch.Tensor:
    """Smooths SPECT projection metadata

    Args:
        scatter (torch.Tensor): Input projection data
        proj_meta (ProjMeta): Projection metadata
        sigma_theta (float, optional): Smoothing in theta (specified in degrees). Defaults to 0.
        sigma_r (float, optional): Smoothing in r (specified in cm). Defaults to 0.
        sigma_z (float, optional): Smoothing in z (specified in cm). Defaults to 0.
        N_sigmas (int, optional): Number of sigmas to include in the smoothing kernel. Defaults to 3.

    Returns:
        torch.Tensor: Smoothed projections
    """
    # spacing
    dr, dz = proj_meta.dr
    dtheta = torch.diff(proj_meta.angles)[0].item()
    dS = [dtheta, dr, dz]
    # kernel size
    kernel_size_theta = 2 * int(np.ceil(N_sigmas * sigma_theta / dtheta)) + 1
    kernel_size_r = 2 * int(np.ceil(N_sigmas * sigma_r / dr)) + 1
    kernel_size_z = 2 * int(np.ceil(N_sigmas * sigma_z / dz)) + 1
    ksize = [kernel_size_theta, kernel_size_r, kernel_size_z]
    sigmas = [sigma_theta, sigma_r, sigma_z]
    # modes
    modes = ['circular', 'replicate', 'replicate']
    for i in range(3):
        if sigmas[i]>pytomography.delta:
            k = get_1d_gaussian_kernel(sigmas[i]/dS[i], ksize[i], modes[i]).to(pytomography.device)
            scatter = scatter.swapaxes(i,2)
            scatter = k(scatter.flatten(end_dim=-2).unsqueeze(1)).reshape(scatter.shape)
            scatter = scatter.swapaxes(i,2)
    return scatter

def compute_EW_scatter(
    projection_lower: torch.Tensor,
    projection_upper: torch.Tensor | None,
    width_lower: float,
    width_upper: float | None,
    width_peak: float,
    weighting_lower: float = 0.5,
    weighting_upper: float = 0.5,
    proj_meta = None,
    sigma_theta: float = 0,
    sigma_r: float = 0,
    sigma_z: float = 0,
    N_sigmas: int = 3,
    return_scatter_variance_estimate: bool = False
    ) -> torch.Tensor | Sequence[torch.Tensor]:
    """Computes triple energy window estimate from lower and upper scatter projections as well as window widths

    Args:
        projection_lower (torch.Tensor): Projection data corresponding to lower energy window
        projection_upper (torch.Tensor): Projection data corresponding to upper energy window
        width_lower (float): Width of lower energy window
        width_upper (float): Width of upper energy window
        width_peak (float): Width of peak energy window
        return_scatter_variance_estimate (bool, optional): Return scatter variance estimate. Defaults to False.

    Returns:
        torch.Tensor | Sequence[torch.Tensor]: Scatter estimate (and scatter variance estimate if specified)
    """
    projection_upper = 0 if projection_upper is None else projection_upper
    width_upper = 1 if width_upper is None else width_upper
    scatter_estimate = (projection_lower/width_lower*weighting_lower + projection_upper/width_upper*weighting_upper)*width_peak
    
    if (sigma_r>0)+(sigma_theta>0)+(sigma_z>0):
        if proj_meta is None:
            raise ValueError("proj_meta must be provided if scatter is to be smoothed")
        scatter_estimate = get_smoothed_scatter(scatter_estimate, proj_meta, sigma_theta, sigma_r, sigma_z, N_sigmas)
    
    if return_scatter_variance_estimate:
        scatter_variance_estimate_diag = (width_peak / width_lower * weighting_lower) ** 2 * projection_lower + (width_peak / width_upper *weighting_upper) ** 2 * projection_upper
        # Returns an operator F^TsF where F is the scatter blurring kernel
        if (sigma_r>0)+(sigma_theta>0)+(sigma_z>0):
            def scatter_variance_estimate(x):
                x_smoothed = get_smoothed_scatter(x, proj_meta, sigma_theta, sigma_r, sigma_z, N_sigmas)
                return x_smoothed * scatter_variance_estimate_diag * x_smoothed
        else:
            scatter_variance_estimate = lambda x: x * scatter_variance_estimate_diag * x
        return scatter_estimate, scatter_variance_estimate
    else:
        return scatter_estimate