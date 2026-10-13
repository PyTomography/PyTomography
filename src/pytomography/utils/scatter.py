"""Energy window scatter estimates of SPECT projections, and their smoothing.

:func:`pytomography.io.SPECT.energy_window_scatter` computes these estimates from the energy windows the readers give.
"""
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
    N_sigmas: int = 3,
    FOV_mask: torch.tensor | None = None,
) -> torch.Tensor:
    """Smooths SPECT projections with Gaussians along the views, r and z, normalised at the edge of the field of view.

    Each dimension whose standard deviation is positive is convolved with a Gaussian kernel truncated at ``N_sigmas``
    standard deviations, wrapping around along the views and repeating the edge values along r and z. The same
    smoothing is applied to the field of view mask (all ones without ``FOV_mask``), and the smoothed projections are
    divided by the smoothed mask and multiplied by the mask: values near the edge of the field of view are not pulled
    towards the zeros outside it, and the result is 0 outside it. The view spacing is the difference of the first two
    angles.

    Args:
        scatter (torch.Tensor): Projections [θ, r, z].
        proj_meta (ProjMeta): Their metadata (:class:`~pytomography.metadata.SPECT.SPECTProjMeta`), for the pixel size
            ``dr`` and the ``angles``.
        sigma_theta (float, optional): Standard deviation of the smoothing along the views, in degrees. Defaults to 0.
        sigma_r (float, optional): Standard deviation of the smoothing along r, in mm. Defaults to 0.
        sigma_z (float, optional): Standard deviation of the smoothing along z, in mm. Defaults to 0.
        N_sigmas (int, optional): Half-width of the kernels, in standard deviations. Defaults to 3.
        FOV_mask (torch.Tensor | None, optional): Mask [r, z] of the detector's field of view. Defaults to None.

    Returns:
        torch.Tensor: The smoothed projections [θ, r, z].
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
    norm_projections = torch.ones_like(scatter)
    if FOV_mask is not None:
        norm_projections = norm_projections * FOV_mask
    norm_projections_blur = torch.clone(norm_projections)
    modes = ['circular', 'replicate', 'replicate']
    for i in range(3):
        if sigmas[i]>pytomography.delta:
            k = get_1d_gaussian_kernel(sigmas[i]/dS[i], ksize[i], modes[i]).to(pytomography.device)
            scatter = scatter.swapaxes(i,2)
            scatter = k(scatter.flatten(end_dim=-2).unsqueeze(1)).reshape(scatter.shape)
            scatter = scatter.swapaxes(i,2)
            norm_projections_blur = norm_projections_blur.swapaxes(i,2)
            norm_projections_blur = k(norm_projections_blur.flatten(end_dim=-2).unsqueeze(1)).reshape(norm_projections_blur.shape)
            norm_projections_blur = norm_projections_blur.swapaxes(i,2)
    
    return scatter*norm_projections / (norm_projections_blur + pytomography.delta)

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
    return_scatter_variance_estimate: bool = False,
    fov_mask: torch.tensor | None = None
    ) -> torch.Tensor | Sequence[torch.Tensor]:
    r"""Dual or triple energy window scatter estimate of the photopeak window of SPECT projections, optionally smoothed.

    The estimate is

    .. math::

        s = \left(w_l \frac{p_l}{\Delta_l} + w_u \frac{p_u}{\Delta_u}\right) \Delta_p,

    with :math:`p_l` and :math:`p_u` the projections of the lower and upper scatter windows, :math:`\Delta_l`,
    :math:`\Delta_u` and :math:`\Delta_p` the widths of the lower, upper and photopeak windows, and :math:`w_l` and
    :math:`w_u` the weights (0.5 each for the triple energy window method). Without an upper window (the dual energy
    window method) its term is left out. If any of ``sigma_theta``, ``sigma_r`` and ``sigma_z`` is positive, the
    estimate is then smoothed with :func:`get_smoothed_scatter`. :func:`pytomography.io.SPECT.energy_window_scatter`
    calls this function with the widths of ``proj_meta.energy_windows``.

    Args:
        projection_lower (torch.Tensor): Projections of the lower scatter window [θ, r, z] (any shape without
            smoothing).
        projection_upper (torch.Tensor | None): Projections of the upper scatter window, of the same shape, or None for
            the dual energy window method.
        width_lower (float): Width of the lower window, in keV.
        width_upper (float | None): Width of the upper window, in keV, or None without an upper window.
        width_peak (float): Width of the photopeak window, in keV.
        weighting_lower (float, optional): Weight :math:`w_l` of the lower window. Defaults to 0.5.
        weighting_upper (float, optional): Weight :math:`w_u` of the upper window. Defaults to 0.5.
        proj_meta (SPECTProjMeta | None, optional): Metadata of the projections, needed only to smooth (for the pixel
            size and the angles). Defaults to None.
        sigma_theta (float, optional): Standard deviation of the smoothing along the views, in degrees. Defaults to 0
            (none).
        sigma_r (float, optional): Standard deviation of the smoothing along r, in mm. Defaults to 0 (none).
        sigma_z (float, optional): Standard deviation of the smoothing along z, in mm. Defaults to 0 (none).
        N_sigmas (int, optional): Half-width of the smoothing kernels, in standard deviations. Defaults to 3.
        return_scatter_variance_estimate (bool, optional): Also return the variance of the estimate, as a function (see
            Returns). Defaults to False.
        fov_mask (torch.Tensor | None, optional): Mask [r, z] of the detector's field of view
            (:func:`pytomography.io.SPECT.fov_mask`), so that smoothing does not blur across its edge. Defaults to None.

    Returns:
        torch.Tensor | Sequence[torch.Tensor]: The scatter estimate [θ, r, z]. With
        ``return_scatter_variance_estimate``, the tuple (estimate, variance), where variance is a function that maps a
        tensor :math:`x` [θ, r, z] to
        :math:`F(x)\,v\,F(x)` (element by element), with :math:`F` the smoothing (the identity without smoothing) and
        :math:`v = (w_l \Delta_p / \Delta_l)^2 p_l + (w_u \Delta_p / \Delta_u)^2 p_u` the variance of the unsmoothed
        estimate.

    Raises:
        ValueError: Smoothing is asked for without ``proj_meta``.
    """
    projection_upper = 0 if projection_upper is None else projection_upper
    width_upper = 1 if width_upper is None else width_upper
    scatter_estimate = (projection_lower/width_lower*weighting_lower + projection_upper/width_upper*weighting_upper)*width_peak
    
    if (sigma_r>0)+(sigma_theta>0)+(sigma_z>0):
        if proj_meta is None:
            raise ValueError("proj_meta must be provided if scatter is to be smoothed")
        scatter_estimate = get_smoothed_scatter(scatter_estimate, proj_meta, sigma_theta, sigma_r, sigma_z, N_sigmas, fov_mask)
    
    if return_scatter_variance_estimate:
        scatter_variance_estimate_diag = (width_peak / width_lower * weighting_lower) ** 2 * projection_lower + (width_peak / width_upper *weighting_upper) ** 2 * projection_upper
        # Returns an operator F^TsF where F is the scatter blurring kernel
        if (sigma_r>0)+(sigma_theta>0)+(sigma_z>0):
            def scatter_variance_estimate(x):
                x_smoothed = get_smoothed_scatter(x, proj_meta, sigma_theta, sigma_r, sigma_z, N_sigmas, fov_mask)
                return x_smoothed * scatter_variance_estimate_diag * x_smoothed
        else:
            scatter_variance_estimate = lambda x: x * scatter_variance_estimate_diag * x
        return scatter_estimate, scatter_variance_estimate
    else:
        return scatter_estimate