from __future__ import annotations
from collections.abc import Sequence
import torch

def compute_EW_scatter(
    projection_lower: torch.Tensor,
    projection_upper: torch.Tensor | None,
    width_lower: float,
    width_upper: float | None,
    width_peak: float,
    weighting_lower: float = 0.5,
    weighting_upper: float = 0.5,
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
    if return_scatter_variance_estimate:
        scatter_variance_estimate = (width_peak / width_lower * weighting_lower) ** 2 * projection_lower + (width_peak / width_upper *weighting_upper) ** 2 * projection_upper
        return scatter_estimate, scatter_variance_estimate
    else:
        return scatter_estimate