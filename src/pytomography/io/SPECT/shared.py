from __future__ import annotations
import torch
from torch.nn.functional import avg_pool2d, avg_pool3d


def subsample_projections(
    projections: torch.Tensor,
    N_pixel: int,
    N_angle: int,
    N_angle_start: int = 0
    ) -> torch.Tensor:
    """Subsamples SPECT projections [θ, r, z] by adding up N_pixel x N_pixel pixels and keeping every N_angle-th view.
    The grid that goes with them is ``object_meta.subsample(N_pixel)`` and
    ``proj_meta.subsample(N_pixel, N_angle, N_angle_start)``.

    Args:
        projections (torch.Tensor): Projections to subsample, [θ, r, z].
        N_pixel (int): Pixel reduction factor (1 means no reduction)
        N_angle (int): Angle reduction factor (1 means no reduction)
        N_angle_start (int): Angle index to start at. Defaults to 0.

    Returns:
        torch.Tensor: Subsampled projections
    """
    result = avg_pool2d(projections.unsqueeze(0), N_pixel).squeeze(0) * N_pixel**2
    return result[N_angle_start::N_angle]


def subsample_amap(amap: torch.Tensor, N: int) -> torch.Tensor:
    """Subsamples 3D attenuation map by averaging over N x N x N regions

    Args:
        amap (torch.Tensor): Original attenuation map
        N (int): Factor to reduce by

    Returns:
        torch.Tensor: Subsampled attenuation map
    """
    return avg_pool3d(amap.unsqueeze(0).unsqueeze(0), N).squeeze()
