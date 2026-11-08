"""Subsampling SPECT projections and attenuation maps, to reconstruct quickly on a coarser grid."""
from __future__ import annotations
import torch
from torch.nn.functional import avg_pool2d, avg_pool3d


def subsample_projections(
    projections: torch.Tensor,
    N_pixel: int,
    N_angle: int,
    N_angle_start: int = 0
    ) -> torch.Tensor:
    """Subsamples SPECT projections by adding up blocks of pixels and keeping every ``N_angle``-th view.

    Blocks of ``N_pixel`` x ``N_pixel`` pixels (r, z) are summed, so the counts are kept (a remainder along r or z is
    dropped). The metadata that goes with the result is ``object_meta.subsample(N_pixel)`` and
    ``proj_meta.subsample(N_pixel, N_angle, N_angle_start)``.

    Args:
        projections (torch.Tensor): Projections to subsample, [θ, r, z].
        N_pixel (int): Pixel reduction factor (1 means no reduction).
        N_angle (int): Angle reduction factor (1 means no reduction).
        N_angle_start (int, optional): Index of the first view kept. Defaults to 0.

    Returns:
        torch.Tensor: The subsampled projections, [θ / N_angle, r / N_pixel, z / N_pixel].
    """
    result = avg_pool2d(projections.unsqueeze(0), N_pixel).squeeze(0) * N_pixel**2
    return result[N_angle_start::N_angle]


def subsample_amap(amap: torch.Tensor, N: int) -> torch.Tensor:
    """Subsamples a 3D attenuation map by averaging blocks of N x N x N voxels.

    The values (1/mm) are averaged, not summed; a remainder along any axis is dropped. The grid that goes with the
    result is ``object_meta.subsample(N)``.

    Args:
        amap (torch.Tensor): The attenuation map, [Lx, Ly, Lz].
        N (int): Factor to reduce by.

    Returns:
        torch.Tensor: The subsampled map, [Lx / N, Ly / N, Lz / N].
    """
    return avg_pool3d(amap.unsqueeze(0).unsqueeze(0), N).squeeze()
