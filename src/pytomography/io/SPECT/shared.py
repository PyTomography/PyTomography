from __future__ import annotations
from collections.abc import Sequence
import torch
from torch.nn.functional import avg_pool2d, avg_pool3d
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta

def subsample_projections(
    projections: torch.Tensor,
    N_pixel: int,
    N_angle: int,
    N_angle_start: int = 0
    ) -> torch.Tensor:
    """Subsamples SPECT projections by averaging over N_pixel x N_pixel pixel regions and by removing certain angles

    Args:
        projections (torch.Tensor): Projections to subsample
        N_pixel (int): Pixel reduction factor (1 means no reduction)
        N_angle (int): Angle reduction factor (1 means no reduction)
        N_angle_start (int): Angle index to start at. Defaults to 0.

    Returns:
        torch.Tensor: Subsampled projections
    """
    result = avg_pool2d(projections.unsqueeze(0), N_pixel).squeeze() * N_pixel**2
    return result[N_angle_start::N_angle]

def subsample_metadata(
    object_meta: SPECTObjectMeta,
    proj_meta: SPECTProjMeta,
    N_pixel: int = 1,
    N_angle: int = 1,
    N_angle_start: int = 0
):
    """Subsample SPECT metadata with the specified pixel and angle reduction factors

    Args:
        object_meta (SPECTObjectMeta): Original object metadata
        proj_meta (SPECTProjMeta): Original projection metadata
        N_pixel (int): Pixel reduction factor (1 means no reduction). Defaults to 1.
        N_angle (int): Angle reduction factor (1 means no reduction). Defaults to 1.
        N_angle_start (int): Angle index to start at. Defaults to 0.

    Returns:
        Sequence: Modified object metadata, modified projection metadata
    """
    object_meta.dr = tuple([N_pixel*dri for dri in object_meta.dr])
    object_meta.shape = tuple([int(s/N_pixel) for s in object_meta.shape])
    object_meta.compute_padded_shape()
    object_meta.dx*=N_pixel
    object_meta.dy*=N_pixel
    object_meta.dz*=N_pixel
    proj_meta.dr = tuple([N_pixel*dri for dri in proj_meta.dr])
    proj_meta.shape = (int(proj_meta.shape[0]/N_angle), int(proj_meta.shape[1]/N_pixel), int(proj_meta.shape[2]/N_pixel))
    proj_meta.compute_padded_shape()
    proj_meta.radii = proj_meta.radii[N_angle_start::N_angle]
    proj_meta.angles = proj_meta.angles[N_angle_start::N_angle]
    proj_meta.num_projections = int(proj_meta.num_projections/N_angle)
    return object_meta, proj_meta

def subsample_projections_and_modify_metadata(
    object_meta: SPECTObjectMeta,
    proj_meta: SPECTProjMeta,
    projections: torch.Tensor,
    N_pixel: int = 1,
    N_angle: int = 1,
    N_angle_start: int = 0
    ) -> Sequence[SPECTObjectMeta, SPECTProjMeta, torch.Tensor]:
    """Subsamples SPECT projection and modifies metadata accordingly

    Args:
        object_meta (ObjectMeta): Object metadata
        proj_meta (SPECTProjMeta): Projection metadata
        projections (torch.Tensor): Projections to subsample
        N_pixel (int): Pixel reduction factor (1 means no reduction). Defaults to 1.
        N_angle (int): Angle reduction factor (1 means no reduction). Defaults to 1.
        N_angle_start (int): Angle index to start at. Defaults to 0.

    Returns:
        Sequence: Modified object metadata, modified projection metadata, subsampled projections
    """
    object_meta, proj_meta = subsample_metadata(object_meta, proj_meta, N_pixel, N_angle, N_angle_start)
    projections_processed = []
    for projections_batch in projections:
        projections_batch = projections_batch
        projections_batch = subsample_projections(projections_batch, N_pixel, N_angle, N_angle_start)
        projections_processed.append(projections_batch)
    projections_processed = torch.stack(projections_processed, dim=0)
    return object_meta, proj_meta, projections_processed

def subsample_amap(amap: torch.Tensor, N: int) -> torch.Tensor:
    """Subsamples 3D attenuation map by averaging over N x N x N regions

    Args:
        amap (torch.Tensor): Original attenuation map
        N (int): Factor to reduce by

    Returns:
        torch.Tensor: Subsampled attenuation map
    """
    return avg_pool3d(amap.unsqueeze(0).unsqueeze(0), N).squeeze()