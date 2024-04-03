from __future__ import annotations
import torch
from torch.nn.functional import avg_pool2d, avg_pool3d

def subsample_projections(projections, N_pixel, N_angle, N_angle_start):
    result = avg_pool2d(projections, N_pixel) * N_pixel**2
    return result[:,N_angle_start::N_angle]

def reduce_projections(object_meta, proj_meta, projections, N_pixel=1, N_angle=1, N_angle_start=0):
    object_meta.dr = tuple([N_pixel*dri for dri in object_meta.dr])
    object_meta.shape = tuple([int(s/N_pixel) for s in object_meta.shape])
    object_meta.compute_padded_shape()
    object_meta.dx*=N_pixel
    object_meta.dy*=N_pixel
    proj_meta.dr = tuple([N_pixel*dri for dri in proj_meta.dr])
    proj_meta.shape = (int(proj_meta.shape[0]/N_angle), int(proj_meta.shape[1]/N_pixel), int(proj_meta.shape[2]/N_pixel))
    proj_meta.compute_padded_shape()
    proj_meta.radii = proj_meta.radii[N_angle_start::N_angle]
    proj_meta.angles = proj_meta.angles[N_angle_start::N_angle]
    proj_meta.num_projections = int(proj_meta.num_projections/N_angle)
    projections_processed = []
    for projections_batch in projections:
        projections_batch = projections_batch.unsqueeze(0)
        projections_batch = subsample_projections(projections_batch, N_pixel, N_angle, N_angle_start)
        projections_processed.append(projections_batch)
    projections_processed = torch.cat(projections_processed, dim=0)
    return object_meta, proj_meta, projections_processed

def reduce_amap(amap, N):
    return avg_pool3d(amap.unsqueeze(1), N).squeeze(1)