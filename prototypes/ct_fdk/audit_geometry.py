"""Check how a DICOM-CT-PD scan's geometry is read, by data consistency: reconstruct it with OS-SART (2 iterations x 20
subsets) under the geometry as CTGen3ProjMeta reads it and under alternative conventions, and compare the residual
|Hf - g| / |g| over every view. The convention that matches the scanner reproduces its own data best. A rotation of the
whole geometry cannot be found this way (the data stay consistent); run_vendor_match.py finds that one.

    python audit_geometry.py <projection folder> "as read" "columns reversed" "central column -0.5" ... [--cache scan.pt]

Variants: "as read", "no focal spot shifts", "z shift sign flipped", "radial shift sign flipped", "angular shift sign
flipped", "columns reversed", "rows reversed", "central column <offset>", "central row <offset>".
GPU memory: the projector's ray coordinates (about 28 bytes per ray per call, --ray-cap-log2) plus about six object
volumes; 2.6 GB for a 512 x 512 x 384 object at 1 mm with 2^23 rays per call, 1.2 GB for 384 x 384 x 272.
"""
import argparse
import copy
import glob
import json
import os
import time

import numpy as np
import pydicom
import torch

import pytomography.projectors.CT.ct_gen3_system_matrix as gen3
from pytomography.algorithms import SART
from pytomography.io.CT import dicom_ct_pd
from pytomography.metadata import ObjectMeta
from pytomography.projectors.CT import CTGen3SystemMatrix

parser = argparse.ArgumentParser()
parser.add_argument('projections')
parser.add_argument('variants', nargs='+')
parser.add_argument('--cache', default=None)
parser.add_argument('--size', default='512,512', help='in-plane voxels')
parser.add_argument('--voxel', type=float, default=1.0)
parser.add_argument('--ray-cap-log2', type=int, default=23)
parser.add_argument('--out', default='geometry_audit.json')
args = parser.parse_args()
gen3._MAX_RAYS_PER_CALL = 2 ** args.ray_cap_log2

if args.cache and os.path.exists(args.cache):
    d = torch.load(args.cache, weights_only=False)
    proj, base = d['proj'], d['meta']
else:
    paths = glob.glob(os.path.join(args.projections, '*.dcm'))
    order = [int(pydicom.dcmread(p, stop_before_pixels=True, specific_tags=['InstanceNumber']).InstanceNumber) for p in paths]
    proj, base = dicom_ct_pd.get_projections_and_metadata_gen3([p for _, p in sorted(zip(order, paths))])
    if args.cache:
        torch.save(dict(proj=proj, meta=base), args.cache)
nx, ny = (int(v) for v in args.size.split(','))
nz = int(np.ceil((float(base.source_zs.max() - base.source_zs.min()) + 20) / args.voxel))
shape, dr = (nx, ny, nz), (args.voxel,) * 3


def variant(v):
    m, g = copy.copy(base), proj
    rho, phi, zc = m.source_rhos, m.source_phis, m.source_zs
    dphi, dz, drho = m.source_phi_offsets, m.source_z_offsets, m.source_rho_offsets
    spots = lambda r, p, z: torch.stack([r * torch.cos(p), r * torch.sin(p), z], dim=-1)
    if v == 'as read':
        pass
    elif v == 'no focal spot shifts':
        m.source_focal_spots = m.source_focal_centers.clone()
    elif v == 'z shift sign flipped':
        m.source_focal_spots = spots(rho + drho, phi + dphi, zc - dz)
    elif v == 'radial shift sign flipped':
        m.source_focal_spots = spots(rho - drho, phi + dphi, zc + dz)
    elif v == 'angular shift sign flipped':
        m.source_focal_spots = spots(rho + drho, phi - dphi, zc + dz)
    elif v == 'columns reversed':
        g = proj.flip(1).contiguous()
    elif v == 'rows reversed':
        g = proj.flip(2).contiguous()
    elif v.startswith('central column ') or v.startswith('central row '):
        shift = float(v.split()[-1])
        dc, drow = (shift, 0.0) if v.startswith('central column ') else (0.0, shift)
        n_col, n_row = m.shape
        phis_det = (torch.arange(1, n_col + 1) - (m.detector_centers_col_idx[0] + dc)) * m.col_det_spacing
        zs_det = (torch.arange(1, n_row + 1) - (m.detector_centers_row_idx[0] + drow)) * m.row_det_spacing
        m.phis_det, m.zs_det = torch.meshgrid(phis_det, zs_det, indexing='ij')
    else:
        raise ValueError(v)
    return m, g


free, _ = torch.cuda.mem_get_info()
print(f'object {shape} at {args.voxel} mm, {2 ** args.ray_cap_log2:,} rays per call; GPU {free / 1e9:.1f} GB free', flush=True)
res = json.load(open(args.out)) if os.path.exists(args.out) else {}
for v in args.variants:
    m, g = variant(v)
    sm = CTGen3SystemMatrix(ObjectMeta(dr=dr, shape=shape), m, N_splits=2, device='cpu')
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); mem0 = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    recon = SART(sm, g)(n_iters=2, n_subsets=20)
    num = den = 0.0
    for k in range(20):
        fp = sm.forward(recon, k)
        gk = sm.get_projection_subset(g, k)
        num += float(((fp - gk) ** 2).double().sum())
        den += float((gk ** 2).double().sum())
    res[v] = dict(rel_residual=(num / den) ** 0.5, seconds=time.perf_counter() - t0, gpu_peak_GB=(torch.cuda.max_memory_allocated() - mem0) / 1e9)
    print(f'{v:32s} |Hf - g| / |g| = {res[v]["rel_residual"]:.5f}  ({res[v]["seconds"]:.0f} s, GPU peak {res[v]["gpu_peak_GB"]:.2f} GB)', flush=True)
    json.dump(res, open(args.out, 'w'), indent=1)
    del recon, sm
    torch.cuda.empty_cache()
