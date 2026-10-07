"""Reconstruct a DICOM-CT-PD scan (no flying focal spot) with the WFBP prototype on a 1 mm grid, within a GPU budget.

    python run_dicom_ct_pd.py <projection folder> [--cache scan.pt] [--out wfbp.npy] [--budget-gb 1.5]
        [--central-column-offset -0.42]          # GE scanners, see README

The image is centred as CTGen3SystemMatrix centres it, so CTGen3ProjMeta.get_patient_affine places it in patient
coordinates. run_vendor_match.py compares it with the scanner's own reconstruction.
"""
import argparse
import glob
import os

import numpy as np
import pydicom
import torch

from pytomography.io.CT import dicom_ct_pd
import fdk_prototype

parser = argparse.ArgumentParser()
parser.add_argument('projections')
parser.add_argument('--cache', default=None, help='torch file to save the loaded scan to, or load it from')
parser.add_argument('--out', default='wfbp.npy')
parser.add_argument('--size', default='512,512', help='in-plane voxels (1 mm)')
parser.add_argument('--budget-gb', type=float, default=1.5)
parser.add_argument('--central-column-offset', type=float, default=0.0)
parser.add_argument('--apodization', default='hann')
args = parser.parse_args()

if args.cache and os.path.exists(args.cache):
    d = torch.load(args.cache, weights_only=False)
    proj, meta = d['proj'], d['meta']
else:
    paths = glob.glob(os.path.join(args.projections, '*.dcm'))
    order = [int(pydicom.dcmread(p, stop_before_pixels=True, specific_tags=['InstanceNumber']).InstanceNumber) for p in paths]
    proj, meta = dicom_ct_pd.get_projections_and_metadata_gen3([p for _, p in sorted(zip(order, paths))])
    if args.cache:
        torch.save(dict(proj=proj, meta=meta), args.cache)
if args.central_column_offset:
    n_col, n_row = meta.shape
    phis_det = (torch.arange(1, n_col + 1) - (meta.detector_centers_col_idx[0] + args.central_column_offset)) * meta.col_det_spacing
    zs_det = (torch.arange(1, n_row + 1) - meta.detector_centers_row_idx[0]) * meta.row_det_spacing
    meta.phis_det, meta.zs_det = torch.meshgrid(phis_det, zs_det, indexing='ij')
nx, ny = (int(v) for v in args.size.split(','))
extent = float(meta.source_zs.max() - meta.source_zs.min())
nz = int(np.ceil(extent + 20))
X, Y = torch.meshgrid(torch.arange(nx) - (nx - 1) / 2.0, torch.arange(ny) - (ny - 1) / 2.0, indexing='ij')
Z = np.arange(nz) - (nz - 1) / 2.0
free, _ = torch.cuda.mem_get_info()
print(f'{proj.shape[0]} views -> {nx} x {ny} x {nz} at 1 mm; output {nx * ny * nz * 4 / 1e9:.2f} GB on the GPU; '
      f'budget {fdk_prototype.gpu_budget(args.budget_gb * 1e9) / 1e9:.2f} GB of {free / 1e9:.1f} GB free', flush=True)
recon, t = fdk_prototype.wfbp(proj, meta, X.float(), Y.float(), Z, apodization=args.apodization, budget=args.budget_gb * 1e9)
print({k: round(v, 3) for k, v in t.items()}, flush=True)
np.save(args.out, recon.cpu().numpy())
