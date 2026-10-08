"""Reconstruct a DICOM-CT-PD scan (no flying focal spot) with the WFBP prototype on a 1 mm grid, within a GPU budget.

    python run_dicom_ct_pd.py <projection folder> [--cache scan.pt] [--out wfbp.npy] [--budget-gb 1.5]
        [--central-column-offset -0.42]          # GE scanners, see README
        [--low-signal 30]                         # filter photon-starved rays, see low_signal.py
        [--channel-correction fit.json]           # per-channel scale fitted by channel_correction.py
        [--table-feed-from-pitch]                 # focal spot z on the scanner's nominal table feed, see README

The image is centred as CTGen3SystemMatrix centres it, so CTGen3ProjMeta.get_patient_affine places it in patient
coordinates. run_vendor_match.py compares it with the scanner's own reconstruction.
"""
import argparse
import glob
import json
import os

import numpy as np
import pydicom
import torch

from pytomography.io.CT import dicom_ct_pd
import fdk_prototype
import channel_correction
import low_signal

parser = argparse.ArgumentParser()
parser.add_argument('projections')
parser.add_argument('--cache', default=None, help='torch file to save the loaded scan to, or load it from')
parser.add_argument('--out', default='wfbp.npy')
parser.add_argument('--size', default='512,512', help='in-plane voxels (1 mm)')
parser.add_argument('--budget-gb', type=float, default=1.5)
parser.add_argument('--central-column-offset', type=float, default=0.0)
parser.add_argument('--apodization', default='hann')
parser.add_argument('--low-signal', type=float, default=0.0, help='filter photon-starved rays to about this many photons (0: off)')
parser.add_argument('--channel-correction', default=None, help='JSON written by channel_correction.py')
parser.add_argument('--table-feed-from-pitch', action='store_true', help="rescale the focal spot z to pitch x collimation (the scanner's own feed)")
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
if args.low_signal:
    n0 = low_signal.photon_statistics(low_signal.acquisition_order(args.projections))
    filtered, frac = low_signal.filter_low_signal(proj.numpy(), n0, args.low_signal)
    proj = torch.from_numpy(filtered)
    print(f'photon-starved rays filtered to about {args.low_signal:g} photons: {100 * frac:.3f}% of rays changed', flush=True)
if args.channel_correction:
    proj = torch.from_numpy(channel_correction.apply_channel_scale(proj.numpy(), meta, json.load(open(args.channel_correction))))
    print(f'per-channel correction from {args.channel_correction}', flush=True)
if args.table_feed_from_pitch:
    head = pydicom.dcmread(glob.glob(os.path.join(args.projections, '*.dcm'))[0], stop_before_pixels=True)
    feed = fdk_prototype.nominal_table_feed(meta, float(head.SpiralPitchFactor))
    k = fdk_prototype.rescale_table_feed(meta, feed)
    print(f'focal spot z rescaled by {k:.6f} about the last view: table feed {feed:.3f} mm per rotation from the pitch', flush=True)
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
