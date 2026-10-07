"""Run the FDK prototype on a DICOM-CT-PD scan (no flying focal spot), and optionally compare it with the scanner's own
reconstruction of the same acquisition, placed with CTGen3ProjMeta.get_patient_affine.

    python run_dicom_ct_pd.py <projection folder> [--images <scanner image folder>] [--cache c145.pt] [--out fdk.npy]

TCIA LDCT-and-Projection-data case C145 (GE, chest, CC BY 4.0) is the case the README numbers come from.
"""
import argparse
import glob
import os

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from scipy import ndimage

from pytomography.io.CT import dicom_ct_pd
import fdk_prototype

parser = argparse.ArgumentParser()
parser.add_argument('projections')
parser.add_argument('--images', default=None)
parser.add_argument('--cache', default=None, help='torch file to save the loaded scan to, or load it from')
parser.add_argument('--out', default=None)
parser.add_argument('--shape', default='512,512,384')
parser.add_argument('--taper', type=int, default=4)
parser.add_argument('--apodization', default='hann')
args = parser.parse_args()

if args.cache and os.path.exists(args.cache):
    d = torch.load(args.cache, weights_only=False)
    proj, meta = d['proj'], d['meta']
else:
    paths = glob.glob(os.path.join(args.projections, '*.dcm'))
    order = [int(pydicom.dcmread(p, stop_before_pixels=True, specific_tags=['InstanceNumber']).InstanceNumber) for p in paths]
    paths = [p for _, p in sorted(zip(order, paths))]
    proj, meta = dicom_ct_pd.get_projections_and_metadata_gen3(paths)
    if args.cache:
        torch.save(dict(proj=proj, meta=meta), args.cache)
shape, dr = tuple(int(n) for n in args.shape.split(',')), (1.0, 1.0, 1.0)
recon, t = fdk_prototype.fdk(proj, meta, shape, dr, taper_rows=args.taper, apodization=args.apodization, fov_radius=250.0)
print(f'{proj.shape[0]} views: filter {t["filter_s"]:.1f} s, back projection {t["backproject_s"]:.1f} s')
if args.out:
    np.save(args.out, recon.cpu().numpy())

if args.images:
    slices = sorted((pydicom.dcmread(f) for f in glob.glob(os.path.join(args.images, '*.dcm'))), key=lambda d: float(d.ImagePositionPatient[2]))
    first = pydicom.dcmread(glob.glob(os.path.join(args.projections, '*.dcm'))[0], stop_before_pixels=True)
    mu_w = float(first[0x7041, 0x1001].value.decode().strip('\x00 '))
    hu_s = np.stack([s.pixel_array * float(s.RescaleSlope) + float(s.RescaleIntercept) for s in slices]).astype(np.float32)
    ipp = np.array([[float(v) for v in s.ImagePositionPatient] for s in slices])
    ps = [float(v) for v in slices[0].PixelSpacing]
    zs, ys, xs = ipp[:, 2], ipp[0, 1] + np.arange(hu_s.shape[1]) * ps[0], ipp[0, 0] + np.arange(hu_s.shape[2]) * ps[1]
    ks = np.arange(len(zs))[len(zs) // 5: -len(zs) // 5: 10]                      # slices away from the ends
    Ainv = np.linalg.inv(meta.get_patient_affine(type('O', (), dict(shape=shape, dr=dr))()).numpy())
    Pz, Py, Px = np.meshgrid(zs[ks], ys, xs, indexing='ij')
    P = np.stack([Px, Py, Pz, np.ones_like(Px)], -1) @ Ainv.T
    grid = torch.from_numpy(np.stack([P[..., 2] / (shape[2] - 1) * 2 - 1, P[..., 1] / (shape[1] - 1) * 2 - 1, P[..., 0] / (shape[0] - 1) * 2 - 1], -1)).float().cuda()
    ours = F.grid_sample(recon[None, None].cuda(), grid[None], align_corners=True)[0, 0].cpu().numpy()
    ours_hu = 1000 * (ours / mu_w - 1)
    valid = hu_s[ks] > -1500
    smooth = ndimage.uniform_filter(np.where(valid, hu_s[ks], -1000), size=(1, 7, 7))
    print(f'{"class":12s} {"scanner":>9s} {"FDK":>9s}')
    for name, (lo, hi), er in (('soft tissue', (10, 70), 9), ('fat', (-130, -70), 7), ('lung', (-900, -700), 7)):
        m = ndimage.binary_erosion((smooth >= lo) & (smooth < hi) & valid, structure=np.ones((1, er, er)))
        print(f'{name:12s} {hu_s[ks][m].mean():+8.1f}  {ours_hu[m].mean():+8.1f}  HU')
