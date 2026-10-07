"""Check the FDK prototype on synthetic data: a known phantom forward projected with CTGen3SystemMatrix on a GE-like
geometry (source radius 538.5 mm, DSD 946.7 mm, 984 views per rotation, 600 channels over the same fan, 32 rows), then
reconstructed with fdk_prototype.

    python run_synthetic.py --rotations 1 --pitch 0      # circular scan: exact in the central slice
    python run_synthetic.py --rotations 3 --pitch 1      # helical scan, pitch 1
"""
import argparse

import numpy as np
import torch

import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.CT import CTGen3ProjMeta
from pytomography.projectors.CT import CTGen3SystemMatrix
import fdk_prototype

parser = argparse.ArgumentParser()
parser.add_argument('--rotations', type=int, default=3)
parser.add_argument('--pitch', type=float, default=1.0)
parser.add_argument('--taper', type=int, default=4, help='detector rows over which the row window rises')
parser.add_argument('--apodization', default='hann', choices=['ram-lak', 'shepp-logan', 'hann'])
parser.add_argument('--save', default=None, help='save the reconstruction and the phantom to this .npz')
args = parser.parse_args()

views_per_rotation, ncol, nrow, DSD, rho, row_spacing = 984, 600, 32, 946.7, 538.5, 1.0987
N = views_per_rotation * args.rotations
feed = args.pitch * nrow * row_spacing * rho / DSD                         # table feed per rotation
zero = torch.zeros(N)
meta = CTGen3ProjMeta(torch.arange(N) * 2 * np.pi / views_per_rotation, torch.full((N,), rho),
                      torch.linspace(-feed * args.rotations / 2, feed * args.rotations / 2, N), zero.clone(), zero.clone(), zero.clone(),
                      torch.full((N,), (ncol + 1) / 2), torch.full((N,), (nrow + 1) / 2), 1.0239 / DSD * 888 / ncol, row_spacing, DSD,
                      (ncol, nrow), patient_position='FFS')
shape, dr, mu_w = (256, 256, 64), (1.0, 1.0, 1.0), 0.02
x, z = torch.arange(256) - 127.5, torch.arange(64) - 31.5
X, Y, Z = torch.meshgrid(x, x, z, indexing='ij')
phantom = mu_w * ((X ** 2 + Y ** 2) <= 100 ** 2).float()                                  # water cylinder, 0 HU
phantom += mu_w * 1.0 * (((X - 40) ** 2 + Y ** 2) <= 15 ** 2).float()                     # +1000 HU insert
phantom -= mu_w * 0.5 * (((X + 40) ** 2 + (Y - 20) ** 2) <= 15 ** 2).float()              # -500 HU insert
phantom += mu_w * 0.1 * ((X ** 2 + (Y + 50) ** 2 + Z ** 2) <= 10 ** 2).float()             # +100 HU sphere, 20 mm
projections = CTGen3SystemMatrix(ObjectMeta(dr=dr, shape=shape), meta).forward(phantom.to(pytomography.device)).cpu()
recon, t = fdk_prototype.fdk(projections, meta, shape, dr, taper_rows=args.taper, apodization=args.apodization, fov_radius=127.0)
recon = recon.cpu()
hu = lambda a: 1000 * (a / mu_w - 1)
k = 32
x2, y2 = X[:, :, k], Y[:, :, k]
background = ((x2 ** 2 + y2 ** 2) <= 90 ** 2) & (((x2 - 40) ** 2 + y2 ** 2) > 22 ** 2) & (((x2 + 40) ** 2 + (y2 - 20) ** 2) > 22 ** 2) & ((x2 ** 2 + (y2 + 50) ** 2) > 15 ** 2)
print(f'{args.rotations} rotation(s), pitch {args.pitch}, {N} views; filter {t["filter_s"]:.2f} s, back projection {t["backproject_s"]:.2f} s')
print(f'{"central slice":22s} {"FDK":>9s} {"truth":>9s} {"SD":>7s}')
for name, m in (('water background', background), ('+1000 HU insert', ((x2 - 40) ** 2 + y2 ** 2) <= 10 ** 2),
                ('-500 HU insert', ((x2 + 40) ** 2 + (y2 - 20) ** 2) <= 10 ** 2), ('+100 HU sphere', (x2 ** 2 + (y2 + 50) ** 2) <= 5 ** 2)):
    v = recon[:, :, k][m]
    print(f'{name:22s} {float(hu(v.mean())):+8.1f}  {float(hu(phantom[:, :, k][m].mean())):+8.1f}  {float(1000 * v.std() / mu_w):6.1f}')
# slices every view covers: within 6 mm of the plane for a circular scan, all but the ends of the phantom for a helix
k0, k1 = (26, 38) if args.pitch == 0 else (12, 52)
e = (hu(recon) - hu(phantom))[:, :, k0:k1][background[..., None].expand(-1, -1, k1 - k0)]
print(f'water background, slices {k0} to {k1 - 1}: RMS error {float(e.pow(2).mean().sqrt()):.1f} HU')
if args.save:
    np.savez_compressed(args.save, recon=recon.numpy(), phantom=phantom.numpy())
