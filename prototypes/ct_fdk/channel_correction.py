"""A per-channel scale of the line integrals that makes WFBP of DICOM-CT-PD data match the scanner's own images: the
form a bowtie-dependent (per detector column) beam hardening calibration takes. CPU only.

For C145 (GE), the difference between the scanner's image and ours, projected along parallel rays, is not a function of
path length (water beam hardening), of the path through bone, or of exp(p) (scatter); it is a per-channel scale:
scanner - ours = p g(t), with t the distance of the ray from the isocentre (a fan channel at angle gamma has
t = rho sin(gamma)). The fit uses parallel projections of the two images (smoothed by 2 mm) through axial slices,
leaving out rays that cross the body near the edge of the scanner's image, where the body may continue beyond it.

    python channel_correction.py <our volume .npy on the scanner grid> <scanner image folder> --exclude -215,-145 --out fit.json

The volume is (slices, rows, cols) attenuation per mm on the grid of the scanner's images (as run_vendor_match.py
saves it). --exclude leaves a z range out of the fit, to test the correction there.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def parallel_projections(img: np.ndarray, pixel: float, angles: np.ndarray) -> np.ndarray:
    """(slices, angles, t) line integrals of (slices, n, n) images by rotation and summation; t spacing = pixel."""
    n = img.shape[-1]
    x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[:, None]
    out = np.empty((img.shape[0], len(angles), n), dtype=np.float32)
    lin = torch.linspace(-1, 1, n)
    S, T = torch.meshgrid(lin, lin, indexing='ij')
    for i, a in enumerate(angles):
        ca, sa = float(np.cos(a)), float(np.sin(a))
        grid = torch.stack([T * ca - S * sa, T * sa + S * ca], -1)[None].expand(img.shape[0], n, n, 2)
        out[:, i] = (F.grid_sample(x, grid, align_corners=True)[:, 0].sum(1) * pixel).numpy()
    return out


def fit_channel_scale(ours: np.ndarray, scanner: np.ndarray, valid: np.ndarray, pixel: float, mu_water: float,
                      n_angles: int = 180, t_hold: float = 140.0) -> dict:
    """Fit scanner - ours = p (g0 + g2 (|t| / 100 mm)^2) on axial slices. ours, scanner: (slices, n, n) attenuation per mm
    on the same grid (negative values are clipped to 0); valid: the scanner's field of view. Also returns g per 10 mm
    band of |t| (least-squares slope)."""
    n = ours.shape[-1]
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n]
    r_mm = np.hypot(xx - c, yy - c) * pixel
    inside = valid & (r_mm < 165)[None]
    blur = lambda a: ndimage.gaussian_filter(np.where(inside, np.clip(a, 0, None), 0), (0, 2 / pixel, 2 / pixel)).astype(np.float32)
    body = ndimage.gaussian_filter(np.where(inside, scanner, 0), (0, 4, 4)) > 0.3 * mu_water
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    dp = parallel_projections(blur(scanner) - blur(ours), pixel, angles)
    p = parallel_projections(blur(ours), pixel, angles)
    ring = parallel_projections((body & (r_mm > 150)[None]).astype(np.float32), pixel, angles)
    t = np.broadcast_to(((np.arange(n) - c) * pixel)[None, None], p.shape)
    keep = (ring < 1e-3) & (p > 0.3)
    dp, p, at = (a[keep].astype(np.float64) for a in (dp, p, np.abs(t)))
    band = []
    for lo in range(0, 170, 10):
        m = (at >= lo) & (at < lo + 10)
        if m.sum() > 500:
            band.append((lo + 5.0, float((dp[m] * p[m]).sum() / (p[m] ** 2).sum()), int(m.sum())))
    A = np.stack([p, p * (at / 100) ** 2], 1)
    (g0, g2), *_ = np.linalg.lstsq(A, dp, rcond=None)
    resid = dp - A @ np.array([g0, g2])
    return dict(g0=float(g0), g2=float(g2), t_hold=t_hold, r2=float(1 - resid.var() / dp.var()), rays=int(keep.sum()), band=band)


def channel_scale(meta, fit: dict) -> np.ndarray:
    """(columns,) factors 1 + g(t) for the detector columns of a CTGen3ProjMeta, g held constant beyond t_hold."""
    rho = float(meta.source_rhos.double().mean())
    t = np.abs(rho * np.sin(meta.phis_det[:, 0].double().numpy()))
    return 1 + fit['g0'] + fit['g2'] * (np.minimum(t, fit.get('t_hold', 140.0)) / 100) ** 2


def apply_channel_scale(proj: np.ndarray, meta, fit: dict) -> np.ndarray:
    """proj (views, columns, rows) line integrals scaled per detector column, as a new float32 array."""
    return np.asarray(proj, dtype=np.float32) * channel_scale(meta, fit).astype(np.float32)[None, :, None]


if __name__ == '__main__':
    import pydicom
    parser = argparse.ArgumentParser()
    parser.add_argument('volume', help='(slices, rows, cols) attenuation per mm holding every scanner slice')
    parser.add_argument('images', help="folder of the scanner's images")
    parser.add_argument('--mu-water', type=float, required=True, help='(7041,1001) of the projections, per mm')
    parser.add_argument('--exclude', default=None, help='z range (mm) left out of the fit, e.g. -215,-145')
    parser.add_argument('--z-range', default='-305,-80', help='z range (mm) of the slices used')
    parser.add_argument('--every', type=float, default=7.5, help='spacing (mm) of the slices used')
    parser.add_argument('--out', default='channel_fit.json')
    args = parser.parse_args()
    sl = sorted((pydicom.dcmread(f) for f in glob.glob(os.path.join(args.images, '*.dcm'))), key=lambda s: float(s.ImagePositionPatient[2]))
    z = np.array([float(s.ImagePositionPatient[2]) for s in sl])
    vol = np.load(args.volume, mmap_mode='r')
    assert vol.shape[0] == len(z), 'the volume must hold every scanner slice'
    lo, hi = (float(v) for v in args.z_range.split(','))
    ex = [float(v) for v in args.exclude.split(',')] if args.exclude else None
    ks = sorted({int(np.argmin(np.abs(z - zz))) for zz in np.arange(lo, hi, args.every) if not (ex and ex[0] <= zz <= ex[1])})
    hu = np.stack([sl[k].pixel_array * float(sl[k].RescaleSlope) + float(sl[k].RescaleIntercept) for k in ks]).astype(np.float32)
    valid = hu > -1500
    scanner = np.where(valid, np.clip(args.mu_water * (1 + hu / 1000), 0, None), 0).astype(np.float32)
    fit = fit_channel_scale(np.asarray(vol[ks], dtype=np.float32), scanner, valid, float(sl[0].PixelSpacing[0]), args.mu_water)
    fit.update(slices=[float(z[k]) for k in ks], excluded=ex)
    json.dump(fit, open(args.out, 'w'), indent=1)
    print(f"{len(ks)} slices, {fit['rays']:,} rays: g = {100 * fit['g0']:+.2f}% {100 * fit['g2']:+.2f}% (|t| / 100 mm)^2, "
          f"held beyond {fit['t_hold']:.0f} mm; R^2 {fit['r2']:.3f}")
