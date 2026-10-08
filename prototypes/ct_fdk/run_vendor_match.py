"""Steps from the FBP prototype towards the scanner's own reconstruction of a DICOM-CT-PD scan (no flying focal spot),
each measured against it on the scanner's grid. GPU memory stays within --budget-gb (default 1.5).

    python run_vendor_match.py <projections> <scanner images> --cache scan.pt --out results.json

Steps (each adds one correction): WFBP on a 1 mm grid; on the scanner grid; the in-plane registration to the scanner
image (scale, rotation, shift) and a focal spot angle offset that removes the rotation; the scanner's kernel estimated
from the images; 1.25 mm slices. Reports tissue class means, the radial trend in soft tissue, the RMS difference from
the scanner image, noise, timings and the measured GPU peak.
"""
import argparse
import glob
import json
import os

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from scipy import ndimage, optimize

from pytomography.io.CT import dicom_ct_pd
import fdk_prototype
import channel_correction
import low_signal

parser = argparse.ArgumentParser()
parser.add_argument('projections')
parser.add_argument('images')
parser.add_argument('--cache', default=None)
parser.add_argument('--block', default='-200,-160', help='patient z range (mm) of the slices used')
parser.add_argument('--all-slices', action='store_true')
parser.add_argument('--budget-gb', type=float, default=1.5)
parser.add_argument('--start-at', type=int, default=1, help='first step to run; earlier results are read from --out')
parser.add_argument('--angle-offset-deg', type=float, default=None, help='with --start-at 4 or later: the offset step 3 found '
                    '(default: read from --out)')
parser.add_argument('--central-column-offset', type=float, default=0.0, help='channels added to the DetectorCentralElement column')
parser.add_argument('--low-signal', type=float, default=0.0, help='filter photon-starved rays to about this many photons (0: off)')
parser.add_argument('--channel-correction', default=None, help='JSON written by channel_correction.py (applied after the central column)')
parser.add_argument('--out', default='vendor_match.json')
parser.add_argument('--save-dir', default='.')
args = parser.parse_args()
dev, budget = 'cuda', args.budget_gb * 1e9
free, total = torch.cuda.mem_get_info()
print(f'GPU: {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB; this run keeps to {fdk_prototype.gpu_budget(budget) / 1e9:.2f} GB', flush=True)

# ---------------- data (projections stay on the CPU)
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
    print(f'central column {float(meta.detector_centers_col_idx[0]) + args.central_column_offset:.3f}', flush=True)
if args.low_signal:
    n0 = low_signal.photon_statistics(low_signal.acquisition_order(args.projections))
    filtered, frac = low_signal.filter_low_signal(proj.numpy(), n0, args.low_signal)
    proj = torch.from_numpy(filtered)
    print(f'photon-starved rays filtered to about {args.low_signal:g} photons: {100 * frac:.3f}% of rays changed', flush=True)
if args.channel_correction:
    proj = torch.from_numpy(channel_correction.apply_channel_scale(proj.numpy(), meta, json.load(open(args.channel_correction))))
    print(f'per-channel correction from {args.channel_correction}', flush=True)
first = pydicom.dcmread(glob.glob(os.path.join(args.projections, '*.dcm'))[0], stop_before_pixels=True)
mu_w = float(first[0x7041, 0x1001].value.decode().strip('\x00 '))
slices = sorted((pydicom.dcmread(f) for f in glob.glob(os.path.join(args.images, '*.dcm'))), key=lambda s: float(s.ImagePositionPatient[2]))
ipp = np.array([[float(v) for v in s.ImagePositionPatient] for s in slices])
ps = [float(v) for v in slices[0].PixelSpacing]
zs_all = ipp[:, 2]
xs = ipp[0, 0] + np.arange(int(slices[0].Columns)) * ps[1]
ys = ipp[0, 1] + np.arange(int(slices[0].Rows)) * ps[0]
lo, hi = (float(v) for v in args.block.split(','))
ks = np.arange(len(zs_all)) if args.all_slices else np.nonzero((zs_all >= lo) & (zs_all <= hi))[0]
zs = zs_all[ks]
hu_v = np.stack([slices[k].pixel_array * float(slices[k].RescaleSlope) + float(slices[k].RescaleIntercept) for k in ks]).astype(np.float32)
valid = hu_v > -1500
mu_v = np.where(valid, np.clip(mu_w * (1 + hu_v / 1000), 0, None), 0).astype(np.float32)
print(f'{len(ks)} scanner slices, z {zs[0]:.2f} to {zs[-1]:.2f} mm, pixel {ps[0]:.4f} mm, mu_water {mu_w}/mm, '
      f'thickness {slices[0].SliceThickness} mm, kernel {slices[0].get("ConvolutionKernel")}', flush=True)
meta.get_patient_affine(type('O', (), dict(shape=(1, 1, 1), dr=(1.0, 1.0, 1.0)))())          # raises unless FFS
X, Y = torch.meshgrid(torch.tensor(xs), torch.tensor(ys), indexing='ij')              # recon[i, j] <-> scanner[j, i]
Zobj = meta.z_center - zs                                                               # FFS: z_obj = z_center - z

# ---------------- metrics on the scanner grid
hu_smooth = ndimage.uniform_filter(np.where(valid, hu_v, -1000), size=(3, 7, 7))
classes = {'soft tissue': ((10, 70), 9), 'fat': ((-130, -70), 7), 'lung': ((-900, -700), 7), 'bone': ((500, 3000), 3)}
masks = {n: ndimage.binary_erosion((hu_smooth >= a) & (hu_smooth < b) & valid, structure=np.ones((3, e, e))) for n, ((a, b), e) in classes.items()}
Xg, Yg = np.meshgrid(xs, ys)
R = np.hypot(Xg, Yg)[None]
body = valid & (ndimage.gaussian_filter(mu_v, (0, 4, 4)) > 0.3 * mu_w)
soft = masks['soft tissue']
hp = lambda a: a - ndimage.gaussian_filter(a, (0, 6, 6))


def metrics(img_mu, label, t=None):
    """img_mu: (slices, rows, cols) attenuation per mm on the scanner grid."""
    hu = 1000 * (img_mu / mu_w - 1)
    d = hu - hu_v
    out = dict(step=label, **{f'{n} HU': float(hu[m].mean()) for n, m in masks.items()})
    out['centre soft - scanner'] = float(d[soft & (R < 40)].mean())
    out['edge soft - scanner'] = float(d[soft & (R >= 140) & (R < 180)].mean())
    out['RMS vs scanner (body)'] = float(np.sqrt((d[body] ** 2).mean()))
    out['noise (soft, high-pass SD)'] = float(hp(hu)[soft].std())
    if t:
        out.update(t)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
    return out


def register(img_mu):
    """In-plane similarity transform (scale, rotation, shift) that best maps our image onto the scanner's (both blurred
    1.5 px), as (scale, rotation in degrees, shift x mm, shift y mm). Small GPU use: two blocks of slices."""
    A_ref = torch.tensor(ndimage.gaussian_filter(mu_v, (0, 1.5, 1.5))).to(dev)
    A_our = torch.tensor(ndimage.gaussian_filter(img_mu, (0, 1.5, 1.5))).to(dev)
    n = mu_v.shape[1]; c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n]
    mask = torch.tensor(((xx - c) ** 2 + (yy - c) ** 2) <= (0.45 * n) ** 2).to(dev)

    def cost(p):
        s, a, dx, dy = p
        th = torch.tensor([[s * np.cos(a), -s * np.sin(a), dx / c], [s * np.sin(a), s * np.cos(a), dy / c]], dtype=torch.float32, device=dev)
        g = F.affine_grid(th[None].expand(A_our.shape[0], 2, 3), (A_our.shape[0], 1, n, n), align_corners=True)
        w = F.grid_sample(A_our[:, None], g, align_corners=True)[:, 0]
        return float(((w - A_ref)[:, mask] ** 2).mean())

    r = optimize.minimize(cost, [1.0, 0.0, 0.0, 0.0], method='Nelder-Mead',
                          options=dict(xatol=1e-5, fatol=1e-12, maxiter=600, initial_simplex=[[1, 0, 0, 0], [1.01, 0, 0, 0], [1, 0.01, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]]))
    s, a, dx, dy = r.x
    out = dict(scale=float(s), rotation_deg=float(np.degrees(a)), shift_x_mm=float(dx * ps[1]), shift_y_mm=float(dy * ps[0]))
    print('registration:', json.dumps({k: round(v, 5) for k, v in out.items()}), flush=True)
    return out


to_grid = lambda vol: vol.permute(2, 1, 0).cpu().numpy()                 # (Nx, Ny, Nz) -> (slices, rows, cols)
save = lambda name, img: np.save(os.path.join(args.save_dir, f'vendor_match_{name}.npy'), img)
step_of = lambda r: 4 if 'Ram-Lak' in r['step'] else (int(r['step'].split()[0]) if r['step'].split()[0].isdigit() else 0)
if args.start_at > 1:                                    # keep the earlier steps; steps 2 and 3 run together
    results = [r for r in json.load(open(args.out))['results'] if step_of(r) < (2 if args.start_at <= 3 else args.start_at)]
else:
    results = [dict(step='scanner', **{f'{n} HU': float(hu_v[m].mean()) for n, m in masks.items()},
                    **{'noise (soft, high-pass SD)': float(hp(hu_v)[soft].std())})]
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in results[0].items()}), flush=True)
angle_offset = np.radians(args.angle_offset_deg) if args.angle_offset_deg is not None else 0.0
step3 = [r for r in results if step_of(r) == 3]
if args.angle_offset_deg is None and args.start_at >= 4 and step3:
    angle_offset = np.radians(step3[-1]['angle_offset_deg'])
reg_final = {k[4:]: v for k, v in step3[-1].items() if k.startswith('reg ')} if step3 else None
dump = lambda: json.dump(dict(slices=[float(z) for z in zs], angle_offset_deg=float(np.degrees(angle_offset)), results=results), open(args.out, 'w'), indent=1)


def run(label, **kw):
    vol, t = fdk_prototype.wfbp(proj, meta, X, Y, Zobj, budget=budget, **kw)
    img = to_grid(vol)
    del vol
    torch.cuda.empty_cache()
    return img, {k: round(v, 3) for k, v in t.items()}


# 1. WFBP on PyTomography's 1 mm grid, resampled onto the scanner grid
if args.start_at <= 1:
    xg = torch.arange(512) - 255.5
    Xo, Yo = torch.meshgrid(xg, xg, indexing='ij')
    zo = np.arange(int(np.floor(Zobj.min())) - 2, int(np.ceil(Zobj.max())) + 3, 1.0)
    vol, t = fdk_prototype.wfbp(proj, meta, Xo, Yo, zo, apodization='hann', budget=budget)
    Pz, Py, Px = torch.meshgrid(torch.tensor(Zobj).to(dev), torch.tensor(ys).to(dev), torch.tensor(xs).to(dev), indexing='ij')
    grid = torch.stack([(Pz - zo[0]) / (zo[-1] - zo[0]) * 2 - 1, (Py + 255.5) / 511 * 2 - 1, (Px + 255.5) / 511 * 2 - 1], -1)
    img = F.grid_sample(vol[None, None], grid[None].float(), align_corners=True)[0, 0].cpu().numpy()
    del vol, grid, Pz, Py, Px
    torch.cuda.empty_cache()
    results.append(metrics(img, '1 parallel rebinning + WFBP weights (1 mm grid, resampled)', {k: round(v, 3) for k, v in t.items()}))
    dump()

# 2. on the scanner grid
if args.start_at <= 3:
    img, t = run('grid', apodization='hann')
    save('grid', img)
    reg = register(img)
    results.append(metrics(img, '2 + reconstructed on the scanner grid', dict(t, **{f'reg {k}': v for k, v in reg.items()})))

    # 3. focal spot angle offset: rotate the acquisition geometry to cancel the residual rotation (a global angle offset
    #    leaves the data consistent, so only the comparison with the scanner can find it)
    best = (abs(reg['rotation_deg']), 0.0, img, reg)
    for offset in (np.radians(reg['rotation_deg']), -np.radians(reg['rotation_deg'])):
        img_o, t_o = run('angle', apodization='hann', angle_offset=float(offset))
        reg_o = register(img_o)
        if abs(reg_o['rotation_deg']) < best[0]:
            best = (abs(reg_o['rotation_deg']), float(offset), img_o, reg_o)
    angle_offset = best[1]
    img = best[2]
    reg_final = best[3]
    save('angle', img)
    results.append(metrics(img, f'3 + focal spot angle offset {np.degrees(angle_offset):+.3f} deg', dict(t, **{f'reg {k}': v for k, v in best[3].items()}, angle_offset_deg=float(np.degrees(angle_offset)))))
    dump()

# 4. the scanner's kernel relative to Ram-Lak, from the cross spectrum of the two images (radially averaged)
ramlak, t = run('ramlak', apodization='ram-lak', angle_offset=angle_offset)
taper = np.clip((165 - R[0]) / 10, 0, 1) * valid.all(axis=0)
Fv = np.fft.fft2(mu_v * taper)
Fr = np.fft.fft2(ramlak * taper)
fy, fx = np.meshgrid(np.fft.fftfreq(mu_v.shape[1], ps[0]), np.fft.fftfreq(mu_v.shape[2], ps[1]), indexing='ij')
which = np.digitize(np.hypot(fx, fy), np.arange(0, 1.2, 0.01)) - 1
f_mid = np.arange(0, 1.2, 0.01)[:120] + 0.005
power_v = np.bincount(which.ravel(), (np.abs(Fv) ** 2).sum(0).ravel(), minlength=120)


def spectra(Fr):
    """Radially averaged cross spectrum (real part), our power, and the coherence of the two images."""
    cs = (Fv * np.conj(Fr)).sum(0).ravel()
    cross = np.bincount(which.ravel(), cs.real, minlength=120)
    cross_im = np.bincount(which.ravel(), cs.imag, minlength=120)
    power = np.bincount(which.ravel(), (np.abs(Fr) ** 2).sum(0).ravel(), minlength=120)
    return cross, power, np.hypot(cross, cross_im) / np.sqrt(np.maximum(power, 1e-30) * np.maximum(power_v, 1e-30))


# The registration leaves our image shifted against the scanner's (0.44 mm on C145). Averaged over directions, a shift d
# scales the cross spectrum by J0(2 pi f d), 20% at 0.35 cycles/mm, so remove it first: a phase ramp, which unlike
# resampling does not blur. The sign is checked on the data (expected +1: scanner(x) = ours(x + d)).
cross0, power, coherence0 = spectra(Fr)
cross, coherence = cross0, coherence0
if reg_final is not None:
    ramp = np.exp(2j * np.pi * (fx * reg_final['shift_x_mm'] + fy * reg_final['shift_y_mm']))
    band = (f_mid > 0.15) & (f_mid < 0.4)
    trials = {sign: spectra(Fr * (ramp if sign > 0 else np.conj(ramp))) for sign in (1, -1)}
    sign = max(trials, key=lambda k: trials[k][2][band].mean())
    cross, _, coherence = trials[sign]
    print(f"kernel estimate: removed the shift ({reg_final['shift_x_mm']:.3f}, {reg_final['shift_y_mm']:.3f}) mm with sign "
          f"{sign:+d}; coherence at 0.15-0.4 cycles/mm {coherence0[band].mean():.3f} -> {coherence[band].mean():.3f} "
          f"(other sign {trials[-sign][2][band].mean():.3f}); rotation {reg_final['rotation_deg']:.4f} deg and scale "
          f"{reg_final['scale']:.5f} left as they are", flush=True)
ok = power > 0
H, H0 = (c / np.maximum(power, 1e-30) for c in (cross, cross0))
H, H0 = H / np.mean(H[ok][:4]), H0 / np.mean(H0[ok][:4])
kernel = (f_mid[ok], np.clip(ndimage.uniform_filter1d(H, 3)[ok], 0, 2))
np.savetxt(os.path.join(args.save_dir, 'vendor_kernel.txt'), np.stack([f_mid[ok], H[ok], coherence[ok], H0[ok], coherence0[ok]], 1),
           header='spatial frequency (cycles/mm), scanner image / Ram-Lak image (cross spectrum / power), coherence, '
                  'and the same two without removing the residual shift')
print('scanner kernel / Ram-Lak at 0.1 ... 0.7 cycles/mm:', np.round(np.interp(np.arange(0.1, 0.71, 0.1), *kernel), 3).tolist(),
      '| coherence:', np.round(np.interp(np.arange(0.1, 0.71, 0.1), f_mid[ok], coherence[ok]), 3).tolist(), flush=True)
f_nyq = 0.5 / ps[0]
apod = lambda f: np.where(f <= f_nyq, np.interp(f, *kernel), np.interp(f_nyq, *kernel) * np.clip(1 - (f - f_nyq) / 0.1, 0, 1))
results.append(metrics(ramlak, '   (Ram-Lak, for the kernel estimate)', t))
img, t = run('kernel', apodization=apod, angle_offset=angle_offset)
save('kernel', img)
results.append(metrics(img, '4 + the scanner kernel, estimated from the images', t))
dump()

# 5. 1.25 mm slices: four sub-slices 0.25 mm apart averaged into each slice (same output size, four times the work)
img, t = run('thickness', apodization=apod, angle_offset=angle_offset, z_offsets=(-0.375, -0.125, 0.125, 0.375))
save('thickness', img)
results.append(metrics(img, '5 + 1.25 mm slices', t))
dump()
print('done', flush=True)
