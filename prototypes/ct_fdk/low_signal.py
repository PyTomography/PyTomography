"""Adaptive filtering of photon-starved rays for DICOM-CT-PD data, in the spirit of Hsieh (1998) and Kachelriess et al.
(2001). Plain NumPy on the CPU: no GPU memory.

A ray expecting N = N0 exp(-p) photons has a line integral with variance about 1/N, and at a few photons the log is
biased as well: these rays make the streaks between the shoulders. DICOM-CT-PD stores N0, the incident photons per
detector column of every view, in (7033,1065) PhotonStatistics (bowtie and tube current modulation included).

Each ray with N below n_target is replaced by -log of the mean transmission exp(-p) over a neighbourhood
(views x columns x rows) of about n_target / N rays, i.e. about n_target photons. The neighbourhood grows through a fixed
set of box sizes, blending between neighbouring sizes in log size. Averaging transmission rather than line integrals
keeps the mean right. N is estimated from transmission smoothed over 3 x 3 x 3 rays, so that the choice of filter is not
driven by the noise itself. Rays with enough photons are left exactly as they are.
"""
import glob
import os

import numpy as np
import pydicom
from scipy import ndimage

#: neighbourhoods (views, columns, rows), in order of size
SIZES = ((1, 1, 1), (1, 3, 1), (1, 3, 3), (3, 3, 3), (3, 5, 3), (5, 5, 3), (5, 7, 5))


def acquisition_order(folder: str) -> list:
    """The DICOM-CT-PD files of a folder in acquisition order (InstanceNumber)."""
    paths = glob.glob(os.path.join(folder, '*.dcm'))
    order = [int(pydicom.dcmread(p, stop_before_pixels=True, specific_tags=['InstanceNumber']).InstanceNumber) for p in paths]
    return [p for _, p in sorted(zip(order, paths))]


def photon_statistics(paths) -> np.ndarray:
    """(views, columns) incident photons per detector column, (7033,1065) PhotonStatistics, for files in the given
    (acquisition) order."""
    return np.stack([np.frombuffer(pydicom.dcmread(p, stop_before_pixels=True, specific_tags=[(0x7033, 0x1065)])[0x7033, 0x1065].value,
                                   dtype='<f4') for p in paths])


def filter_low_signal(proj: np.ndarray, n0: np.ndarray, n_target: float, views=None, chunk: int = 400) -> tuple:
    """proj: (views, columns, rows) line integrals; n0: (views, columns) incident photons per detector column.
    views: optional (start, stop) range to filter (the others are copied unchanged). Returns the filtered line
    integrals (a new float32 array) and the fraction of all rays that were changed."""
    out = np.array(proj, dtype=np.float32, copy=True)
    V = proj.shape[0]
    v0, v1 = views if views is not None else (0, V)
    k = np.array([a * b * c for a, b, c in SIZES], dtype=np.float64)
    logk = np.log(k)
    margin = max(s[0] for s in SIZES) // 2 + 1
    changed = 0
    for s0 in range(v0, v1, chunk):
        s1 = min(v1, s0 + chunk)
        lo, hi = max(0, s0 - margin), min(V, s1 + margin)
        trans = np.exp(-np.asarray(proj[lo:hi], dtype=np.float32))
        n_est = n0[lo:hi, :, None] * ndimage.uniform_filter(trans, (3, 3, 3), mode='nearest')
        need = np.clip(n_target / np.maximum(n_est, 1e-9), 1.0, k[-1])
        if need.max() <= 1.0:
            continue
        x = np.interp(np.log(need), logk, np.arange(len(k)))                         # fractional size index
        acc = np.zeros_like(trans)
        for i, size in enumerate(SIZES):
            w = np.clip(1.0 - np.abs(x - i), 0.0, 1.0).astype(np.float32)
            if not w.any():
                continue
            acc += w * (trans if i == 0 else ndimage.uniform_filter(trans, size, mode='nearest'))
        filt = -np.log(np.maximum(acc, 1e-12))
        mask = need > 1.0
        sl = slice(s0 - lo, s0 - lo + (s1 - s0))
        out[s0:s1] = np.where(mask[sl], filt[sl], out[s0:s1])
        changed += int(mask[sl].sum())
    return out, changed / float(np.prod(proj.shape))
