"""Prototype helical FBP for third generation CT (cylindrical detector, helical focal spot path), in plain PyTorch:
rebinning to parallel beams and weighted filtered back projection (WFBP, in the manner of Stierstorfer et al. 2004).

Not part of the package: this is the starting point of the implementation planned in the pull request that adds it.

    f(x) = dtheta * sum_j w_j(x) Q_j(t_j(x), r_j(x))

  Q_j       parallel projection j, rebinned from the fan projections, weighted by the cone cosine and ramp filtered
            along t (Ram-Lak kernel, apodized)
  t, r      detector position (t) and row (r) where the ray of parallel angle theta_j through voxel x meets the detector;
            its focal spot is the one of fan view beta = theta_j - asin(t / rho), on the helix
  w_j       WFBP weight: the row weight W(q) of the ray divided by the sum of W(q) over the rays through x at the angles
            theta_j + k pi that were measured (q = normalized row coordinate)

GPU memory is bounded by an explicit budget (``gpu_budget``, bytes; by default 1.5 GB and never more than a quarter of
what is free when the function starts): projections stay on the CPU and are streamed in batches, the back projection's
4D temporaries are cut into z chunks sized from the budget, and the output volume is the only full-size array on the GPU.
Every step reports its measured peak. Flying focal spots are not supported yet.

An earlier fan-beam version of this file (in the git history of this branch) applied the redundancy weights after
fan-beam filtering, which is not valid in fan geometry and left low-frequency shading of tens of HU on helical data.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

#: bytes per element of the back projection's 4D (views x Nx x Ny x z) temporaries, at their peak; checked against
#: torch.cuda.max_memory_allocated on C145 (see README)
BYTES_PER_4D_ELEMENT = 40


def gpu_budget(budget: float | None = None, device: str = 'cuda') -> float:
    """Bytes this process may use on the GPU: the requested budget (default 1.5 GB), never more than a quarter of what is
    free now, since the GPU is shared."""
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    return min(1.5e9 if budget is None else budget, 0.25 * free)


def helix(meta) -> dict:
    """The focal spot path of a CTGen3ProjMeta as an angle grid and a line z = z0 + slope (beta - beta0)."""
    beta = np.unwrap(meta.source_phis.double().numpy())
    dbeta = (beta[-1] - beta[0]) / (len(beta) - 1)
    z = meta.source_focal_centers[:, 2].double().numpy()
    slope, z0 = np.polyfit(beta - beta[0], z, 1)
    gam = meta.phis_det[:, 0].double().numpy()
    if float(meta.source_phi_offsets.abs().max()) > 0 or float(meta.source_rho_offsets.abs().max()) > 0 or float(meta.source_z_offsets.abs().max()) > 0:
        raise NotImplementedError('flying focal spots are not handled yet: the rays of alternate views come from different focal spots')
    return dict(beta0=float(beta[0]), beta1=float(beta[-1]), dbeta=float(dbeta), slope=float(slope), z0=float(z0),
                rho=float(meta.source_rhos.double().mean()), DSD=float(meta.DSD), gamma0=float(gam[0]),
                dgamma=float(gam[1] - gam[0]), gamma_max=float(min(abs(gam[0]), abs(gam[-1]))),
                v0=float(meta.zs_det[0, 0]), dv=float(meta.zs_det[0, 1] - meta.zs_det[0, 0]),
                angle_error=float(np.abs(beta - (beta[0] + dbeta * np.arange(len(beta)))).max()))


def nominal_table_feed(meta, pitch: float) -> float:
    """Table feed per rotation (mm) from the pitch, (0018,9311) of the projections, times the collimation at the
    isocentre (rows x row spacing, scaled from the detector to the isocentre)."""
    return pitch * meta.shape[1] * float(meta.row_det_spacing) * float(meta.source_rhos.double().mean()) / float(meta.DSD)


def rescale_table_feed(meta, feed: float, anchor_view: int = -1) -> float:
    """Scale the focal spot z positions of a CTGen3ProjMeta, in place, about one view (the last by default) so that
    the table advances `feed` mm per rotation. For C145 the scanner's images use its nominal feed (39.375 mm) while the
    projections' focal spots advance 0.2% more; rescaled about the last view, our image lands on the scanner's z axis.
    Returns the scale applied."""
    k = feed / (abs(helix(meta)['slope']) * 2 * np.pi)
    seen = set()
    for name in ('source_focal_centers', 'source_focal_spots'):
        a = getattr(meta, name, None)
        if a is None or id(a) in seen:
            continue
        seen.add(id(a))
        z = a[:, 2].double()
        a[:, 2] = (z[anchor_view] + (z - z[anchor_view]) * k).to(a.dtype)
    return k


def rebin_to_parallel(proj: torch.Tensor, meta, budget: float | None = None, device: str = 'cuda',
                      angle_offset: float = 0.0) -> tuple:
    """Rebin fan projections (views, channels, rows; on the CPU) to parallel projections (angles, t, rows; on the CPU)
    on a uniform t grid, by bilinear interpolation in (view, channel). Only the fan views a chunk of parallel angles
    needs are copied to the GPU. angle_offset (rad) is added to every focal spot angle."""
    h = helix(meta)
    h['beta0'] += angle_offset; h['beta1'] += angle_offset
    N, ncol, nrow = proj.shape
    rho, dg = h['rho'], h['dgamma']
    dt = rho * abs(dg)
    t_max = rho * np.sin(h['gamma_max'])
    M = 2 * int(np.ceil(t_max / dt))
    t = (np.arange(M) - (M - 1) / 2) * dt
    gam_t = np.arcsin(np.clip(t / rho, -1, 1))
    n_pi = int(round(np.pi / abs(h['dbeta'])))                          # the gantry may turn either way
    dtheta = np.pi / n_pi
    beta_min, beta_max = min(h['beta0'], h['beta1']), max(h['beta0'], h['beta1'])
    theta0 = beta_min + h['gamma_max']
    J = int(np.floor((beta_max - h['gamma_max'] - theta0) / dtheta)) + 1
    theta = theta0 + np.arange(J) * dtheta
    # chunk size from the budget: the fan views a chunk spans plus the gathered and interpolated outputs
    fan_span = int(np.ceil(2 * h['gamma_max'] / abs(h['dbeta']))) + 3
    per_view = ncol * nrow * 4 + 6 * M * nrow * 4
    chunk = int(max(1, min(256, gpu_budget(budget, device) // per_view - fan_span)))
    col = torch.tensor((gam_t - h['gamma0']) / dg, device=device)
    c0 = col.floor().clamp(0, ncol - 2).long()
    fc = (col - c0).float()[None, :, None]
    inside = torch.tensor(np.abs(t) <= t_max, device=device).float()[None, :, None]
    gam_t_d = torch.tensor(gam_t, device=device)
    P = torch.empty((J, M, nrow), dtype=torch.float32)
    for s in range(0, J, chunk):
        th = torch.tensor(theta[s:s + chunk], device=device)
        iv = (th[:, None] - gam_t_d[None, :] - h['beta0']) / h['dbeta']       # fractional fan view, (j, M)
        lo = max(0, int(iv.min().floor()) - 1)
        hi = min(N, int(iv.max().ceil()) + 2)
        p = proj[lo:hi].to(device, torch.float32)                          # only the fan views this chunk needs
        i0 = (iv.floor().clamp(0, N - 2).long() - lo).clamp(0, hi - lo - 2)
        fi = (iv - (i0 + lo)).float().clamp(0, 1)[..., None]
        cc = c0[None, :].expand_as(i0)
        v = ((1 - fi) * ((1 - fc) * p[i0, cc] + fc * p[i0, cc + 1]) + fi * ((1 - fc) * p[i0 + 1, cc] + fc * p[i0 + 1, cc + 1]))
        P[s:s + chunk] = (v * inside).cpu()
        del p, v
    return P, theta, t, dict(h, dt=dt, dtheta=dtheta, n_pi=n_pi, rebin_chunk=chunk)


def ramp_filter_parallel(P: torch.Tensor, geo: dict, apodization='hann', budget: float | None = None,
                         device: str = 'cuda') -> torch.Tensor:
    """Ramp filter parallel projections (on the CPU) along t (Ram-Lak kernel, spacing dt) after the cone cosine weight.
    apodization: 'ram-lak', 'shepp-logan', 'hann', or a function of spatial frequency (cycles/mm) giving the window."""
    J, M, nrow = P.shape
    dt = geo['dt']
    n_pad = int(2 ** np.ceil(np.log2(2 * M)))
    n = torch.arange(-(n_pad // 2), n_pad // 2, dtype=torch.float64)
    h = torch.zeros_like(n)
    h[n == 0] = 1 / (4 * dt ** 2)
    odd = (n.abs() % 2) == 1
    h[odd] = -1 / (np.pi ** 2 * (n[odd] * dt) ** 2)
    G = torch.fft.fft(torch.fft.ifftshift(h)).real
    freq = torch.fft.fftfreq(n_pad)                                                   # cycles per sample
    if callable(apodization):
        G = G * torch.as_tensor(apodization(np.abs(freq.numpy()) / dt), dtype=torch.float64)
    elif apodization == 'hann':
        G = G * (0.5 + 0.5 * torch.cos(2 * np.pi * freq))
    elif apodization == 'shepp-logan':
        G = G * torch.sinc(freq)
    elif apodization != 'ram-lak':
        raise ValueError(apodization)
    G = G.to(device, torch.float32)
    v = geo['v0'] + geo['dv'] * np.arange(nrow)
    cone = torch.tensor(geo['DSD'] / np.sqrt(geo['DSD'] ** 2 + v ** 2), device=device, dtype=torch.float32)
    per_view = M * nrow * 4 * 2 + n_pad * nrow * 8 * 3                    # input, output, complex spectra
    chunk = int(max(1, min(1024, gpu_budget(budget, device) // per_view)))
    Q = torch.empty_like(P)
    for s in range(0, J, chunk):
        p = P[s:s + chunk].to(device) * cone
        S = torch.fft.fft(F.pad(p, (0, 0, 0, n_pad - M)), dim=1)
        del p
        S *= G[None, :, None]
        Q[s:s + chunk] = (dt * torch.fft.ifft(S, dim=1).real[:, :M]).cpu()
        del S
    return Q


def _wfbp_window(q: torch.Tensor, Q: float) -> torch.Tensor:
    """WFBP weight of the normalized row coordinate q: 1 for |q| <= Q, cos^2 down to 0 at |q| = 1, 0 beyond."""
    a = q.abs()
    w = torch.cos(0.5 * np.pi * torch.clamp((a - Q) / (1 - Q), 0, 1)) ** 2
    return w * (a <= 1)


def backproject_wfbp(Q: torch.Tensor, theta: np.ndarray, t: np.ndarray, geo: dict, X: torch.Tensor, Y: torch.Tensor,
                     Z: np.ndarray, Q_weight: float = 0.6, k_range: int = 2, z_offsets=(0.0,), budget: float | None = None,
                     device: str = 'cuda', views_per_batch: int = 4) -> torch.Tensor:
    """Voxel-driven weighted back projection of filtered parallel projections (on the CPU) onto the points
    (X[i, j], Y[i, j], Z[k]) (object frame, mm). Each ray's weight is its WFBP row weight over the sum for the rays
    through the same voxel at theta + k pi (k = -k_range..k_range) that were measured. z_offsets (mm): sub-slices whose
    contributions are averaged into each output slice, to widen its slice sensitivity profile.

    GPU memory: the output (Nx x Ny x Nz floats) plus, at a time, views_per_batch views of Q and 4D temporaries of
    views_per_batch x Nx x Ny x (z chunk) elements, the z chunk being sized so that they stay within the budget."""
    J, M, nrow = Q.shape
    DSD, rho, slope, z0, beta0 = geo['DSD'], geo['rho'], geo['slope'], geo['z0'], geo['beta0']
    dt, v0, dv, dtheta = geo['dt'], geo['v0'], geo['dv'], geo['dtheta']
    t_max = rho * np.sin(geo['gamma_max'])
    theta_lo, theta_hi = float(theta[0]) - 1e-6, float(theta[-1]) + 1e-6
    X, Y = X.to(device, torch.float32), Y.to(device, torch.float32)
    Nx, Ny = X.shape
    Z = np.asarray(Z, dtype=np.float64)
    order = np.argsort(Z)
    Zs = Z[order]
    Zd = torch.tensor(Zs, device=device, dtype=torch.float32)
    num = torch.zeros((Nx, Ny, len(Zs)), device=device)
    budget = gpu_budget(budget, device) - num.numel() * 4 - 6 * Nx * Ny * 4 * views_per_batch * (2 * k_range + 1)
    if budget <= 0:
        raise MemoryError('the output volume alone exceeds the GPU budget; reconstruct fewer slices at a time')
    nz_chunk = int(max(1, budget // (BYTES_PER_4D_ELEMENT * views_per_batch * Nx * Ny)))
    half_rows = (nrow - 1) / 2
    r_fov = float(torch.sqrt(X ** 2 + Y ** 2).max())
    reach = (half_rows + 1) * abs(dv) * (rho + r_fov) / DSD + 1.0 + max(abs(o) for o in z_offsets)
    for s in range(0, J, views_per_batch):
        e = min(J, s + views_per_batch)
        th_np = theta[s:e]
        zs_ends = z0 + slope * (np.concatenate([th_np - geo['gamma_max'], th_np + geo['gamma_max']]) - beta0)
        k0 = int(np.searchsorted(Zs, zs_ends.min() - reach))
        k1 = int(np.searchsorted(Zs, zs_ends.max() + reach, side='right'))
        if k1 <= k0:
            continue
        th = torch.tensor(th_np, device=device, dtype=torch.float32)[:, None, None]
        # in-plane quantities (views x Nx x Ny) of the ray through each voxel at theta + k pi
        rays = []
        for k in range(-k_range, k_range + 1):
            thk = th + k * np.pi
            tk = X[None] * torch.sin(thk) - Y[None] * torch.cos(thk)
            bk = thk - torch.asin(torch.clamp(tk / rho, -1, 1))
            L = -(X[None] - rho * torch.cos(bk)) * torch.cos(thk) - (Y[None] - rho * torch.sin(bk)) * torch.sin(thk)
            measured = ((thk >= theta_lo) & (thk <= theta_hi) & (tk.abs() <= t_max)).float()
            rays.append((k, tk if k == 0 else None, (z0 + slope * (bk - beta0)), DSD / L, measured))
            del bk, L
        t0_ = rays[k_range][1]
        tnorm = ((t0_ - float(t[0])) / dt) / (M - 1) * 2 - 1
        Qb = Q[s:e].to(device, non_blocking=True)[:, None]
        for c0 in range(k0, k1, nz_chunk):
            c1 = min(k1, c0 + nz_chunk)
            acc = torch.zeros((Nx, Ny, c1 - c0), device=device)
            for o in z_offsets:
                Zc = (Zd[c0:c1] + o)[None, None, None, :]
                total = None
                for k, _, zsrc, scale, measured in rays:
                    row = (Zc - zsrc[..., None]) * scale[..., None]
                    row = (row - v0) / dv
                    w = _wfbp_window((row - half_rows) / half_rows, Q_weight) * measured[..., None]
                    if k == 0:
                        row0, w0 = row, w
                    else:
                        del row
                    total = w.clone() if total is None else total.add_(w)
                    if k != 0:
                        del w
                w0 = w0.div_(total.clamp_(min=1e-6))
                del total
                grid = torch.stack([(row0 / (nrow - 1)) * 2 - 1, tnorm[..., None].expand_as(row0)], dim=-1)
                del row0
                samp = F.grid_sample(Qb, grid.reshape(e - s, Nx * Ny, c1 - c0, 2), mode='bilinear', align_corners=True)
                del grid
                acc += (w0 * samp.reshape(e - s, Nx, Ny, c1 - c0)).sum(0)
                del samp, w0
            num[:, :, c0:c1] += acc / len(z_offsets)
            del acc
        del rays, Qb, tnorm
    out = torch.empty_like(num)
    out[:, :, torch.as_tensor(order, device=device)] = dtheta * num                # back to the order Z was given in
    return out


def wfbp(proj: torch.Tensor, meta, X: torch.Tensor, Y: torch.Tensor, Z: np.ndarray, apodization='hann',
         Q_weight: float = 0.6, z_offsets=(0.0,), budget: float | None = None, device: str = 'cuda',
         angle_offset: float = 0.0) -> tuple:
    """Rebin to parallel beams, filter, and back project with WFBP weights onto the given points (object frame, mm).
    Returns the image on the GPU and timings, including the measured peak GPU memory of each stage."""
    stats = {}
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    P, theta, t, geo = rebin_to_parallel(proj, meta, budget, device, angle_offset)
    torch.cuda.synchronize(); stats['rebin_s'] = time.perf_counter() - t0
    stats['rebin_peak_GB'] = (torch.cuda.max_memory_allocated() - base) / 1e9
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    Qf = ramp_filter_parallel(P, geo, apodization, budget, device)
    del P
    torch.cuda.synchronize(); stats['filter_s'] = time.perf_counter() - t0
    stats['filter_peak_GB'] = (torch.cuda.max_memory_allocated() - base) / 1e9
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    recon = backproject_wfbp(Qf, theta, t, geo, X, Y, Z, Q_weight, z_offsets=z_offsets, budget=budget, device=device)
    torch.cuda.synchronize(); stats['backproject_s'] = time.perf_counter() - t0
    stats['backproject_peak_GB'] = (torch.cuda.max_memory_allocated() - base) / 1e9
    stats['budget_GB'] = gpu_budget(budget, device) / 1e9
    return recon, stats
