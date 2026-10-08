"""Helical filtered back projection for third-generation CT (cylindrical detector, helical focal spot path), used by
:meth:`CTGen3SystemMatrix._fbp`. The fan projections are rebinned to parallel beams and reconstructed by weighted
filtered back projection (WFBP; Stierstorfer et al., Phys. Med. Biol. 49, 2209, 2004):

    f(x) = dtheta * sum_j w_j(x) Q_j(t_j(x), r_j(x))

* ``Q_j``: parallel projection j, rebinned from the fan projections, cone weighted and ramp filtered along t (with a
  window);
* ``t, r``: detector position (t) and row (r) of the ray of angle theta_j through voxel x; its focal spot is that of the
  fan view at beta = theta_j - asin(t / rho), on the helix;
* ``w_j``: the row weight W(q) of that ray divided by the sum of W(q) over the rays through x at theta_j + k pi that
  were measured.

With a flying focal spot the views fall into groups that share a focal spot offset (two for the alternating modes);
each group is a helix of its own, with its own focal spot radius, angles, axial offset and fan angles, and is
reconstructed separately; the groups are averaged.

GPU memory is bounded by a budget (:func:`pytomography.utils.gpu_budget`): the projections stay on the host and are
streamed in chunks, the back projection's 4D temporaries are cut into z chunks sized from the budget, and the output
is the only full-size array on the device.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
import pytomography
from pytomography.utils.memory import gpu_budget
from pytomography.utils.fourier_filters import ramp_filter_response

#: bytes per element of the back projection's 4D (views x Nx x Ny x z) temporaries, at their peak
_BYTES_PER_4D_ELEMENT = 40


def view_groups(meta) -> list:
    """Indices of the views sharing a focal spot offset: one group without a flying focal spot, two for the
    alternating modes. Raises for patterns other than a regular alternation of a few positions."""
    off = torch.stack([meta.source_phi_offsets, meta.source_rho_offsets, meta.source_z_offsets], 1).double().numpy()
    keys = np.round(off / np.array([1e-5, 1e-3, 1e-3]))
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    groups = [np.nonzero(inverse == g)[0] for g in range(inverse.max() + 1)]
    if len(groups) > 4 or any(len(np.unique(np.diff(g))) > 1 for g in groups if len(g) > 1):
        raise NotImplementedError('filtered back projection supports a flying focal spot that alternates regularly between '
                                  f'a few positions; this scan has {len(groups)} focal spot positions in an irregular pattern')
    return groups


def group_geometry(meta, idx: np.ndarray) -> dict:
    """Helix and fan of the views ``idx`` (one focal spot group): the focal spot angles (unwrapped) and their spacing,
    the focal spot radius, the line z = z0 + slope (beta - beta0) of the focal spot, the fan angle of every column as
    seen from the focal spot, and the in-plane distance from the focal spot to the detector."""
    spots = meta.source_focal_spots[idx].double().numpy()
    centres = meta.source_focal_centers[idx].double().numpy()
    beta = np.unwrap(np.arctan2(spots[:, 1], spots[:, 0]))
    dbeta = (beta[-1] - beta[0]) / max(len(beta) - 1, 1)
    rho = float(np.hypot(spots[:, 0], spots[:, 1]).mean())
    slope, z0 = np.polyfit(beta - beta[0], spots[:, 2], 1) if len(beta) > 1 else (0.0, float(spots[0, 2]))
    # fan angles from the focal spot: angle(D_k - S) - angle(O - S), which is phis_det without a flying focal spot
    first = torch.as_tensor(idx[:1])
    det = meta.get_detector_coordinates(first)[0, :, 0, :2].double().numpy()
    s = spots[0, :2]
    gamma = np.angle(np.exp(1j * (np.arctan2(det[:, 1] - s[1], det[:, 0] - s[0]) - np.arctan2(-s[1], -s[0]))))
    dsd = np.hypot(det[:, 0] - s[0], det[:, 1] - s[1])
    rows = meta.zs_det[0].double().numpy()
    return dict(idx=idx, beta=beta, beta0=float(beta[0]), beta1=float(beta[-1]), dbeta=float(dbeta), rho=rho,
                slope=float(slope), z0=float(z0), gamma=gamma, gamma_max=float(min(abs(gamma[0]), abs(gamma[-1]))),
                dsd=float(np.interp(0.0, gamma, dsd)), dz=float((spots[:, 2] - centres[:, 2]).mean()),
                v0=float(rows[0]), dv=float(rows[1] - rows[0]) if len(rows) > 1 else 1.0, n_rows=len(rows),
                angle_error=float(np.abs(beta - (beta[0] + dbeta * np.arange(len(beta)))).max()))


def rebin_to_parallel(proj: torch.Tensor, geo: dict, dt: float, budget: float | None, device) -> tuple:
    """Rebin the fan projections of one focal spot group (``proj``: all views, columns, rows, on the host) to parallel
    projections (angles, t, rows, on the host) on a uniform t grid of spacing ``dt``, by bilinear interpolation in (view,
    column). Only the fan views a chunk of parallel angles needs are copied to the device."""
    idx = geo['idx']
    N, ncol, nrow = len(idx), proj.shape[1], proj.shape[2]
    rho, gamma = geo['rho'], geo['gamma']
    t_max = rho * np.sin(geo['gamma_max'])
    M = 2 * int(np.ceil(t_max / dt))
    t = (np.arange(M) - (M - 1) / 2) * dt
    gam_t = np.arcsin(np.clip(t / rho, -1, 1))
    n_pi = int(round(np.pi / abs(geo['dbeta'])))                         # the gantry may turn either way
    dtheta = np.pi / n_pi
    beta_min, beta_max = min(geo['beta0'], geo['beta1']), max(geo['beta0'], geo['beta1'])
    theta0 = beta_min + geo['gamma_max']
    J = int(np.floor((beta_max - geo['gamma_max'] - theta0) / dtheta)) + 1
    if J < 1:
        raise ValueError('the scan covers too small an angle for filtered back projection')
    theta = theta0 + np.arange(J) * dtheta
    fan_span = int(np.ceil(2 * geo['gamma_max'] / abs(geo['dbeta']))) + 3
    per_view = ncol * nrow * 4 + 6 * M * nrow * 4
    chunk = int(max(1, min(256, gpu_budget(budget, device) // per_view - fan_span)))
    order = np.arange(ncol) if gamma[-1] > gamma[0] else np.arange(ncol)[::-1]       # columns may run either way
    col = torch.tensor(np.interp(gam_t, gamma[order], order.astype(np.float64)), device=device)
    c0 = col.floor().clamp(0, ncol - 2).long()
    fc = (col - c0).float()[None, :, None]
    inside = torch.tensor(np.abs(t) <= t_max, device=device).float()[None, :, None]
    gam_t_d = torch.tensor(gam_t, device=device)
    P = torch.empty((J, M, nrow), dtype=torch.float32)
    for s in range(0, J, chunk):
        th = torch.tensor(theta[s:s + chunk], device=device)
        iv = (th[:, None] - gam_t_d[None, :] - geo['beta0']) / geo['dbeta']     # fractional view of the group, (j, M)
        lo = max(0, int(iv.min().floor()) - 1)
        hi = min(N, int(iv.max().ceil()) + 2)
        p = proj[torch.as_tensor(idx[lo:hi])].to(device, torch.float32)
        i0 = (iv.floor().clamp(0, N - 2).long() - lo).clamp(0, hi - lo - 2)
        fi = (iv - (i0 + lo)).float().clamp(0, 1)[..., None]
        cc = c0[None, :].expand_as(i0)
        v = ((1 - fi) * ((1 - fc) * p[i0, cc] + fc * p[i0, cc + 1]) + fi * ((1 - fc) * p[i0 + 1, cc] + fc * p[i0 + 1, cc + 1]))
        P[s:s + chunk] = (v * inside).cpu()
        del p, v
    return P, theta, t, dtheta


def ramp_filter(P: torch.Tensor, geo: dict, dt: float, window, budget: float | None, device) -> torch.Tensor:
    """Cone weight and ramp filter the parallel projections (on the host) along t: the band-limited Ram-Lak kernel of
    spacing ``dt``, times ``window(f, f_nyquist)`` with f in cycles per mm."""
    J, M, nrow = P.shape
    n_pad = int(2 ** np.ceil(np.log2(2 * M)))
    G = ramp_filter_response(n_pad, dt, window).to(device, torch.float32)
    v = geo['v0'] + geo['dv'] * np.arange(nrow) - geo['dz']
    cone = torch.tensor(geo['dsd'] / np.sqrt(geo['dsd'] ** 2 + v ** 2), device=device, dtype=torch.float32)
    per_view = M * nrow * 4 * 2 + n_pad * nrow * 8 * 3
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


def _row_window(q: torch.Tensor, Q: float) -> torch.Tensor:
    """WFBP weight of the normalized row coordinate q: 1 for |q| <= Q, cos^2 down to 0 at |q| = 1, 0 beyond."""
    a = q.abs()
    w = torch.cos(0.5 * np.pi * torch.clamp((a - Q) / (1 - Q), 0, 1)) ** 2
    return w * (a <= 1)


def partner_range(geo: dict, r_fov: float, theta: np.ndarray, reach_extra: float = 0.0) -> int:
    """How many half turns either side can hold a measured ray through the same voxel: the axial reach of the cone at
    the far side of the field of view over the table feed per half turn, and no more half turns than were scanned."""
    reach = (geo['n_rows'] / 2 + 1) * abs(geo['dv']) * (geo['rho'] + r_fov) / geo['dsd'] + reach_extra
    half_turn_feed = abs(geo['slope']) * np.pi
    k_feed = int(np.ceil(reach / half_turn_feed)) if half_turn_feed > 1e-9 else 10 ** 6
    k_scan = int(np.ceil((theta[-1] - theta[0]) / np.pi))
    return max(1, min(k_feed, k_scan))


def backproject(Q: torch.Tensor, theta: np.ndarray, t: np.ndarray, dtheta: float, geo: dict, X: torch.Tensor, Y: torch.Tensor,
                Z: np.ndarray, out: torch.Tensor, Q_weight: float = 0.6, k_range: int | None = None, z_offsets=(0.0,),
                budget: float | None = None, views_per_batch: int = 4) -> None:
    """Voxel-driven weighted back projection of the filtered parallel projections of one focal spot group (on the host)
    onto the points (X[i, j], Y[i, j], Z[k]) (object frame, mm; Z increasing), added into ``out`` (Nx, Ny, Nz). Each
    ray's weight is its WFBP row weight over the sum for the rays through the same voxel at theta + k pi that were
    measured. ``z_offsets`` (mm): sub-slices whose contributions are averaged into each slice, which widens its slice
    sensitivity profile."""
    device = out.device
    J, M, nrow = Q.shape
    rho, slope, z0, beta0, dsd, dz = geo['rho'], geo['slope'], geo['z0'], geo['beta0'], geo['dsd'], geo['dz']
    dt, v0, dv = float(t[1] - t[0]), geo['v0'], geo['dv']
    t_max = rho * np.sin(geo['gamma_max'])
    theta_lo, theta_hi = float(theta[0]) - 1e-6, float(theta[-1]) + 1e-6
    X, Y = X.to(device, torch.float32), Y.to(device, torch.float32)
    Nx, Ny = X.shape
    Z = np.asarray(Z, dtype=np.float64)
    Zd = torch.tensor(Z, device=device, dtype=torch.float32)
    r_fov = float(torch.sqrt(X ** 2 + Y ** 2).max())
    if k_range is None:
        k_range = partner_range(geo, r_fov, theta, max(abs(o) for o in z_offsets))
    budget = gpu_budget(budget, device) - out.numel() * 4 - 6 * Nx * Ny * 4 * views_per_batch * (2 * k_range + 1)
    if budget <= 0:
        raise MemoryError('the GPU budget is too small for this object; raise gpu_budget or reconstruct fewer voxels')
    nz_chunk = int(max(1, budget // (_BYTES_PER_4D_ELEMENT * views_per_batch * Nx * Ny)))
    half_rows = (nrow - 1) / 2
    reach = (half_rows + 1) * abs(dv) * (rho + r_fov) / dsd + 1.0 + max(abs(o) for o in z_offsets) + abs(dz)
    for s in range(0, J, views_per_batch):
        e = min(J, s + views_per_batch)
        th_np = theta[s:e]
        zs_ends = z0 + slope * (np.concatenate([th_np - geo['gamma_max'], th_np + geo['gamma_max']]) - beta0)
        k0 = int(np.searchsorted(Z, zs_ends.min() - reach))
        k1 = int(np.searchsorted(Z, zs_ends.max() + reach, side='right'))
        if k1 <= k0:
            continue
        th = torch.tensor(th_np, device=device, dtype=torch.float32)[:, None, None]
        rays = []                                     # in-plane quantities (views, Nx, Ny) of the ray at theta + k pi
        for k in range(-k_range, k_range + 1):
            thk = th + k * np.pi
            tk = X[None] * torch.sin(thk) - Y[None] * torch.cos(thk)
            bk = thk - torch.asin(torch.clamp(tk / rho, -1, 1))
            L = -(X[None] - rho * torch.cos(bk)) * torch.cos(thk) - (Y[None] - rho * torch.sin(bk)) * torch.sin(thk)
            measured = ((thk >= theta_lo) & (thk <= theta_hi) & (tk.abs() <= t_max)).float()
            rays.append((k, tk if k == 0 else None, z0 + slope * (bk - beta0), dsd / L, measured))
            del bk, L
        tnorm = ((rays[k_range][1] - float(t[0])) / dt) / (M - 1) * 2 - 1
        Qb = Q[s:e].to(device, non_blocking=True)[:, None]
        for c0 in range(k0, k1, nz_chunk):
            c1 = min(k1, c0 + nz_chunk)
            acc = torch.zeros((Nx, Ny, c1 - c0), device=device)
            for o in z_offsets:
                Zc = (Zd[c0:c1] + o)[None, None, None, :]
                total = None
                for k, _, zsrc, scale, measured in rays:
                    row = ((Zc - zsrc[..., None]) * scale[..., None] + dz - v0) / dv
                    w = _row_window((row - half_rows) / half_rows, Q_weight) * measured[..., None]
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
            out[:, :, c0:c1] += dtheta * acc / len(z_offsets)
            del acc
        del rays, Qb, tnorm


def fbp_helical(proj: torch.Tensor, meta, X: torch.Tensor, Y: torch.Tensor, Z: np.ndarray, window, z_offsets=(0.0,),
                Q_weight: float = 0.6, k_range: int | None = None, budget: float | None = None, device=None,
                stats: dict | None = None, backend: str = 'auto') -> torch.Tensor:
    """WFBP of DICOM-CT-PD style projections (views, columns, rows; line integrals) with metadata ``meta``
    (:class:`CTGen3ProjMeta`) onto the points (X[i, j], Y[i, j], Z[k]) (object frame, mm; Z increasing). Returns the
    image (Nx, Ny, Nz) on ``device``. ``stats``, if given, receives the time and measured peak memory of each stage.
    ``backend``: ``'cuda'`` for the fused CUDA kernel (:mod:`._wfbp_cuda`, needs CuPy), ``'torch'`` for PyTorch, or
    ``'auto'``: the fused kernel when it can run (CuPy, a CUDA device, uniformly spaced Z), else PyTorch."""
    import time
    from . import _wfbp_cuda
    device = torch.device(pytomography.device if device is None else device)
    uniform = len(Z) < 2 or np.allclose(np.diff(np.asarray(Z, dtype=np.float64)), float(Z[1] - Z[0]), rtol=1e-6, atol=1e-6)
    if backend not in ('auto', 'cuda', 'torch'):
        raise ValueError(f'unknown backend {backend!r}')
    use_cuda = backend == 'cuda' or (backend == 'auto' and uniform and _wfbp_cuda.available(device)
                                      and (k_range is None or k_range <= _wfbp_cuda.MAX_K_RANGE))
    proj = proj.detach().to('cpu', torch.float32)
    Z = np.asarray(Z, dtype=np.float64)
    if np.any(np.diff(Z) <= 0):
        raise ValueError('Z must increase')
    dt = float(meta.source_rhos.double().mean()) * abs(float(meta.col_det_spacing))
    groups = view_groups(meta)
    out = torch.zeros((X.shape[0], X.shape[1], len(Z)), dtype=torch.float32, device=device)
    cuda = device.type == 'cuda'
    for g, idx in enumerate(groups):
        geo = group_geometry(meta, idx)
        if cuda:
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device); base = torch.cuda.memory_allocated(device)
        t0 = time.perf_counter()
        P, theta, t, dtheta = rebin_to_parallel(proj, geo, dt, budget, device)
        Qf = ramp_filter(P, geo, dt, window, budget, device)
        del P
        t1 = time.perf_counter()
        k = k_range
        if use_cuda and k is None:
            k = partner_range(geo, float(torch.sqrt(X.double() ** 2 + Y.double() ** 2).max()), theta, max(abs(o) for o in z_offsets))
        ran_cuda = use_cuda and k <= _wfbp_cuda.MAX_K_RANGE
        if ran_cuda:
            _wfbp_cuda.backproject(Qf, theta, t, dtheta, geo, X, Y, Z, out, Q_weight=Q_weight, k_range=k, z_offsets=z_offsets, budget=budget)
        else:
            backproject(Qf, theta, t, dtheta, geo, X, Y, Z, out, Q_weight=Q_weight, k_range=k_range, z_offsets=z_offsets, budget=budget)
        del Qf
        if cuda:
            torch.cuda.synchronize(device)
        if stats is not None:
            stats['backend'] = 'cuda' if ran_cuda else 'torch'
            stats.setdefault('groups', []).append(dict(views=len(idx), rebin_filter_s=t1 - t0, backproject_s=time.perf_counter() - t1,
                                                       peak_GB=(torch.cuda.max_memory_allocated(device) - base) / 1e9 if cuda else None))
    out /= len(groups)
    return out
