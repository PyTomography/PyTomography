"""Prototype helical FDK for third generation CT (cylindrical detector, helical focal spot path), in plain PyTorch.

Not part of the package: this is the starting point of the implementation planned in the pull request that adds it.

    f(x) = 2 dbeta * sum_b w_b(x) Q_b(gamma_b(x), v_b(x)) / L_b(x)^2

  Q_b       projections of view b, pre-weighted by D cos(gamma) cos(kappa) and ramp filtered along the detector rows with
            the kernel for equiangular rays, g(n a) = 0.5 (n a / sin(n a))^2 h(n a) (Kak & Slaney, ch. 3)
  gamma, v  fan angle and detector row height of the ray from the focal spot of view b through voxel x
  L_b       in-plane distance from that focal spot to the voxel
  w_b       helical redundancy weight. In-plane the focal spot travels a circle, so the views that measure the same
            in-plane line through x are known exactly: beta + 2 pi m (same direction) and beta + pi + 2 gamma + 2 pi m
            (opposite direction, fan angle -gamma, focal spot to voxel distance 2 rho cos(gamma) - L), at focal spot
            heights given by the helix. w_b is the row window of view b divided by the sum of the row windows of all
            those equivalent rays (cone angle based weighting in the spirit of Tang and Hsieh).

The weights depend on the voxel, so they are applied in the back projection, after filtering; that is an approximation,
and the source of the low frequency shading this prototype still shows on helical data (see README.md). Flying focal
spots are not supported.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F


def ramp_filter(proj: torch.Tensor, meta, apodization: str = 'hann', device: str = 'cuda', chunk: int = 512) -> torch.Tensor:
    """Filtered projections Q (on the CPU) from projections shaped (views, columns, rows), for a CTGen3ProjMeta."""
    N, ncol, nrow = proj.shape
    DSD = float(meta.DSD)
    D = float(meta.source_rhos.double().mean())
    gam = meta.phis_det[:, 0].double()
    vdet = meta.zs_det[0, :].double()
    dgam = float(gam[1] - gam[0])
    n_pad = int(2 ** np.ceil(np.log2(2 * ncol)))
    n = torch.arange(-(n_pad // 2), n_pad // 2, dtype=torch.float64)
    h = torch.zeros_like(n)
    h[n == 0] = 1 / (4 * dgam ** 2)
    odd = (n.abs() % 2) == 1
    h[odd] = -1 / (np.pi ** 2 * (n[odd] * dgam) ** 2)
    ratio = torch.ones_like(n)
    nonzero = n != 0
    ratio[nonzero] = (n[nonzero] * dgam / torch.sin(n[nonzero] * dgam)) ** 2
    G = torch.fft.fft(torch.fft.ifftshift(0.5 * ratio * h)).real
    freq = torch.fft.fftfreq(n_pad)
    if apodization == 'hann':
        G = G * (0.5 + 0.5 * torch.cos(2 * np.pi * freq))
    elif apodization == 'shepp-logan':
        G = G * torch.sinc(freq)
    elif apodization != 'ram-lak':
        raise ValueError(apodization)
    G = G.to(device, torch.float32)
    pre = (D * torch.cos(gam)[:, None] * DSD / torch.sqrt(DSD ** 2 + vdet[None, :] ** 2)).to(device, torch.float32)
    Q = torch.empty((N, ncol, nrow), dtype=torch.float32)
    for s in range(0, N, chunk):
        p = proj[s:s + chunk].to(device, torch.float32) * pre
        P = torch.fft.fft(F.pad(p, (0, 0, 0, n_pad - ncol)), dim=1)
        Q[s:s + chunk] = (dgam * torch.fft.ifft(P * G[None, :, None], dim=1).real[:, :ncol]).cpu()
    return Q


def _window(row: torch.Tensor, nrow: int, taper_rows: int) -> torch.Tensor:
    """Row weight: 0 outside the detector, rising as a cosine over the outer taper_rows rows to 1 inside."""
    edge = torch.minimum(row, (nrow - 1) - row)
    w = 0.5 - 0.5 * torch.cos(np.pi * torch.clamp(edge / taper_rows, 0, 1)) if taper_rows > 0 else torch.ones_like(edge)
    return w * (edge >= 0)


def backproject(Q: torch.Tensor, meta, shape: tuple, dr: tuple, taper_rows: int = 4, fov_radius: float = 250.0,
                device: str = 'cuda', batch: int = 4, m_range: int = 2) -> torch.Tensor:
    """Voxel-driven helical back projection of filtered projections Q onto an object of the given shape and voxel size,
    centred as CTGen3SystemMatrix centres it. Each view only touches the slab of slices its cone can illuminate."""
    N, ncol, nrow = Q.shape
    Nx, Ny, Nz = shape
    dxy, dz = float(dr[0]), float(dr[2])
    DSD = float(meta.DSD)
    rho = float(meta.source_rhos.double().mean())
    beta = torch.from_numpy(np.unwrap(meta.source_phis.double().numpy()))
    Sc = meta.source_focal_centers.double()
    slope = float(np.polyfit(beta.numpy(), Sc[:, 2].numpy(), 1)[0])           # focal spot z per radian of rotation
    dbeta = float(np.abs(np.diff(beta.numpy())).mean())
    gam = meta.phis_det[:, 0].double()
    vdet = meta.zs_det[0, :].double()
    dgam, dv = float(gam[1] - gam[0]), float(vdet[1] - vdet[0])
    x = ((torch.arange(Nx) - (Nx - 1) / 2) * dxy).to(device)
    y = ((torch.arange(Ny) - (Ny - 1) / 2) * float(dr[1])).to(device)
    zc = ((torch.arange(Nz) - (Nz - 1) / 2) * dz).to(device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    fov = (X ** 2 + Y ** 2) <= fov_radius ** 2
    num = torch.zeros((Nx, Ny, Nz), device=device)
    half_cov = (nrow / 2 * abs(dv)) * (rho + fov_radius + 10) / DSD + 1.0
    shifts_same = [2 * np.pi * m for m in range(-m_range, m_range + 1) if m != 0]
    shifts_opposite = [np.pi + 2 * np.pi * m for m in range(-m_range, m_range)]
    b_lo, b_hi = float(beta.min()), float(beta.max())
    for s in range(0, N, batch):
        e = min(N, s + batch)
        zs_b = Sc[s:e, 2]
        k0 = max(0, int(np.floor((float(zs_b.min()) - half_cov - float(zc[0])) / dz)))
        k1 = min(Nz, int(np.ceil((float(zs_b.max()) + half_cov - float(zc[0])) / dz)) + 1)
        if k1 <= k0:
            continue
        b = e - s
        Scb = Sc[s:e].to(device, torch.float32)
        dX = X[None] - Scb[:, 0, None, None]
        dY = Y[None] - Scb[:, 1, None, None]
        L2 = dX ** 2 + dY ** 2
        L = torch.sqrt(L2)
        bb = beta[s:e].to(device, torch.float32)
        gamma = torch.remainder(torch.atan2(dY, dX) - (torch.remainder(bb, 2 * np.pi)[:, None, None] + np.pi) + np.pi, 2 * np.pi) - np.pi
        col = (gamma - float(gam[0])) / dgam
        dZ = zc[k0:k1][None, None, None, :] - Scb[:, 2, None, None, None]
        to_row = lambda dz_, Lx: (dz_ * (DSD / Lx[..., None]) - float(vdet[0])) / dv
        row = to_row(dZ, L)
        W = _window(row, nrow, taper_rows) * ((col >= 0) & (col <= ncol - 1) & fov)[..., None]
        total = W.clone()
        for sh in shifts_same:
            scanned = ((bb + sh >= b_lo) & (bb + sh <= b_hi)).float()[:, None, None, None]
            total += scanned * _window(to_row(dZ - slope * sh, L), nrow, taper_rows)
        Lo = (2 * rho * torch.cos(gamma) - L).clamp(min=1.0)
        for sh in shifts_opposite:
            bo = bb[:, None, None] + sh + 2 * gamma
            scanned = ((bo >= b_lo) & (bo <= b_hi)).float()[..., None]
            total += scanned * _window(to_row(dZ - slope * (sh + 2 * gamma)[..., None], Lo), nrow, taper_rows)
        w = torch.where(total > 0, W / total.clamp(min=1e-6), torch.zeros_like(W))
        grid = torch.stack([(row / (nrow - 1)) * 2 - 1, ((col / (ncol - 1)) * 2 - 1)[..., None].expand_as(row)], dim=-1)
        Qb = Q[s:e].to(device, non_blocking=True)
        samp = F.grid_sample(Qb[:, None], grid.reshape(b, Nx * Ny, k1 - k0, 2), mode='bilinear', align_corners=True)
        num[:, :, k0:k1] += (w * samp.reshape(b, Nx, Ny, k1 - k0) / L2[..., None]).sum(0)
    # the kernel carries the 1/2 of a full scan, where every line is measured twice; here the weights of a line sum to 1
    return 2 * dbeta * num


def fdk(proj: torch.Tensor, meta, shape: tuple, dr: tuple, taper_rows: int = 4, apodization: str = 'hann',
        fov_radius: float = 250.0, device: str = 'cuda', batch: int = 4) -> tuple:
    """Helical FDK reconstruction (attenuation per mm, centred as CTGen3SystemMatrix centres the object), with timings."""
    torch.cuda.synchronize(); t0 = time.perf_counter()
    Q = ramp_filter(proj, meta, apodization, device)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    recon = backproject(Q, meta, shape, dr, taper_rows, fov_radius, device, batch)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    return recon, dict(filter_s=t1 - t0, backproject_s=t2 - t1)
