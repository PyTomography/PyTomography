"""Corrections of DICOM-CT-PD projections (line integrals) before reconstruction. They change the data, so they apply
to every reconstruction (filtered back projection and the iterative algorithms alike). The DICOM-CT-PD reader applies
:func:`filter_low_signal` by default and :func:`scale_columns` on request."""
from __future__ import annotations
import numpy as np
import torch
from scipy import ndimage

#: Neighbourhoods (views, columns, rows) of :func:`filter_low_signal`, in order of size.
LOW_SIGNAL_SIZES = ((1, 1, 1), (1, 3, 1), (1, 3, 3), (3, 3, 3), (3, 5, 3), (5, 5, 3), (5, 7, 5))


def filter_low_signal(projections: torch.Tensor, photon_counts, n_target: float = 30.0, views: tuple | None = None,
                      chunk: int = 400) -> torch.Tensor:
    r"""Adaptive filtering of photon-starved rays, in the spirit of Hsieh (Med. Phys. 25, 2139, 1998) and Kachelriess
    et al. (Med. Phys. 28, 475, 2001).

    A ray that expects :math:`N = N_0 e^{-p}` photons has a line integral with a variance of about :math:`1/N`, and at
    a few photons the logarithm is biased as well: these rays make the streaks between the shoulders, or across the
    abdomen beside the arms. Each ray with :math:`N` below ``n_target`` is replaced by :math:`-\log` of the mean
    transmission :math:`e^{-p}` over a neighbourhood (views, columns, rows) of about ``n_target / N`` rays, so about
    ``n_target`` photons; the neighbourhood grows through :data:`LOW_SIGNAL_SIZES`, blending between neighbouring sizes.
    Averaging transmission rather than line integrals keeps the mean right. :math:`N` is estimated from transmission
    smoothed over 3 x 3 x 3 rays, so the choice of filter is not driven by the noise itself. Rays with enough photons
    are returned exactly as they were. Runs on the host, a chunk of views at a time.

    Args:
        projections (torch.Tensor): Line integrals (views, columns, rows), in acquisition order.
        photon_counts (array): Incident photons per detector column of every view (views, columns), as DICOM-CT-PD
            stores them in (7033,1065) PhotonStatistics.
        n_target (float, optional): Photons a filtered ray should represent. Defaults to 30.
        views (tuple, optional): (start, stop) range of views to filter; the others are returned unchanged.
        chunk (int, optional): Views processed at a time. Defaults to 400.

    Returns:
        torch.Tensor: Filtered line integrals (float32), on the device of ``projections``.
    """
    device = projections.device if isinstance(projections, torch.Tensor) else torch.device('cpu')
    proj = projections.detach().cpu().numpy() if isinstance(projections, torch.Tensor) else np.asarray(projections)
    n0 = photon_counts.detach().cpu().numpy() if isinstance(photon_counts, torch.Tensor) else np.asarray(photon_counts)
    if n0.shape != proj.shape[:2]:
        raise ValueError(f'photon counts {n0.shape} do not match the projections {proj.shape[:2]} (views, columns)')
    out = np.array(proj, dtype=np.float32, copy=True)
    V = proj.shape[0]
    v0, v1 = views if views is not None else (0, V)
    k = np.array([a * b * c for a, b, c in LOW_SIGNAL_SIZES], dtype=np.float64)
    logk = np.log(k)
    margin = max(s[0] for s in LOW_SIGNAL_SIZES) // 2 + 1
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
        for i, size in enumerate(LOW_SIGNAL_SIZES):
            w = np.clip(1.0 - np.abs(x - i), 0.0, 1.0).astype(np.float32)
            if not w.any():
                continue
            acc += w * (trans if i == 0 else ndimage.uniform_filter(trans, size, mode='nearest'))
        filtered = -np.log(np.maximum(acc, 1e-12))
        sl = slice(s0 - lo, s0 - lo + (s1 - s0))
        out[s0:s1] = np.where(need[sl] > 1.0, filtered[sl], out[s0:s1])
    return torch.from_numpy(out).to(device)


def column_scale(proj_meta, g0: float, g2: float, t_hold: float = 140.0) -> torch.Tensor:
    r"""Factors :math:`1 + g(t)` of :func:`scale_columns` for the detector columns of a
    :class:`~pytomography.metadata.CT.CTGen3ProjMeta`, with :math:`g(t) = g_0 + g_2 (t / 100\,\mathrm{mm})^2`, held
    constant beyond ``t_hold``, and :math:`t = \rho \sin\gamma` the distance of a column's rays from the isocentre.

    Returns:
        torch.Tensor: (columns,) float64.
    """
    rho = float(proj_meta.source_rhos.double().mean())
    t = torch.abs(rho * torch.sin(proj_meta.phis_det[:, 0].double()))
    return 1 + g0 + g2 * (torch.clamp(t, max=t_hold) / 100) ** 2


def scale_columns(projections: torch.Tensor, proj_meta, g0: float, g2: float, t_hold: float = 140.0) -> torch.Tensor:
    r"""Scale the line integrals of every detector column by :math:`1 + g(t)` (see :func:`column_scale`).

    This is the form a bowtie-dependent (per column) beam hardening calibration takes. For the GE scan TCIA
    LDCT-and-Projection-data C145, the scanner's own images relate to the exported projections by
    :math:`g_0 = -1.49\%`, :math:`g_2 = +1.86\%`: with that scale, filtered back projection matches the scanner's soft
    tissue, fat and lung within 3 HU, without the radial trend (+24 HU at the centre, -16 HU at 140-180 mm) left
    otherwise. The coefficients come from one scan, so this is not applied unless asked for.

    Args:
        projections (torch.Tensor): Line integrals (views, columns, rows).
        proj_meta (CTGen3ProjMeta): Metadata of the projections.
        g0 (float): Scale at the centre of the detector.
        g2 (float): Coefficient of :math:`(t / 100\,\mathrm{mm})^2`.
        t_hold (float, optional): Distance (mm) beyond which the scale is held constant. Defaults to 140.

    Returns:
        torch.Tensor: Scaled line integrals, float32, on the device of ``projections``.
    """
    factors = column_scale(proj_meta, g0, g2, t_hold).to(torch.float32).to(projections.device)
    return projections.to(torch.float32) * factors[None, :, None]
