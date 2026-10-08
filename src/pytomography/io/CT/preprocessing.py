"""Corrections of DICOM-CT-PD projections (line integrals) before reconstruction. They change the data, so they apply
to every reconstruction (filtered back projection and the iterative algorithms alike). The DICOM-CT-PD reader applies
:func:`filter_low_signal` by default and :func:`scale_columns` on request."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
import pytomography
from pytomography.utils.memory import gpu_budget

#: Neighbourhoods (views, columns, rows) of :func:`filter_low_signal`, in order of size.
LOW_SIGNAL_SIZES = ((1, 1, 1), (1, 3, 1), (1, 3, 3), (3, 3, 3), (3, 5, 3), (5, 5, 3), (5, 7, 5))


def _box(a: torch.Tensor, size) -> torch.Tensor:
    """Mean over a box of odd ``size`` (views, columns, rows) with the edges replicated, as scipy's
    ``uniform_filter(mode='nearest')``."""
    pv, pc, pr = (s // 2 for s in size)
    x = F.pad(a[None, None], (pr, pr, pc, pc, pv, pv), mode='replicate')
    return F.avg_pool3d(x, tuple(size), stride=1)[0, 0]


def _filter_low_signal_gpu(proj: torch.Tensor, n0: torch.Tensor, n_target: float, v0: int, v1: int, device, budget,
                           chunk: int | None) -> torch.Tensor:
    """:func:`filter_low_signal` on a GPU, a chunk of views at a time within ``budget``."""
    V, C, R = proj.shape
    out = proj.to(torch.float32).clone()
    logk = torch.tensor(np.log([a * b * c for a, b, c in LOW_SIGNAL_SIZES]), dtype=torch.float32, device=device)
    margin = max(s[0] for s in LOW_SIGNAL_SIZES) // 2 + 1
    if chunk is None:
        chunk = int(max(8, gpu_budget(budget, device) // (C * R * 4 * 12) - 2 * margin))
    for s0 in range(v0, v1, chunk):
        s1 = min(v1, s0 + chunk)
        lo, hi = max(0, s0 - margin), min(V, s1 + margin)
        p = proj[lo:hi].to(device, torch.float32)
        trans = torch.exp(-p)
        n_est = n0[lo:hi, :, None].to(device, torch.float32) * _box(trans, (3, 3, 3))
        need = torch.clamp(n_target / torch.clamp(n_est, min=1e-9), 1.0, float(torch.exp(logk[-1])))
        del n_est
        if float(need.max()) <= 1.0:
            continue
        ln = torch.log(need)
        i = torch.clamp(torch.searchsorted(logk, ln.contiguous(), right=True) - 1, 0, len(logk) - 2)
        x = torch.clamp(i + (ln - logk[i]) / (logk[i + 1] - logk[i]), 0, len(logk) - 1)    # fractional size index
        del ln, i
        acc = torch.zeros_like(trans)
        for k, size in enumerate(LOW_SIGNAL_SIZES):
            w = torch.clamp(1.0 - (x - k).abs(), 0.0, 1.0)
            if not bool(w.any()):
                continue
            acc += w * (trans if k == 0 else _box(trans, size))
            del w
        sl = slice(s0 - lo, s0 - lo + (s1 - s0))
        out[s0:s1] = torch.where(need[sl] > 1.0, -torch.log(torch.clamp(acc[sl], min=1e-12)), p[sl]).cpu()
        del p, trans, need, x, acc
    return out


def filter_low_signal(projections: torch.Tensor, photon_counts, n_target: float = 30.0, views: tuple | None = None,
                      chunk: int | None = None, device=None, budget: float | None = None) -> torch.Tensor:
    r"""Adaptive filtering of photon-starved rays, in the spirit of Hsieh (Med. Phys. 25, 2139, 1998) and Kachelriess
    et al. (Med. Phys. 28, 475, 2001).

    A ray that expects :math:`N = N_0 e^{-p}` photons has a line integral with a variance of about :math:`1/N`, and at
    a few photons the logarithm is biased as well: these rays make the streaks between the shoulders, or across the
    abdomen beside the arms. Each ray with :math:`N` below ``n_target`` is replaced by :math:`-\log` of the mean
    transmission :math:`e^{-p}` over a neighbourhood (views, columns, rows) of about ``n_target / N`` rays, so about
    ``n_target`` photons; the neighbourhood grows through :data:`LOW_SIGNAL_SIZES`, blending between neighbouring sizes.
    Averaging transmission rather than line integrals keeps the mean right. :math:`N` is estimated from transmission
    smoothed over 3 x 3 x 3 rays, so the choice of filter is not driven by the noise itself. Rays with enough photons
    are returned exactly as they were. It runs a chunk of views at a time: on a CUDA ``device`` within ``budget``
    bytes, otherwise on the host.

    Args:
        projections (torch.Tensor): Line integrals (views, columns, rows), in acquisition order.
        photon_counts (array): Incident photons per detector column of every view (views, columns), as DICOM-CT-PD
            stores them in (7033,1065) PhotonStatistics.
        n_target (float, optional): Photons a filtered ray should represent. Defaults to 30.
        views (tuple, optional): (start, stop) range of views to filter; the others are returned unchanged.
        chunk (int, optional): Views processed at a time. Defaults to what fits the budget (GPU) or 400 (host).
        device (str, optional): Where to compute. Defaults to ``pytomography.device``.
        budget (float, optional): GPU memory budget in bytes (see :func:`pytomography.utils.gpu_budget`).

    Returns:
        torch.Tensor: Filtered line integrals (float32), on the device of ``projections``.
    """
    out_device = projections.device if isinstance(projections, torch.Tensor) else torch.device('cpu')
    work = torch.device(pytomography.device if device is None else device)
    if work.type == 'cuda':
        proj_t = torch.as_tensor(projections).detach().cpu()
        n0_t = torch.as_tensor(photon_counts).detach().cpu()
        if tuple(n0_t.shape) != tuple(proj_t.shape[:2]):
            raise ValueError(f'photon counts {tuple(n0_t.shape)} do not match the projections {tuple(proj_t.shape[:2])} (views, columns)')
        v0, v1 = views if views is not None else (0, proj_t.shape[0])
        return _filter_low_signal_gpu(proj_t, n0_t, n_target, v0, v1, work, budget, chunk).to(out_device)
    chunk = 400 if chunk is None else chunk
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
    return torch.from_numpy(out).to(out_device)


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

    This is the form a bowtie-dependent (per column) beam hardening calibration takes. For GE Discovery CT750 HD
    scans at 100 kV, :func:`fit_column_scale` finds :math:`g_0 = -1.2\%`, :math:`g_2 = +1.65\%` from the scanner's own
    images of TCIA LDCT-and-Projection-data C145, and :math:`-1.35\%`, :math:`+1.84\%` from C001's: without the scale,
    filtered back projection reads soft tissue 25-35 HU above the scanner's at the centre and 25 HU below at 16-20 cm;
    with it, within 6 HU everywhere. It matches the scanner's values, which are not necessarily the true ones, so it is
    not applied unless asked for.

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


def fit_column_scale(projections: torch.Tensor, proj_meta, system_matrix, reference, mask=None, t_hold: float = 140.0,
                     sigma: float = 2.0, **fbp_options) -> dict:
    r"""Fit the per-column scale of :func:`scale_columns` that makes the filtered back projection of ``projections``
    match ``reference``, another reconstruction of the same scan on the object grid of ``system_matrix``: typically the
    scanner's own images, resampled onto that grid (for example with :func:`pytomography.io.shared.align_images_affine`
    and :meth:`~pytomography.metadata.CT.CTGen3ProjMeta.get_patient_affine`).

    Reconstruction is linear in the line integrals, so scaling them by :math:`1 + g_0 + g_2 u^2`, with
    :math:`u = \min(|t|, t_\mathrm{hold}) / 100\,\mathrm{mm}`, changes the image by :math:`g_0 f_0 + g_2 f_2`, where
    :math:`f_0` reconstructs the projections and :math:`f_2` the projections weighted by :math:`u^2`. The fit takes these
    two reconstructions and solves for :math:`g_0, g_2` by least squares over ``mask``, after a Gaussian blur of
    ``sigma`` mm, so that differences in resolution and noise do not count.

    Args:
        projections (torch.Tensor): Line integrals as read (views, columns, rows), before any column scale.
        proj_meta (CTGen3ProjMeta): Their metadata.
        system_matrix (CTGen3SystemMatrix): System matrix of the projections; its object grid is the grid of
            ``reference``.
        reference (torch.Tensor | numpy.ndarray): The image to match, in attenuation per mm, on that grid.
        mask (array of bool, optional): Voxels to fit. Defaults to the soft tissue and fat of the blurred reference
            (-200 to 200 HU, with ``proj_meta.water_attenuation``), 2 sigma away from other materials.
        t_hold (float, optional): As in :func:`column_scale`. Defaults to 140.
        sigma (float, optional): Blur (mm) before the fit. Defaults to 2.
        **fbp_options: Passed to :class:`~pytomography.algorithms.FilteredBackProjection` (``filter``,
            ``slice_thickness``, ...).

    Returns:
        dict: ``g0``, ``g2`` and ``t_hold``, to pass as ``column_scale`` to
        :func:`~pytomography.io.CT.dicom_ct_pd.get_projections_and_metadata_gen3` or as keywords to
        :func:`scale_columns`.
    """
    from pytomography.algorithms import FilteredBackProjection
    u2 = (column_scale(proj_meta, 0.0, 1.0, t_hold) - 1).to(torch.float32)
    f0 = FilteredBackProjection(projections, system_matrix, **fbp_options)().cpu().numpy().astype(np.float64)
    weighted = projections.to(torch.float32) * u2.to(projections.device)[None, :, None]
    f2 = FilteredBackProjection(weighted, system_matrix, **fbp_options)().cpu().numpy().astype(np.float64)
    del weighted
    ref = (reference.detach().cpu().numpy() if isinstance(reference, torch.Tensor) else np.asarray(reference)).astype(np.float64)
    if ref.shape != f0.shape:
        raise ValueError(f'reference is {ref.shape}, but the object grid of the system matrix is {f0.shape}')
    px = sigma / np.asarray(system_matrix.object_meta.dr, dtype=np.float64)
    B0, B2, R = (ndimage.gaussian_filter(a, px) for a in (f0, f2, ref))
    if mask is None:
        mu_w = getattr(proj_meta, 'water_attenuation', None)
        if not mu_w:
            raise ValueError('pass a mask: these projections carry no water attenuation to find soft tissue with')
        hu = 1000 * (R / mu_w - 1)
        soft = ((hu > -200) & (hu < 200) & (f0 != 0)).astype(np.float32)
        size = tuple(2 * max(1, int(round(2 * p))) + 1 for p in px)       # a box 2 sigma either way
        mask = ndimage.uniform_filter(soft, size, mode='constant') > 0.999
    mask = np.asarray(mask, bool)
    if mask.sum() < 100:
        raise ValueError(f'only {int(mask.sum())} voxels to fit; widen the mask')
    A = np.stack([B0[mask], B2[mask]], axis=1)
    g0, g2 = np.linalg.lstsq(A, (R - B0)[mask], rcond=None)[0]
    return dict(g0=float(g0), g2=float(g2), t_hold=float(t_hold))
