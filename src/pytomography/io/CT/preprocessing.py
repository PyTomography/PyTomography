"""Corrections of DICOM-CT-PD projections (line integrals) before reconstruction. They change the data, so they apply
to every reconstruction (filtered back projection and the iterative algorithms alike). The DICOM-CT-PD reader applies
:func:`filter_low_signal` by default and :func:`scale_columns` on request.

Two fits reproduce a scanner's own images from its projections: :func:`fit_column_scale` (the per-column scale of
:func:`scale_columns`) and :func:`fit_window` (the window of filtered back projection, the scanner's kernel)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
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


def _image(a) -> np.ndarray:
    """An image (tensor or array) as a float32 array on the host."""
    a = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    return a.astype(np.float32, copy=False)


def _soft_tissue(blurred_reference: np.ndarray, image: np.ndarray, proj_meta, px, hu_range=(-200, 200)) -> np.ndarray:
    """Voxels of a reference blurred by ``px`` voxels within ``hu_range`` (with ``proj_meta.water_attenuation``; by
    default soft tissue and fat), inside the field of view of ``image`` (not zero there) and ``2 px`` away from
    anything outside that range."""
    mu_w = getattr(proj_meta, 'water_attenuation', None)
    if not mu_w:
        raise ValueError('pass a mask: these projections carry no water attenuation to find soft tissue with')
    hu = 1000 * (blurred_reference / mu_w - 1)
    soft = ((hu > hu_range[0]) & (hu < hu_range[1]) & (image != 0)).astype(np.float32)
    size = tuple(2 * max(1, int(round(2 * p))) + 1 for p in px)       # a box 2 sigma either way
    return ndimage.uniform_filter(soft, size, mode='constant') > 0.999


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
    ref = _image(reference)
    u2 = (column_scale(proj_meta, 0.0, 1.0, t_hold) - 1).to(torch.float32)
    f0 = _image(FilteredBackProjection(projections, system_matrix, **fbp_options)())
    if ref.shape != f0.shape:
        raise ValueError(f'reference is {ref.shape}, but the object grid of the system matrix is {f0.shape}')
    weighted = projections.to(torch.float32) * u2.to(projections.device)[None, :, None]
    f2 = _image(FilteredBackProjection(weighted, system_matrix, **fbp_options)())
    del weighted
    px = sigma / np.asarray(system_matrix.object_meta.dr, dtype=np.float64)
    B0, B2, R = (ndimage.gaussian_filter(a, px) for a in (f0, f2, ref))          # float32, like the images
    del f2
    if mask is None:
        mask = _soft_tissue(R, f0, proj_meta, px)
    mask = np.asarray(mask, bool)
    if mask.sum() < 100:
        raise ValueError(f'only {int(mask.sum())} voxels to fit; widen the mask')
    A = np.stack([B0[mask], B2[mask]], axis=1).astype(np.float64)
    g0, g2 = np.linalg.lstsq(A, R[mask].astype(np.float64) - A[:, 0], rcond=None)[0]
    return dict(g0=float(g0), g2=float(g2), t_hold=float(t_hold))


def _shared_noise_window(f: np.ndarray, ref: np.ndarray, mask: np.ndarray, dr, block: float) -> tuple:
    """The least-squares window, ring by ring, that takes the image ``f`` to ``ref`` over the blocks of ``block`` mm
    inside ``mask`` (see :func:`fit_window`). Returns the centres of the rings (cycles per mm), from the first ring
    above zero to the Nyquist frequency of the grid, and the window there."""
    n = tuple(int(round(block / d)) for d in dr[:2])
    if min(n) < 8 or n[0] > f.shape[0] or n[1] > f.shape[1]:
        raise ValueError(f'blocks of {block} mm are {n} voxels; they need at least 8, and no more than the image')
    step = (n[0] // 2, n[1] // 2)
    # blocks on a grid with half overlap, in every slice, that lie wholly inside the mask
    inside = sliding_window_view(mask, n, axis=(0, 1))[::step[0], ::step[1]].all(axis=(-2, -1))
    bi, bj, bk = np.nonzero(inside)
    if len(bi) < 10:
        raise ValueError(f'only {len(bi)} blocks of {block} mm lie inside the mask; use smaller blocks or a wider mask')
    u, v = np.meshgrid(np.linspace(-1, 1, n[0]), np.linspace(-1, 1, n[1]), indexing='ij')
    basis = np.stack([np.ones_like(u), u, v, u * u, v * v, u * v], axis=-1).reshape(-1, 6)
    pinv = np.linalg.pinv(basis)
    hann = np.outer(np.hanning(n[0]), np.hanning(n[1])).ravel()
    f_blocks, r_blocks = sliding_window_view(f, n, axis=(0, 1)), sliding_window_view(ref, n, axis=(0, 1))
    S_ff, S_rf = np.zeros(n), np.zeros(n)
    for s in range(0, len(bi), 2048):
        at = (bi[s:s + 2048] * step[0], bj[s:s + 2048] * step[1], bk[s:s + 2048])
        a = f_blocks[at].reshape(-1, n[0] * n[1]).astype(np.float64)
        b = r_blocks[at].reshape(-1, n[0] * n[1]).astype(np.float64)
        Fa = np.fft.fft2(((a - (a @ pinv.T) @ basis.T) * hann).reshape(-1, *n))
        Fb = np.fft.fft2(((b - (b @ pinv.T) @ basis.T) * hann).reshape(-1, *n))
        S_ff += (Fa.real ** 2 + Fa.imag ** 2).sum(0)
        S_rf += (Fb * Fa.conj()).real.sum(0)
    nu = np.hypot(*np.meshgrid(np.fft.fftfreq(n[0], d=dr[0]), np.fft.fftfreq(n[1], d=dr[1]), indexing='ij'))
    width = max(1 / (n[0] * dr[0]), 1 / (n[1] * dr[1]))
    n_rings = int(np.floor(0.5 / max(dr[0], dr[1]) / width + 1e-9))         # whole rings below the Nyquist frequency
    ring = np.minimum((nu / width + 1e-9).astype(int), n_rings).ravel()
    W = (np.bincount(ring, S_rf.ravel(), n_rings + 1) / np.bincount(ring, S_ff.ravel(), n_rings + 1))[1:n_rings]
    centres = (np.arange(1, n_rings) + 0.5) * width                         # ring 0 is what detrending removes
    return centres, W


def fit_window(projections: torch.Tensor, proj_meta, system_matrix, reference, mask=None, block: float = 32.0,
               normalize_below: float | None = 0.15, refinements: int = 1, taper: float = 0.1, **fbp_options):
    r"""Fit the window of filtered back projection that gives the reconstruction of ``projections`` the sharpness and
    noise texture of ``reference``, another reconstruction of the same scan on the object grid of ``system_matrix``:
    typically the scanner's own images, resampled onto that grid as for :func:`fit_column_scale`. The window then
    stands in for the scanner's reconstruction kernel.

    Both images come from the same projections, so they share their noise, which spans every frequency. The fit
    reconstructs the projections with the plain ramp (Ram-Lak), :math:`f`, and finds, at each spatial frequency
    :math:`\nu` (cycles per mm), the window that brings :math:`f` closest to the reference :math:`r` by least squares,

    .. math:: W(\nu) = \frac{\mathrm{Re} \sum R\, F^*}{\sum |F|^2},

    with :math:`F` and :math:`R` the in-plane Fourier transforms of the two images over square blocks of ``block`` mm
    that lie inside ``mask``, every slice of a block detrended (by a quadratic surface) and tapered (by a Hann window),
    and the sums taken over the blocks, their slices, and a ring of frequencies around :math:`\nu`.

    Two corrections follow. The noise the images share is scaled by how much their slice profiles overlap, by the
    same factor at every frequency (about 0.92 for GE's images of TCIA LDCT-and-Projection-data C145), so the window
    is scaled to 1 at low frequencies, the rings below ``normalize_below``: it sets sharpness and noise, and leaves the
    CT numbers of large regions to :func:`fit_column_scale`. And where the projections are sampled more finely than
    the object grid, the plain ramp image aliases strong high frequencies into the rings below the Nyquist frequency,
    which pulls the window down there; each refinement reconstructs with the window found so far and multiplies it by
    the ratio fitted between that image and the reference. On a phantom, one refinement brings the fit within 0.01 of
    the true window, and within 0.02 below the top ring on a grid much coarser than the projections.

    Args:
        projections (torch.Tensor): Line integrals (views, columns, rows), with any column scale already applied.
        proj_meta (CTGen3ProjMeta): Their metadata.
        system_matrix (CTGen3SystemMatrix): System matrix of the projections; its object grid is the grid of
            ``reference``.
        reference (torch.Tensor | numpy.ndarray): The image to match, in attenuation per mm, on that grid.
        mask (array of bool, optional): Voxels the blocks must lie in, which should be uniform: the taper leaks the
            strong low frequencies of an edge into the frequencies around them, where they would pull the window up.
            Defaults to uniform soft tissue: the reference blurred by 2 mm between -50 and 100 HU (with
            ``proj_meta.water_attenuation``; so neither fat nor bone), 4 mm or more from anything else.
        block (float, optional): Side of the blocks (mm); the table is spaced 1 / ``block`` cycles per mm. Defaults
            to 32.
        normalize_below (float | None, optional): Frequency (cycles per mm) below which the window is 1 on average;
            None keeps the ratio as fitted. Defaults to 0.15.
        refinements (int, optional): Refinements, each one more reconstruction. Defaults to 1.
        taper (float, optional): Beyond the table, which ends at the Nyquist frequency of the object grid, the window
            falls linearly to zero over this many cycles per mm. Defaults to 0.1.
        **fbp_options: Passed to :class:`~pytomography.algorithms.FilteredBackProjection` (``slice_thickness``, ...),
            except ``filter``.

    Returns:
        TabulatedFilter: The window, to pass as ``filter`` to :class:`~pytomography.algorithms.FilteredBackProjection`;
        its ``frequencies`` and ``values`` hold the table.
    """
    from pytomography.algorithms import FilteredBackProjection
    from pytomography.utils import TabulatedFilter
    if 'filter' in fbp_options:
        raise TypeError('fit_window finds the filter: it reconstructs with the plain ramp, so do not pass one')
    ref = _image(reference)
    f = _image(FilteredBackProjection(projections, system_matrix, filter='ram-lak', **fbp_options)())
    if ref.shape != f.shape:
        raise ValueError(f'reference is {ref.shape}, but the object grid of the system matrix is {f.shape}')
    dr = np.asarray(system_matrix.object_meta.dr, dtype=np.float64)
    if mask is None:
        px = 2.0 / dr
        mask = _soft_tissue(ndimage.gaussian_filter(ref, px), f, proj_meta, px, hu_range=(-50, 100))
    mask = np.asarray(mask, bool)
    centres, W = _shared_noise_window(f, ref, mask, dr, block)
    del f

    def normalized(W):
        low = centres < (normalize_below or 0)
        return W / W[low].mean() if low.any() else W

    def table(W):
        return TabulatedFilter(np.concatenate([[0.0], centres]), np.concatenate([[1.0], np.clip(W, 0, None)]), taper=taper)

    W = normalized(W)
    for _ in range(refinements):
        g = _image(FilteredBackProjection(projections, system_matrix, filter=table(W), **fbp_options)())
        W = W * normalized(_shared_noise_window(g, ref, mask, dr, block)[1])
        del g
    return table(W)
