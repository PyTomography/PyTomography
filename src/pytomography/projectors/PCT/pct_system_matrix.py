"""System matrix of proton CT list mode data (:class:`PCTSystemMatrix`): one row per proton, the length of its estimated
path in each voxel."""
from __future__ import annotations
import functools
import math
import warnings
import numpy as np
import torch
import pytomography
from pytomography.projectors import SystemMatrix
from pytomography.projectors.system_matrix import on_device
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PCT import PCTProjMeta
from pytomography.metadata.PCT.pct_metadata import FWHM_PER_SD
from pytomography.utils.fourier_filters import get_fbp_filter, ramp_filter
from pytomography.utils.memory import MemoryPart, gpu_budget
from . import _pct_cuda

#: The path models :class:`PCTSystemMatrix` knows.
PATHS = ('straight', 'spline')
#: The lines filtered back projection can bin each proton by.
LINES = ('points', 'hull')
#: A band of paths spans this many standard deviations of the entry's spread, in x and in z.
BAND_SPAN = 5.0
#: The R2 low discrepancy sequence that shifts each proton's band.
_R2 = (0.7548776662466927, 0.5698402909980532)
#: Device bytes per path of one projector call at its peak (the band's paths and temporaries, their projections):
#: 89 measured on an RTX 5090 for 5 x 5 bands (30 Oct 2026), rounded up.
_BYTES_PER_PATH = 100.0
#: The same for filtered back projection's binning (the paths, their hull crossings and the binning's temporaries):
#: 208 measured for 5 x 5 bands binned by the hull chord, rounded up.
_BIN_BYTES_PER_PATH = 220.0
#: Paths binned at a time by filtered back projection.
_BIN_CHUNK = 2**24


@functools.lru_cache
def _band_ratio(n_side: int, n_shifts: int = 2001) -> float:
    """The band weights' Gaussian width over the entry's standard deviation that gives a proton's weighted spread that
    standard deviation on average over the band's shifts (the grid's edges cut off the tails: 1.0687 for 5 x 5)."""
    u = np.linspace(-0.5, 0.5, n_shifts, endpoint=False) + 0.5 / n_shifts
    x = ((np.arange(n_side) - (n_side - 1) / 2)[None, :] + u[:, None]) * (BAND_SPAN / n_side)

    def mean_sd(sw):
        w = np.exp(-x ** 2 / (2 * sw ** 2))
        w /= w.sum(1, keepdims=True)
        m = (w * x).sum(1)
        return np.sqrt((w * (x - m[:, None]) ** 2).sum(1)).mean()
    lo, hi = 0.5, 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mean_sd(mid) < 1.0 else (lo, mid)
    return 0.5 * (lo + hi)


def band_paths(protons: dict, n_side: int, beam_divergence: float) -> dict:
    r"""Each proton's band of paths: its unknown entry spread over an ``n_side`` x ``n_side`` grid on the entry plane,
    in the view's x and z, :data:`BAND_SPAN` standard deviations of its entry spread across, each point a path with a
    Gaussian weight (the weights summing to 1 per proton, their width chosen so that the weighted spread is the
    entry's on average). The grid is shifted per proton by the R2 low discrepancy sequence of its band index (within
    one spacing), so that the protons of a pencil beam do not share one lattice of entries; each point's direction
    turns with its offset by ``beam_divergence``. With ``n_side`` 1 the band is the stored entry alone.

    Args:
        protons (dict): A view's protons, as :meth:`~pytomography.metadata.PCT.PCTProjMeta.load` gives them.
        n_side (int): Paths per side of the grid (a number).
        beam_divergence (float): Turn of the direction per mm of offset, in degrees per mm.

    Returns:
        dict: ``entry_points``, ``entry_directions``, ``exit_points``, ``exit_directions`` ([n K, 3], mm and unit
        vectors) and ``weights`` ([n K], no units), ``K = n_side ** 2`` consecutive paths per proton.
    """
    n = protons['entry_points'].shape[0]
    device = protons['entry_points'].device
    if n_side == 1:
        return {**protons, 'weights': torch.ones(n, dtype=torch.float32, device=device)}
    K = n_side * n_side
    sd = protons['entry_fwhm'] / FWHM_PER_SD
    i = protons['index'].double()
    u = torch.remainder(0.5 + _R2[0] * i, 1.0) - 0.5
    v = torch.remainder(0.5 + _R2[1] * i, 1.0) - 0.5
    k = torch.arange(n_side, device=device, dtype=torch.float64) - (n_side - 1) / 2
    spacing = (BAND_SPAN * sd / n_side).double()[:, None]
    sigma = (sd * _band_ratio(n_side)).double()[:, None].clamp(min=1e-12)
    gx = (k[None, :] + u[:, None]) * spacing
    gz = (k[None, :] + v[:, None]) * spacing
    ox = gx[:, :, None].expand(-1, -1, n_side).reshape(n, -1)
    oz = gz[:, None, :].expand(-1, n_side, -1).reshape(n, -1)
    w = torch.exp(-(ox ** 2 + oz ** 2) / (2 * sigma ** 2))
    w = (w / w.sum(dim=1, keepdim=True)).float()
    ox, oz = ox.float(), oz.float()
    c, s = torch.cos(protons['angle'])[:, None], torch.sin(protons['angle'])[:, None]
    off = torch.stack([c * ox, -s * ox, oz], dim=2)                 # the offsets in the object frame, [n, K, 3]
    e = protons['entry_directions'][:, None, :] + math.radians(beam_divergence) * off
    return {'entry_points': (protons['entry_points'][:, None, :] + off).reshape(n * K, 3),
            'entry_directions': (e / e.norm(dim=2, keepdim=True)).reshape(n * K, 3),
            'exit_points': protons['exit_points'].repeat_interleave(K, dim=0),
            'exit_directions': protons['exit_directions'].repeat_interleave(K, dim=0),
            'weights': w.reshape(n * K)}


def hull_from_image(image: torch.Tensor, object_meta: ObjectMeta, fwhm: float = 4.71, opening: float = 2.0,
                    threshold: float = 0.5, grow: int = 0) -> torch.Tensor:
    """The object's outline from a first reconstruction: smoothed, above ``threshold``, opened, its largest piece,
    then grown by ``grow`` voxels in each axial slice (a square of 3 x 3 voxels at a time).

    Args:
        image (torch.Tensor): RSP, [Lx, Ly, Lz] (no units).
        object_meta (ObjectMeta): Its grid (mm).
        fwhm (float, optional): FWHM of the Gaussian smoothing, in mm. Defaults to 4.71.
        opening (float, optional): Radius of the opening's ellipsoid, in mm. Defaults to 2.
        threshold (float, optional): RSP above which a voxel is inside (no units). Defaults to 0.5.
        grow (int, optional): Voxels to grow the outline by, in x and y. Defaults to 0.

    Returns:
        torch.Tensor: The outline, [Lx, Ly, Lz], bool, on the host.
    """
    from scipy import ndimage
    image = image.cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    dr = np.asarray(object_meta.dr, dtype=np.float64)
    smooth = ndimage.gaussian_filter(image, fwhm / FWHM_PER_SD / dr)
    semi = opening / dr
    r = np.floor(semi).astype(int)
    g = np.mgrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    ball = sum((g[k] / max(semi[k], 1e-9)) ** 2 for k in range(3)) <= 1
    opened = ndimage.binary_opening(smooth > threshold, structure=ball)
    labels, n = ndimage.label(opened)
    hull = labels == (1 + int(np.argmax(ndimage.sum(opened, labels, range(1, n + 1))))) if n else opened
    if grow:
        hull = ndimage.binary_dilation(hull, structure=np.ones((3, 3, 1), bool), iterations=int(grow))
    return torch.as_tensor(hull)


# --- the walk in PyTorch: the CUDA kernels' arithmetic (_pct_cuda), vectorised over paths, for any device

def _first_reach(p, d, dist, shape, dr):
    """How far along ``p + s d`` (local coordinates, [n, 3]) each line first reaches the hull, and whether it does
    (``first_reach`` of the kernels)."""
    n = torch.tensor(shape, device=p.device)
    hi = torch.tensor([float(np.float32(s) * np.float32(v)) for s, v in zip(shape, dr)], device=p.device)
    drt = torch.tensor(dr, dtype=torch.float32, device=p.device)
    dk = torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    t1, t2 = (0 - p) / dk, (hi - p) / dk
    near = torch.minimum(t1, t2).amax(dim=1).clamp(min=0)
    far = torch.maximum(t1, t2).amin(dim=1)
    hit = far > near
    if dist is None:
        return near, hit
    margin = math.sqrt(sum(float(v) ** 2 for v in dr))
    step = 0.5 * min(float(v) for v in dr)
    s_out = torch.zeros_like(near)
    idx = torch.nonzero(hit).ravel()
    hit = torch.zeros_like(hit)
    s = near[idx]
    for _ in range(512):
        if idx.numel() == 0:
            break
        v = torch.minimum(torch.floor((p[idx] + s[:, None] * d[idx]) / drt).long().clamp(min=0), n - 1)
        dv = dist[v[:, 2], v[:, 1], v[:, 0]]
        inside = dv <= 0
        hit[idx[inside]] = True
        s_out[idx[inside]] = s[inside]
        s = s + torch.clamp(dv - margin, min=step)
        keep = ~inside & (s < far[idx])
        idx, s = idx[keep], s[keep]
    return s_out, hit


def _trace(a, b, shape, dr, image=None, sums=None, acc=None, value=None):
    """The straight pieces ``a`` -> ``b`` (local coordinates, [m, 3]) voxel by voxel, as the kernels' ``trace``:
    forward, adds each piece's sum of length times ``image`` and its length to ``sums`` ([m, 2]); back, adds ``value``
    times each length and the weight ``value[:, 1]`` times it to ``acc`` ([voxels, 2]); images x fastest."""
    n = list(shape)
    drt = torch.tensor(dr, dtype=torch.float32, device=a.device)
    d = b - a
    L = torch.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
    ok = L > 1e-7
    t0, t1 = torch.zeros_like(L), torch.ones_like(L)
    for k in range(3):
        hi = float(np.float32(n[k]) * np.float32(dr[k]))
        flat = d[:, k].abs() < 1e-12
        ok &= ~(flat & ((a[:, k] < 0) | (a[:, k] >= hi)))
        dk = torch.where(flat, torch.ones_like(L), d[:, k])
        ta, tb = (0 - a[:, k]) / dk, (hi - a[:, k]) / dk
        t0 = torch.where(flat, t0, torch.maximum(t0, torch.minimum(ta, tb)))
        t1 = torch.where(flat, t1, torch.minimum(t1, torch.maximum(ta, tb)))
    rows = torch.nonzero(ok & (t0 < t1)).ravel()
    a, d, L, t0, t1 = a[rows], d[rows], L[rows], t0[rows], t1[rows]
    tmid = 0.5 * (t0 + t1) - 0.5 * (t1 - t0) * 0.999
    nt = torch.tensor(n, device=a.device)
    c = torch.minimum(torch.floor((a + tmid[:, None] * d) / drt).long().clamp(min=0), nt - 1)
    pos, neg = d > 0, d < 0
    st = pos.long() - neg.long()
    far = torch.full_like(d, 3e38)
    tm = torch.where(pos, ((c + 1) * drt - a) / d, torch.where(neg, (c * drt - a) / d, far))
    td = torch.where(pos, drt / d, torch.where(neg, -drt / d, far))
    t = t0
    for _ in range(4096):
        if rows.numel() == 0:
            break
        k = torch.where((tm[:, 0] <= tm[:, 1]) & (tm[:, 0] <= tm[:, 2]), 0,
                        torch.where(tm[:, 1] <= tm[:, 2], 1, 2))[:, None]
        tn = torch.minimum(tm.gather(1, k)[:, 0], t1)
        seg = (tn - t) * L
        good = seg > 0
        idx = c[:, 0] + n[0] * (c[:, 1] + n[1] * c[:, 2])
        if acc is None:
            sums.index_add_(0, rows, torch.where(good[:, None], torch.stack([seg * image[idx], seg], dim=1), 0))
        else:
            acc.index_add_(0, idx[good], (seg[:, None] * value[rows])[good])
        t = tn
        c.scatter_add_(1, k, st.gather(1, k))
        ck = c.gather(1, k)[:, 0]
        cont = (tn < t1) & (ck >= 0) & (ck < nt[k[:, 0]])
        tm.scatter_add_(1, k, td.gather(1, k))
        rows, a, d, L, t, t1, c, st, tm, td = (x[cont] for x in (rows, a, d, L, t, t1, c, st, tm, td))


def _project_torch(image_x, paths, distance, segments, shape, dr, lo, residual=None, K=1, length=None, acc=None):
    """:func:`._pct_cuda.project` in PyTorch."""
    lo = torch.tensor(lo, dtype=torch.float32, device=paths['entry_points'].device)
    a, b = paths['entry_points'] - lo, paths['exit_points'] - lo
    da, db = paths['entry_directions'], paths['exit_directions']
    s0, hit0 = _first_reach(a, da, distance, shape, dr)
    s1, hit1 = _first_reach(b, -db, distance, shape, dr)
    inside = (hit0 & hit1)[:, None]
    g0, g1 = a + s0[:, None] * da, b - s1[:, None] * db
    chord = torch.sqrt(((g1 - g0) ** 2).sum(dim=1))[:, None]
    points = [a, torch.where(inside, g0, b)]                       # a path that misses the hull: a -> b alone
    for s in range(1, segments + 1):
        t = torch.tensor(s, dtype=torch.float32) / segments
        t2, t3 = t * t, t * t * t
        q = ((2 * t3 - 3 * t2 + 1) * g0 + (t3 - 2 * t2 + t) * chord * da + (-2 * t3 + 3 * t2) * g1
             + (t3 - t2) * chord * db)
        points.append(torch.where(inside, q, b))
    points.append(b)
    if residual is None:
        sums = torch.zeros((a.shape[0], 2), dtype=torch.float32, device=a.device)
        for p, q in zip(points[:-1], points[1:]):
            _trace(p, q, shape, dr, image=image_x, sums=sums)
        return sums[:, 0], sums[:, 1]
    w = paths['weights']
    value = torch.stack([w * residual.repeat_interleave(K), w], dim=1)
    if length is not None:
        value = value * (length > 0)[:, None]
    acc2 = acc.view(-1, 2)
    for p, q in zip(points[:-1], points[1:]):
        _trace(p, q, shape, dr, acc=acc2, value=value)


class PCTSystemMatrix(SystemMatrix):
    r"""System matrix of proton CT list mode data: the forward projection of a relative stopping power (RSP) map is
    each proton's water-equivalent path length, :math:`g_i = \int_{\text{path}_i} \mathrm{RSP}\,dl`, in mm.

    A proton's path is straight from its entry point to where its entry line reaches the ``hull``, a curve inside it
    to where its exit line, followed back, leaves it, and straight to its exit point (straight from entry to exit
    point if it misses the hull). The curve is the cubic spline with the entry and exit directions, each scaled by the
    chord (Collins-Fekete et al. 2015 with :math:`\Lambda = 1`; ``path='spline'``), or the chord (``'straight'``). A
    row of the system matrix is the exact length of the path in each voxel. Without a front tracker a proton's entry
    is uncertain (``proj_meta.entry_fwhm``): its row is then the weighted mean of a ``band`` x ``band`` grid of paths
    over its possible entries (:func:`band_paths`). Subsets are groups of views: subset :math:`m` of :math:`n` holds
    views :math:`m, m + n, \dots`.

    With CuPy on a CUDA GPU the paths are followed by CUDA kernels (:mod:`._pct_cuda`), one thread per path; otherwise
    by the same walk in PyTorch, much slower (it warns on a CUDA device). :class:`~pytomography.algorithms.SART` runs
    each subset in two walks of its paths, one forward (the WEPLs and lengths), one back (the residuals and the
    normalisation); :class:`~pytomography.algorithms.FilteredBackProjection` bins the paths' WEPLs into a parallel-beam
    sinogram by a straight line each (its options: ``n_angles``, angular bins over 360 degrees, default 360;
    ``r_spacing``, the radial bin in mm, default the voxel size in x; ``line``, ``'points'`` (entry to exit point,
    default) or ``'hull'`` (the hull chord)). Protons go to the GPU a view at a time, split to keep each call within
    ``gpu_budget``.

    Args:
        object_meta (ObjectMeta): The object grid, centred on the rotation axis and on ``proj_meta.z_center`` (mm).
        proj_meta (PCTProjMeta): The protons.
        path (str, optional): ``'spline'`` or ``'straight'``. Defaults to ``'spline'``.
        band (int, optional): Paths per side of each proton's band (a number; 1: the stored entry alone). Defaults to 1.
        hull (torch.Tensor | np.ndarray | None, optional): The object's outline on the object grid, [Lx, Ly, Lz], True
            inside (:func:`hull_from_image` of a first reconstruction). Defaults to None: the whole grid.
        segments (int, optional): Pieces of the spline (a number). Defaults to 24.
        gpu_budget (float | None, optional): Device bytes a projector call's paths may take. Defaults to None: 1.5
            GB, at most a quarter of the free GPU memory.
    """
    #: Memory estimates: the projector runs CUDA kernels.
    memory_cuda_projector = True

    def __init__(
        self,
        object_meta: ObjectMeta,
        proj_meta: PCTProjMeta,
        path: str = 'spline',
        band: int = 1,
        hull: torch.Tensor | np.ndarray | None = None,
        segments: int = 24,
        gpu_budget: float | None = None,
    ) -> None:
        super().__init__(object_meta, proj_meta)
        if path not in PATHS:
            raise ValueError(f'path must be one of {PATHS}, got {path!r}')
        self.path, self.band = path, int(band)
        self.segments = int(segments) if path == 'spline' else 1
        self.hull = None if hull is None else torch.as_tensor(hull).bool().cpu()
        if self.hull is not None and tuple(self.hull.shape) != tuple(object_meta.shape):
            raise ValueError(f'hull must have the object grid\'s shape {tuple(object_meta.shape)}, got '
                             f'{tuple(self.hull.shape)}')
        self.gpu_budget = gpu_budget
        self.subset_indices_array = None
        self._distance = None
        self._lo = tuple(float(np.float32(-s / 2.0 * d)) for s, d in zip(object_meta.shape, object_meta.dr))
        self._cuda = _pct_cuda.available(pytomography.device)
        if torch.device(pytomography.device).type == 'cuda' and not self._cuda:
            warnings.warn('CuPy is not installed: proton CT paths are followed in PyTorch on the GPU, far more slowly '
                          'than by its CUDA kernels (pip install cupy-cuda12x, or cupy-cuda13x)', stacklevel=2)

    # --- geometry, paths and subsets
    def _distance_map(self):
        """The hull's distance map (mm from each voxel to the nearest hull voxel, 0 inside), x fastest, on the device;
        None for the whole grid."""
        if self.hull is None:
            return None
        if self._distance is None:
            from scipy.ndimage import distance_transform_edt
            distance = distance_transform_edt(~self.hull.numpy(), sampling=tuple(self.object_meta.dr))
            self._distance = torch.as_tensor(distance, dtype=torch.float32,
                                             device=pytomography.device).permute(2, 1, 0).contiguous()
        return self._distance

    def _project(self, image_x, paths, residual=None, K=1, length=None, acc=None):
        args = (image_x, paths, self._distance_map(), self.segments, self.object_meta.shape, self.object_meta.dr,
                self._lo, residual, K, length, acc)
        return _pct_cuda.project(*args) if self._cuda else _project_torch(*args)

    def hull_points(self, paths: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Where each path's entry line first reaches the hull and its exit line, followed back, last leaves it.

        Args:
            paths (dict): ``entry_points``, ``entry_directions``, ``exit_points``, ``exit_directions`` ([n, 3], mm and
                unit vectors, object frame).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The two crossings ([n, 3], mm) and whether both exist.
        """
        shape, dr, dist = self.object_meta.shape, self.object_meta.dr, self._distance_map()
        if self._cuda:
            return _pct_cuda.hull_points(paths, dist, shape, dr, self._lo)
        lo = torch.tensor(self._lo, dtype=torch.float32, device=paths['entry_points'].device)
        a, b = paths['entry_points'] - lo, paths['exit_points'] - lo
        s0, hit0 = _first_reach(a, paths['entry_directions'], dist, shape, dr)
        s1, hit1 = _first_reach(b, -paths['exit_directions'], dist, shape, dr)
        return (a + s0[:, None] * paths['entry_directions'] + lo, b - s1[:, None] * paths['exit_directions'] + lo,
                hit0 & hit1)

    def set_n_subsets(self, n_subsets: int) -> None:
        """Partition the protons into ``n_subsets`` subsets by view.

        Subset :math:`m` holds views :math:`m, m + n, \\dots`.

        Args:
            n_subsets (int): Number of subsets, from 1 to the number of views.
        """
        self._check_n_subsets(n_subsets, self.proj_meta.N_angles, 'views')
        self.subset_indices_array = [torch.arange(m, self.proj_meta.N_angles, n_subsets) for m in range(n_subsets)]

    def _views(self, subset_idx):
        return range(self.proj_meta.N_angles) if subset_idx is None else self.subset_indices_array[subset_idx].tolist()

    def _chunks(self, subset_idx, bytes_per_path: float = _BYTES_PER_PATH):
        """(view, first proton, end) of each projector call: the subset's views in order, each split to keep its
        paths within the GPU budget."""
        per = max(1, int(gpu_budget(self.gpu_budget) // (bytes_per_path * self.band ** 2)))
        for v in self._views(subset_idx):
            n = int(self.proj_meta.view_sizes[v])
            for start in range(0, n, per):
                yield v, start, min(n, start + per)

    def _paths(self, view, start, stop):
        protons = self.proj_meta.load(view, pytomography.device, start, stop)
        return band_paths(protons, self.band, self.proj_meta.beam_divergence)

    def get_projection_subset(self, projections: torch.Tensor, subset_idx: int | None) -> torch.Tensor:
        """The WEPLs of the protons of subset ``subset_idx`` (all of them for None), view by view.

        Args:
            projections (torch.Tensor): WEPL of every proton, [N], in mm.
            subset_idx (int | None): Subset index, or None.

        Returns:
            torch.Tensor: The subset's WEPLs, in mm (on ``pytomography.device`` for a subset).
        """
        if subset_idx is None:
            return projections
        o = self.proj_meta.view_offsets
        return torch.cat([projections[o[v]:o[v + 1]].to(pytomography.device, non_blocking=True)
                          for v in self._views(subset_idx)])

    def get_weighting_subset(self, subset_idx: int | None) -> float:
        """The share of the protons in subset ``subset_idx``; 1 for None.

        Args:
            subset_idx (int | None): Subset index, or None.

        Returns:
            float: The share (no units).
        """
        if subset_idx is None:
            return 1.0
        return float(sum(self.proj_meta.view_sizes[v] for v in self._views(subset_idx))) / self.proj_meta.N_protons

    # --- projections
    def _xfast(self, object):
        return object.to(pytomography.device, torch.float32).permute(2, 1, 0).contiguous().view(-1)

    def _from_xfast(self, image_x):
        nx, ny, nz = self.object_meta.shape
        return image_x.view(nz, ny, nx).permute(2, 1, 0).contiguous()

    def forward(self, object: torch.Tensor, subset_idx: int | None = None) -> torch.Tensor:
        r"""Forward projection :math:`H_m f`: the WEPL of each proton of subset :math:`m` through the RSP map ``object``
        (with a band, the weighted mean over its paths).

        Args:
            object (torch.Tensor): RSP on the object grid, [Lx, Ly, Lz] (no units).
            subset_idx (int | None, optional): Subset index. Defaults to None: every proton.

        Returns:
            torch.Tensor: WEPL of each proton, [n], float32, in mm, on ``pytomography.device``.
        """
        image_x = self._xfast(object)
        out = [torch.zeros(0, device=pytomography.device)]
        for v, start, stop in self._chunks(subset_idx):
            paths = self._paths(v, start, stop)
            fwd, _ = self._project(image_x, paths)
            out.append((paths['weights'] * fwd).view(stop - start, -1).sum(1))
            del paths, fwd                   # before the next call's paths are made
        return torch.cat(out)

    def _back(self, proj: torch.Tensor, subset_idx: int | None):
        """:math:`H_m^T g` and :math:`H_m^T 1`, in one walk of the subset's paths."""
        acc = torch.zeros((math.prod(self.object_meta.shape), 2), dtype=torch.float32, device=pytomography.device)
        offset = 0
        for v, start, stop in self._chunks(subset_idx):
            paths = self._paths(v, start, stop)
            y = proj[offset:offset + stop - start].to(pytomography.device, torch.float32, non_blocking=True)
            offset += stop - start
            self._project(None, paths, residual=y, K=self.band ** 2, acc=acc)
            del paths, y
        return self._from_xfast(acc[:, 0].contiguous()), self._from_xfast(acc[:, 1].contiguous())

    def backward(self, proj: torch.Tensor, subset_idx: int | None = None) -> torch.Tensor:
        r"""Back projection :math:`H_m^T g`: each proton's value spread along its paths by their lengths and weights
        (the transpose of :meth:`forward`).

        Args:
            proj (torch.Tensor): A value per proton of subset :math:`m`, [n] (in mm for WEPLs).
            subset_idx (int | None, optional): Subset index. Defaults to None: every proton.

        Returns:
            torch.Tensor: [Lx, Ly, Lz], float32, on ``pytomography.device`` (in mm times the units of ``proj``).
        """
        return self._back(proj, subset_idx)[0]

    def compute_normalization_factor(self, subset_idx: int | None = None) -> torch.Tensor:
        r""":math:`H_m^T 1`, the summed path length of the subset's protons in every voxel.

        Args:
            subset_idx (int | None, optional): Subset index. Defaults to None: every proton.

        Returns:
            torch.Tensor: [Lx, Ly, Lz], in mm.
        """
        n = sum(int(self.proj_meta.view_sizes[v]) for v in self._views(subset_idx))
        return self._back(torch.ones(n, device=pytomography.device), subset_idx)[1]

    def _sart_terms(self, object: torch.Tensor, projections: torch.Tensor, subset_idx: int | None):
        r"""SART's update terms for subset :math:`m`, :math:`H_m^T ((g_m - H_m f) / H_m 1)` and :math:`H_m^T 1`, in two
        walks of each projector call's paths: forward (each path's WEPL and length), then back with each proton's
        residual. :class:`~pytomography.likelihoods.SARTWeightedNegativeMSELikelihood` calls it in place of three
        projections and a stored :math:`H_m^T 1` per subset."""
        image_x = self._xfast(object)
        acc = torch.zeros((image_x.numel(), 2), dtype=torch.float32, device=pytomography.device)
        K, offset = self.band ** 2, 0
        for v, start, stop in self._chunks(subset_idx):
            paths = self._paths(v, start, stop)
            g = projections[offset:offset + stop - start].to(pytomography.device, non_blocking=True)
            offset += stop - start
            fwd, length = self._project(image_x, paths)
            w = paths['weights']
            F, L = (w * fwd).view(-1, K).sum(1), (w * length).view(-1, K).sum(1)
            r = torch.where(L > 0, (g - F) / L.clamp(min=1e-6), torch.zeros_like(L))
            self._project(None, paths, residual=r, K=K, length=length, acc=acc)
            del paths, fwd, length, w, F, L, r, g
        return self._from_xfast(acc[:, 0].contiguous()), self._from_xfast(acc[:, 1].contiguous())

    # --- filtered back projection
    def _sinogram_geometry(self, n_angles: int, r_spacing: float | None) -> tuple[float, int, int]:
        """(radial bin in mm, radial bins, sinogram bins) of the binned sinogram."""
        (nx, ny, nz), (dx, dy, _) = self.object_meta.shape, self.object_meta.dr
        r_spacing = float(r_spacing) if r_spacing is not None else dx
        n_r = 2 * int(math.ceil(0.5 * math.hypot(nx * dx, ny * dy) / r_spacing)) + 1
        return r_spacing, n_r, n_angles * n_r * nz

    def _bin_into(self, total, count, p0, p1, wepl, weights, n_angles, r_spacing, n_r):
        """Add the paths' WEPLs times their weights, and their weights, to the bins of their lines ``p0`` -> ``p1``:
        angle :math:`\\phi` of the line in x and y, signed distance :math:`r = -x\\sin\\phi + y\\cos\\phi` from the
        rotation axis, slice where it passes closest to the axis."""
        nz, dz = self.object_meta.shape[2], self.object_meta.dr[2]
        two_pi = 2 * math.pi
        for start in range(0, wepl.shape[0], _BIN_CHUNK):
            s = slice(start, start + _BIN_CHUNK)
            a, d = p0[s], p1[s] - p0[s]
            phi = torch.remainder(torch.atan2(d[:, 1], d[:, 0]), two_pi)
            r = -a[:, 0] * torch.sin(phi) + a[:, 1] * torch.cos(phi)
            closest = -(a[:, 0] * d[:, 0] + a[:, 1] * d[:, 1]) / (d[:, 0] ** 2 + d[:, 1] ** 2).clamp(min=1e-12)
            z = a[:, 2] + closest * d[:, 2]
            ia = torch.remainder(torch.round(phi / (two_pi / n_angles)).long(), n_angles)
            ir = torch.round(r / r_spacing + (n_r - 1) / 2).long()
            iz = torch.round(z / dz + (nz - 1) / 2).long()
            ok = (ir >= 0) & (ir < n_r) & (iz >= 0) & (iz < nz)
            flat = ((ia * n_r + ir) * nz + iz)[ok]
            w = weights[s][ok].double()
            total += torch.bincount(flat, weights=wepl[s][ok].double() * w, minlength=total.numel())
            count += torch.bincount(flat, weights=w, minlength=total.numel())

    def _fbp(self, projections: torch.Tensor, filter, n_angles: int = 360, r_spacing: float | None = None,
             line: str = 'points') -> torch.Tensor:
        r"""Filtered back projection, called by :class:`~pytomography.algorithms.FilteredBackProjection`: every path
        (each proton's band) binned with its weight into a parallel-beam sinogram by a straight line, through its
        entry and exit points (``line='points'``) or along its hull chord (``'hull'``: closer to the path inside the
        object, as the proton flies straight in the air outside it); each bin's mean WEPL ramp filtered along
        :math:`r` (times the window ``filter``) and back projected slice by slice with the weight :math:`\pi/N` of
        :math:`N` angles over 360 degrees. Angular bin :math:`j` is centred on :math:`j \cdot 360/N` degrees, where
        step-and-shoot views sit.

        Args:
            projections (torch.Tensor): WEPL of every proton, [N], in mm.
            filter (FBPFilter): Window on top of the ramp filter.
            n_angles (int, optional): Angular bins over 360 degrees (a number). Defaults to 360.
            r_spacing (float | None, optional): Radial bin size, in mm. Defaults to None: the voxel size in x.
            line (str, optional): ``'points'`` or ``'hull'``. Defaults to ``'points'``.

        Returns:
            torch.Tensor: RSP, [Lx, Ly, Lz] (no units), on ``pytomography.device``.
        """
        if line not in LINES:
            raise ValueError(f'line must be one of {LINES}, got {line!r}')
        window = get_fbp_filter(filter)
        (nx, ny, nz), (dx, dy, _) = self.object_meta.shape, self.object_meta.dr
        r_spacing, n_r, n_bins = self._sinogram_geometry(n_angles, r_spacing)
        device = pytomography.device
        total = torch.zeros(n_bins, dtype=torch.float64, device=device)
        count = torch.zeros(n_bins, dtype=torch.float64, device=device)
        K, offset = self.band ** 2, 0
        for v, start, stop in self._chunks(None, _BIN_BYTES_PER_PATH):
            paths = self._paths(v, start, stop)
            g = projections[offset:offset + stop - start].to(device, non_blocking=True).repeat_interleave(K)
            offset += stop - start
            p0, p1 = paths['entry_points'], paths['exit_points']
            if line == 'hull':
                h0, h1, hit = self.hull_points(paths)
                p0, p1 = torch.where(hit[:, None], h0, p0), torch.where(hit[:, None], h1, p1)
                del h0, h1, hit
            self._bin_into(total, count, p0, p1, g, paths['weights'], n_angles, r_spacing, n_r)
            del paths, g, p0, p1
        total /= torch.where(count > 0, count, 1.0)      # the mean WEPL, 0 where there are no protons
        q = ramp_filter(total.float().view(n_angles, n_r, nz), r_spacing, window, dim=1)
        del total, count
        x = (torch.arange(nx, device=device) - (nx - 1) / 2) * dx
        y = (torch.arange(ny, device=device) - (ny - 1) / 2) * dy
        image = torch.zeros((nx, ny, nz), dtype=torch.float32, device=device)
        for j in range(n_angles):
            phi = j * 2 * math.pi / n_angles
            u = (-x[:, None] * math.sin(phi) + y[None, :] * math.cos(phi)) / r_spacing + (n_r - 1) / 2
            i0 = torch.clamp(torch.floor(u), 0, n_r - 2).long()
            w = torch.clamp(u - i0, 0, 1)[:, :, None]
            image += (1 - w) * q[j][i0] + w * q[j][i0 + 1]
        return image * (math.pi / n_angles)

    # --- memory
    def _call_bytes(self, bytes_per_path: float = _BYTES_PER_PATH) -> float:
        """Device bytes of the largest projector call."""
        rows = self.band ** 2 * float(max(self.proj_meta.view_sizes, default=0))
        return min(rows * bytes_per_path, gpu_budget(self.gpu_budget))

    def _memory_parts(self, n_subsets: int, N_splits: int | None) -> list[MemoryPart]:
        image = 4 * math.prod(self.object_meta.shape)
        device = pytomography.device
        return [MemoryPart('protons (entry and exit tables, entry spreads)', ram_bytes=self.proj_meta.nbytes()),
                MemoryPart('image x fastest, back projection and normalisation (paired and apart), hull distance map',
                           scope='subset', **on_device(device, 6 * image)),
                MemoryPart('one projector call: the paths of its bands, their WEPLs and lengths', scope='chunk',
                           **on_device(device, self._call_bytes()))]

    def _fbp_memory_parts(self, projections: torch.Tensor, filter, n_angles: int = 360,
                          r_spacing: float | None = None, line: str = 'points') -> list[MemoryPart]:
        (nx, ny, nz) = self.object_meta.shape
        _, n_r, n_bins = self._sinogram_geometry(n_angles, r_spacing)
        n_pad = int(2 ** math.ceil(math.log2(2 * n_r)))
        voxels = nx * ny * nz
        binning = 16 * n_bins + self._call_bytes(_BIN_BYTES_PER_PATH) + 4 * voxels * (line == 'hull')
        filtering = 4 * voxels + 4 * n_bins + n_angles * nz * (2 * 8 * (n_pad // 2 + 1) + 4 * n_pad) + 16 * voxels
        device = pytomography.device
        return [MemoryPart('protons (entry and exit tables, entry spreads) and their WEPLs',
                           ram_bytes=self.proj_meta.nbytes() + 4 * self.proj_meta.N_protons),
                MemoryPart('binning: sums and counts, one call\'s paths' if binning >= filtering else
                           'filtering: sinogram, FFTs, image and back projection', scope='subset',
                           **on_device(device, max(binning, filtering)))]
