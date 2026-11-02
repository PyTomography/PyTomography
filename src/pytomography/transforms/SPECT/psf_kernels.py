"""A SPECT PSF from arbitrary 2D kernels, one per distance from the collimator, e.g. made from angular response
functions (ARFs) by spectarf."""
from __future__ import annotations
from collections.abc import Callable
import numpy as np
import torch
import pytomography
from pytomography.transforms import Transform
from pytomography.utils import get_distance
from pytomography.utils.memory import BLOCK_CAP_BYTES, block_size

#: Kernels at the exact plane distances while these number at most this many times the planes of a view (a few orbit
#: radii); beyond that, kernels on a grid of distances (see :class:`SPECTKernelPSFTransform`).
EXACT_PLANES_PER_PLANE_COUNT = 4
#: The grid's spacing: consecutive kernels' distances from the source to the kernels' centring plane differ by this
#: fraction. A collimator's kernel scales with that distance, so linear interpolation errs about equally at every
#: distance: below 0.04 % of the kernel's peak at 2 %, midway between nodes (Ac-225 440 keV on a high energy
#: collimator and Lu-177 208 keV on a low energy one, 0-600 mm), with a sixth of a 1 mm grid's kernels.
DEFAULT_DISTANCE_RATIO = 0.02
#: That centring plane's depth behind the collimator face, when the kernel function does not give its own
#: (``centring``): about a collimator's hole length.
DEFAULT_CENTRING_MM = 50.0


def _fast_len(n: int) -> int:
    """The smallest integer at least ``n`` whose prime factors are 2, 3 and 5 (a fast FFT length)."""
    n = max(int(n), 1)
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1



def _attribute(kernels, name: str, default: float) -> float:
    """``kernels.<name>`` (mm), or ``kernels.<name>_mm`` as spectarf named it before PyTomography 4, or ``default``."""
    return getattr(kernels, name, getattr(kernels, f"{name}_mm", default))


class SPECTKernelPSFTransform(Transform):
    r"""Object-to-object transform that models the SPECT collimator-detector response with an arbitrary 2D kernel for
    each distance from the collimator.

    For each view, every plane of the rotated object parallel to the detector (each x) is convolved with the kernel of
    its distance from the collimator, so that the sum along x is the blurred projection. The kernels may have any
    shape: no symmetry, separability or Gaussian form is assumed, so :meth:`backward` is the correlation with the same
    kernels, the true transpose of :meth:`forward`. Both are computed by FFT with zero padding to at least
    :math:`n + k - 1`, so nothing wraps.

    The kernels come from ``kernels``, a function of the plane distances: spectarf's kernels from angular response
    functions (``spectarf.pytomography``), a measured PSF, or any model. It is called once, in :meth:`configure`.

    * Distances: the distance of a plane from the collimator face (:func:`~pytomography.utils.get_distance` of the
      view's radius) plus ``front_cover``, in mm.
    * Axes: a kernel's first axis runs along r (the rotated object's second axis, the projection's transaxial axis) and
      its second along z; its centre, element ``(k_r // 2, k_z // 2)``, is the plane's own position.
    * Scale: the kernels as given (spectarf's are counts per emitted photon, so a reconstruction is in emitted photons);
      ``normalize=True`` scales each kernel to sum to one, the scale of :class:`SPECTPSFTransform`'s Gaussian.

    Non-circular orbits put every view's planes at different distances. While the distances of all the views number at
    most :data:`EXACT_PLANES_PER_PLANE_COUNT` times the planes of one view (a few radii), each plane gets its exact
    kernel. Beyond that, kernels are made on a grid of distances and each plane's kernel is interpolated linearly
    between its two grid kernels (on their spectra, which the grid keeps). The grid is geometric: the distances from the
    source to the kernels' centring plane (the plane distance plus ``centring``) grow by ``distance_ratio`` from one
    kernel to the next. ``distance_step`` gives a uniform grid instead.

    Args:
        kernels (Callable): The kernel function, ``kernels(distances, pixel_size, shape)``: the plane distances (a 1D
            array, in mm), the pixel size ``(r, z)`` in mm and the kernel shape ``(k_r, k_z)``; it returns the kernels,
            ``[n_distances, k_r, k_z]`` (a tensor or array). It may carry the attributes ``front_cover`` and
            ``centring``, used when those arguments are not given.
        kernel_shape (tuple[int, int] | None, optional): The kernels' shape ``(k_r, k_z)``, odd. Defaults to None:
            :math:`(2 L_y - 1, 2 L_z - 1)` of the padded object, which couples every pair of voxels in a plane.
        normalize (bool, optional): Scale each kernel to sum to one. Defaults to False: the kernels' own scale.
        clip_negative (bool, optional): Set negative kernel values to zero (a kernel corrected for scatter, e.g. with
            the triple energy window estimate, can go negative in its tails). The clipped share of each kernel (the sum
            of its negative values over the sum of its absolute values) is kept in ``clipped_fraction``. Defaults to
            True.
        front_cover (float | None, optional): Added to every plane distance, in mm: the kernels' distances are
            measured from a plane this far behind the collimator face. Defaults to None: ``kernels.front_cover``, else
            0.
        centring (float | None, optional): The geometric grid's offset (above), in mm. Defaults to None:
            ``kernels.centring``, else :data:`DEFAULT_CENTRING_MM`.
        distance_step (float | None, optional): Kernels on a uniform grid of distances this far apart (mm). Defaults
            to None: exact distances or the geometric grid, as above.
        distance_ratio (float, optional): The geometric grid's spacing. Defaults to :data:`DEFAULT_DISTANCE_RATIO`.

    Attributes:
        distances (np.ndarray): After :meth:`configure`, the sorted distances the kernels were made at: every plane's
            of every view, or the grid's.
        kernels (torch.Tensor): The kernels, ``[n_distances, k_r, k_z]``, on the device.
        clipped_fraction (np.ndarray): The clipped share of each kernel.

    Example:
        A kernel function from spectarf's ARFs, then the transform in a system matrix (see the ARF tutorial)::

            kernels = spectarf.pytomography.kernels(arfs)
            psf = SPECTKernelPSFTransform(kernels)
            system_matrix = SPECTSystemMatrix([SPECTAttenuationTransform(mu), psf], [], object_meta, proj_meta)
    """
    def __init__(
        self,
        kernels: Callable,
        kernel_shape: tuple[int, int] | None = None,
        normalize: bool = False,
        clip_negative: bool = True,
        front_cover: float | None = None,
        centring: float | None = None,
        distance_step: float | None = None,
        distance_ratio: float = DEFAULT_DISTANCE_RATIO,
    ) -> None:
        super().__init__()
        if not callable(kernels):
            raise TypeError('kernels must be a function of (distances, pixel_size, shape)')
        if distance_ratio <= 0:
            raise ValueError('distance_ratio must be positive')
        self.kernel_function = kernels
        self.kernel_shape = kernel_shape
        self.normalize = bool(normalize)
        self.clip_negative = bool(clip_negative)
        # the kernel function's own values; spectarf releases before PyTomography 4 name them with an _mm suffix
        self.front_cover = float(front_cover if front_cover is not None else _attribute(kernels, 'front_cover', 0.0))
        self.centring = float(centring if centring is not None
                              else _attribute(kernels, 'centring', DEFAULT_CENTRING_MM))
        self.distance_step = distance_step
        self.distance_ratio = float(distance_ratio)

    # -- configuration --------------------------------------------------------------------------------------------
    def _make_kernels(self, distances: np.ndarray, pixel_size: tuple[float, float], shape: tuple[int, int]) -> None:
        """Calls the kernel function, checks its shape, records the clipped share, clips and normalizes, in place. The
        kernels keep the function's floating point type (``pytomography.dtype`` for other types)."""
        k = torch.as_tensor(self.kernel_function(distances, pixel_size, shape)).to(self.device)
        if not k.is_floating_point():
            k = k.to(pytomography.dtype)
        if tuple(k.shape) != (len(distances), *shape):
            raise ValueError(f'the kernel function returned shape {tuple(k.shape)}, expected {(len(distances), *shape)}')
        self.clipped_fraction = np.empty(k.shape[0])
        for i0 in range(0, k.shape[0], 32):        # a block of kernels at a time: one copy of them
            b = k[i0:i0 + 32]
            total = b.abs().sum(dim=(1, 2), dtype=torch.float64)
            negative = -b.clamp(max=0).sum(dim=(1, 2), dtype=torch.float64)
            self.clipped_fraction[i0:i0 + 32] = (negative / total.clamp(min=1e-300)).cpu().numpy()
            if self.clip_negative:
                b.clamp_(min=0)
            if self.normalize:
                b.div_(b.sum(dim=(1, 2), keepdim=True).clamp(min=torch.finfo(b.dtype).tiny))
        self.kernels = k

    def _grid_distances(self, lo: float, hi: float) -> np.ndarray:
        """The grid's distances, from below ``lo`` to above ``hi``: uniform with ``distance_step``, else geometric."""
        if self.distance_step is not None:
            step = float(self.distance_step)
            start = step * np.floor(lo / step)
            return start + step * np.arange(max(1, int(np.ceil((hi - start) / step))) + 1)
        off = self.centring
        n = max(1, int(np.ceil(np.log((hi + off) / (lo + off)) / np.log1p(self.distance_ratio))))
        return (lo + off) * (1.0 + self.distance_ratio) ** np.arange(n + 1) - off

    def configure(self, object_meta, proj_meta) -> None:
        r"""Makes the kernels for the system matrix's grid and the radius of every view; the system matrix calls it.

        Args:
            object_meta (SPECTObjectMeta): The object grid.
            proj_meta (SPECTProjMeta): The projection space, with the radius of every view.
        """
        super().configure(object_meta, proj_meta)
        n_x, n_y, n_z = (int(v) for v in object_meta.padded_shape)
        pixel_size = (float(object_meta.dr[1]), float(object_meta.dr[2]))
        shape = tuple(int(s) for s in (self.kernel_shape or (2 * n_y - 1, 2 * n_z - 1)))
        if shape[0] % 2 == 0 or shape[1] % 2 == 0:
            raise ValueError(f'kernel_shape must be odd, got {shape}')
        radii = np.asarray(proj_meta.radii, dtype=np.float64).reshape(-1)
        per_radius = {float(r): get_distance(n_x, float(r), float(object_meta.dx)) + self.front_cover
                      for r in np.unique(radii)}
        distances = np.unique(np.round(np.concatenate(list(per_radius.values())), 6))
        self._grid = self.distance_step is not None or distances.size > EXACT_PLANES_PER_PLANE_COUNT * n_x
        if self._grid:
            distances = self._grid_distances(float(distances.min()), float(distances.max()))
            self._planes = {}                        # each plane between grid kernels i and i + 1, a fraction t along
            for r, d in per_radius.items():
                i = np.clip(np.searchsorted(distances, d, side='right') - 1, 0, distances.size - 2)
                t = np.clip((d - distances[i]) / (distances[i + 1] - distances[i]), 0.0, 1.0)
                self._planes[r] = (torch.as_tensor(i, device=self.device),
                                   torch.as_tensor(t, device=self.device, dtype=pytomography.dtype))
        else:
            self._planes = {r: torch.as_tensor(np.searchsorted(distances, np.round(d, 6)), device=self.device)
                            for r, d in per_radius.items()}
        self.distances = distances
        self._make_kernels(distances, pixel_size, shape)
        self._radii = radii.tolist()
        self._spectra = {}

    # -- application ----------------------------------------------------------------------------------------------
    def _fft_shape(self, n_y: int, n_z: int) -> tuple[int, int]:
        k_r, k_z = self.kernels.shape[1:]
        return _fast_len(n_y + k_r - 1), _fast_len(n_z + k_z - 1)

    def _kernel_spectra(self, k: torch.Tensor, f_y: int, f_z: int, dtype: torch.dtype) -> torch.Tensor:
        """The spectra of kernels ``k`` in ``dtype``, centred on element 0, made a block of kernels at a time."""
        k_r, k_z = k.shape[1:]
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        out = torch.empty((k.shape[0], f_y, f_z // 2 + 1), device=k.device, dtype=complex_dtype)
        per_kernel = 3 * f_y * f_z * torch.finfo(dtype).bits / 8    # the padded kernel, its rolled copy, its spectrum
        step = block_size(per_kernel, default=max(1, int(BLOCK_CAP_BYTES // per_kernel)))
        for i0 in range(0, k.shape[0], step):
            b = k[i0:i0 + step]
            padded = torch.zeros((b.shape[0], f_y, f_z), device=k.device, dtype=dtype)
            padded[:, :k_r, :k_z] = b
            out[i0:i0 + step] = torch.fft.rfft2(torch.roll(padded, shifts=(-(k_r // 2), -(k_z // 2)), dims=(1, 2)))
            del padded
        return out

    def _view_spectra(self, radius: float, n_y: int, n_z: int, dtype: torch.dtype) -> tuple[torch.Tensor, int, int]:
        """The kernel spectra of a view's planes: kept per radius (exact distances), or interpolated from the grid's."""
        f_y, f_z = self._fft_shape(n_y, n_z)
        if self._grid:
            grid = self._spectra.get(('grid', f_y, f_z, dtype))
            if grid is None:
                grid = self._spectra[('grid', f_y, f_z, dtype)] = self._kernel_spectra(self.kernels, f_y, f_z, dtype)
            i, t = self._planes[radius]
            t = t.to(grid.real.dtype)[:, None, None]
            spectra = grid[i].mul_(1 - t)            # grid[i] is a copy (indexing by a tensor)
            spectra.add_(grid[i + 1].mul_(t))
            return spectra, f_y, f_z
        key = (radius, f_y, f_z, dtype)
        spectra = self._spectra.get(key)
        if spectra is None:
            spectra = self._spectra[key] = self._kernel_spectra(self.kernels[self._planes[radius]], f_y, f_z, dtype)
        return spectra, f_y, f_z

    def _apply(self, object: torch.Tensor, ang_idx: int, transpose: bool) -> torch.Tensor:
        n_y, n_z = object.shape[-2:]
        spectra, f_y, f_z = self._view_spectra(float(self._radii[int(ang_idx)]), n_y, n_z, object.dtype)
        if transpose:
            spectra = spectra.conj()
        planes = torch.fft.rfft2(object, s=(f_y, f_z))
        planes.mul_(spectra)
        return torch.fft.irfft2(planes, s=(f_y, f_z))[..., :n_y, :n_z]

    @torch.no_grad()
    def forward(self, object: torch.Tensor, ang_idx: int) -> torch.Tensor:
        r"""Convolves every plane of the rotated object with the kernel of its distance from the collimator.

        Args:
            object (torch.Tensor): The object rotated for the view, [Lx, Ly, Lz] (padded), projected along x.
            ang_idx (int): Index of the view in ``proj_meta``: its radius sets the plane distances.

        Returns:
            torch.Tensor: The blurred object, [Lx, Ly, Lz]; its sum along x is the blurred projection.
        """
        return self._apply(object, ang_idx, transpose=False)

    @torch.no_grad()
    def backward(self, object: torch.Tensor, ang_idx: int) -> torch.Tensor:
        r"""Correlates every plane with the kernel of its distance: the transpose of :meth:`forward`, for any kernel.

        Args:
            object (torch.Tensor): A back projected view, [Lx, Ly, Lz] (padded), in the view's rotated frame.
            ang_idx (int): Index of the view in ``proj_meta``.

        Returns:
            torch.Tensor: The result, [Lx, Ly, Lz].
        """
        return self._apply(object, ang_idx, transpose=True)

    def memory(self, object_shape) -> tuple[float, float, float]:
        """What the transform keeps and the most one call holds, in bytes, for padded objects of ``object_shape`` in
        ``pytomography.dtype``: (RAM kept, GPU kept, call's peak on its device). The system matrix's memory estimate
        asks for it. Kept: the kernels, and their spectra (the grid's, or one set per radius, made at the first view of
        each). A call: three spectra of the padded planes and one real copy of them (the object's spectrum, the copy
        cuFFT's inverse transform takes of it and the result; on a grid, the view's interpolated kernel spectra; with
        exact kernels, the FFT's workspace). Measured on an RTX 5090 (14 Oct 2026, a 128^3 object padded to
        182 x 182 x 128, kernels 363 x 255): 0.840 GB on a grid of 139 kernels (this estimate 0.839 GB) and 0.781 GB
        with one radius (0.875 GB), the input object included.

        Args:
            object_shape (Sequence[int]): The padded object's shape [Lx, Ly, Lz].

        Returns:
            tuple[float, float, float]: RAM kept, GPU kept, and the peak of one call, in bytes.
        """
        n_x, n_y, n_z = (int(s) for s in object_shape)
        real = torch.finfo(pytomography.dtype).bits / 8
        k_r, k_z = self.kernels.shape[1:] if hasattr(self, 'kernels') else (self.kernel_shape or (2 * n_y - 1, 2 * n_z - 1))
        f_y, f_z = _fast_len(n_y + k_r - 1), _fast_len(n_z + k_z - 1)
        spectrum = 2 * real * f_y * (f_z // 2 + 1)                  # of one plane or kernel, complex
        n_d = len(self.distances) if hasattr(self, 'distances') else n_x
        kept = (torch.finfo(self.kernels.dtype).bits / 8 if hasattr(self, 'kernels') else real) * n_d * k_r * k_z
        call = n_x * (3 * spectrum + real * f_y * f_z)
        if getattr(self, '_grid', False):
            kept += n_d * spectrum
        else:
            kept += n_x * spectrum * max(1, len(getattr(self, '_planes', {None: 0})))
        return (0.0, kept, call) if torch.device(self.device).type == 'cuda' else (kept, 0.0, call)
