"""An image read from a file (or made from an array), with its place in the patient, and its resampling onto a
reconstruction grid."""
from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import torch

import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.metadata import FRAMES

#: Along an axis where a target voxel is more than this many source voxels long, ``Image.on`` (mode 'auto') takes
#: the mean over the target voxel instead of interpolating at its centre.
AVERAGE_ABOVE = 1.5


def _numpy(x) -> np.ndarray:
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class Image():
    r"""A 3D image with its own geometry: what :func:`~pytomography.io.load_image` returns, and what the writers
    accept in place of a tensor and its ``object_meta``.

    Args:
        data (np.ndarray | torch.Tensor): the image, axes (x, y, z) (a leading axis of length 1 is dropped). Kept as a
            numpy array on the CPU.
        affine (array-like): 4 x 4 matrix from voxel index (i, j, k) to the voxel's centre in LPS mm. Any orientation.
        units (str, optional): the units of the values, e.g. ``'HU'``, ``'1/cm'``, ``'counts'``, ``'label'``.
        modality (str, optional): ``'CT'``, ``'MR'``, ``'PT'``, ``'NM'``...
        meta (dict, optional): what the source says about itself (KVP, Manufacturer, the series description...).
        frame (str, optional): ``'patient'`` when the affine is in the study's patient coordinates, ``'scanner'`` when
            the file only places the image relative to the scanner's centre. Defaults to ``'patient'``.
        labels (dict, optional): for a label image (a mask or a segmentation), each label's name and its value, or
            the tuple of every value whose voxels belong to it (when labels overlap, each overlap has a value of its
            own).

    Attributes:
        centre (np.ndarray): the position of the image's centre in LPS mm.
        voxel_size (tuple): the distance between neighbouring voxels along i, j and k, in mm.
        object_meta (ObjectMeta): the image's own grid as an :class:`~pytomography.metadata.ObjectMeta` (for an image
            whose axes run along +x, +y and +z).
    """
    def __init__(self, data, affine, units: str | None = None, modality: str | None = None, meta: dict | None = None,
                 frame: str = 'patient', labels: dict | None = None) -> None:
        data = _numpy(data)
        while data.ndim > 3 and data.shape[0] == 1:
            data = data[0]
        if data.ndim != 3:
            raise ValueError(f'an Image is 3D (x, y, z), got shape {data.shape}')
        A = np.asarray(_numpy(affine), dtype=np.float64).reshape(4, 4)
        if not np.allclose(A[3], [0, 0, 0, 1]):
            raise ValueError(f'not an affine matrix: last row {A[3]}')
        if abs(np.linalg.det(A[:3, :3])) < 1e-12:
            raise ValueError('the affine is singular')
        if frame not in FRAMES:
            raise ValueError(f"frame must be one of {FRAMES}, not {frame!r}")
        self.data = data
        self.affine = A
        self.units = units
        self.modality = None if modality is None else str(modality).upper()
        self.meta = dict(meta or {})
        self.frame = frame
        self.labels = dict(labels) if labels else None

    @property
    def shape(self) -> tuple:
        return tuple(int(n) for n in self.data.shape)

    @property
    def voxel_size(self) -> tuple:
        return tuple(float(v) for v in np.linalg.norm(self.affine[:3, :3], axis=0))

    @property
    def centre(self) -> np.ndarray:
        return (self.affine @ np.array([*((np.asarray(self.shape) - 1) / 2), 1.0]))[:3]

    @property
    def object_meta(self) -> ObjectMeta:
        A = self.affine[:3, :3]
        d = np.diag(A)
        if not (np.allclose(A, np.diag(d), atol=1e-6 * np.abs(d).max()) and np.all(d > 0)):
            raise ValueError('this image is not on an LPS-aligned grid (its axes are not +x, +y and +z), so it has no '
                             'ObjectMeta; resample it onto one with Image.on')
        return ObjectMeta(dr=d, shape=self.shape, centre=self.centre, frame=self.frame, units=self.units,
                          modality=self.modality)

    def __repr__(self):
        return (f'Image(shape={self.shape}, voxel_size={tuple(round(v, 4) for v in self.voxel_size)}, '
                f'centre={tuple(round(float(c), 3) for c in self.centre)}, units={self.units!r}, '
                f'modality={self.modality!r}, frame={self.frame!r})')

    # ---------- onto a reconstruction grid ----------

    def on(self, object_meta: ObjectMeta, *, mode: str = 'auto', label: int | str | None = None,
           threshold: float | None = None, fill: float = 0.0) -> torch.Tensor:
        """The image resampled onto the grid of ``object_meta`` (any grid: a reconstruction's, a subsampled or a
        stitched one), as a float32 tensor on ``pytomography.device``.

        The image is taken to fill its voxels: inside a voxel's extent its value is the voxel's, and between voxel
        centres it is interpolated linearly; outside the image it is ``fill``.

        Args:
            object_meta (ObjectMeta): the target grid.
            mode (str): ``'auto'``: trilinear interpolation at the target voxel centres, or, along any axis where a
                target voxel is more than 1.5 source voxels long, the mean over the target voxel (so a 1 mm CT on a
                4.8 mm SPECT grid is averaged, not point sampled, #6). ``'nearest'``: the value of the nearest voxel.
            label (int | str | tuple, optional): for a label image, the label (its value, a tuple or list of values,
                or its name in ``labels``) whose coverage is returned: the fraction of each target voxel inside it (0 or 1 with ``mode='nearest'``).
            threshold (float, optional): return a bool tensor, True where the resampled value (or coverage) is at
                least ``threshold``.
            fill (float): the value outside the image. Defaults to 0.

        Returns:
            torch.Tensor: [Lx, Ly, Lz] on ``pytomography.device``, float32 (bool with ``threshold``).
        """
        if mode not in ('auto', 'nearest'):
            raise ValueError(f"mode must be 'auto' or 'nearest', not {mode!r}")
        if self.frame != object_meta.frame:
            warnings.warn(f"the image is in frame '{self.frame}' and the grid in frame '{object_meta.frame}': the "
                          'image is placed by coordinates that may not refer to the same origin')
        value = None
        if label is not None:
            if isinstance(label, str):
                if not self.labels or label not in self.labels:
                    raise KeyError(f"no label '{label}'; the image's labels are {sorted(self.labels or {})}")
                value = self.labels[label]
            else:
                value = label
            if isinstance(value, (list, tuple, np.ndarray)):
                value = tuple(np.asarray(value).ravel().tolist())
        M = np.linalg.inv(self.affine) @ object_meta.affine      # target index -> source index
        out = _resample(self.data, M, tuple(object_meta.shape), mode, value, float(fill))
        out = out.to(pytomography.device)
        if threshold is not None:
            return out >= threshold
        return out


# ---------- reorientation ----------

def reorient_to_lps(image: Image) -> Image:
    """The image with its axes permuted and flipped (never interpolated) so that i, j and k run as close as possible
    to +x, +y and +z (LPS). Any remaining obliquity stays in the affine."""
    A = image.affine
    cosines = np.abs(A[:3, :3] / np.linalg.norm(A[:3, :3], axis=0))        # [LPS axis, array axis]
    best = max(itertools.permutations(range(3)), key=lambda p: sum(cosines[p[a], a] for a in range(3)))
    source_axis = [best.index(t) for t in range(3)]                          # new axis t holds old axis source_axis[t]
    data = np.transpose(image.data, source_axis)
    A_new = A.copy()
    A_new[:, :3] = A[:, source_axis]
    flips = []
    for t in range(3):
        if A_new[t, t] < 0:
            A_new[:3, 3] += A_new[:3, t] * (data.shape[t] - 1)
            A_new[:3, t] *= -1
            flips.append(t)
    if flips:
        data = np.flip(data, axis=flips)
    if source_axis != [0, 1, 2] or flips:
        data = np.ascontiguousarray(data)
    return Image(data, A_new, units=image.units, modality=image.modality, meta=image.meta, frame=image.frame,
                 labels=image.labels)


# ---------- resampling ----------

def _chunk(bytes_per_slice: float) -> int:
    from pytomography.utils.memory import BLOCK_CAP_BYTES, block_size
    return block_size(bytes_per_slice, default=max(1, int(BLOCK_CAP_BYTES // max(bytes_per_slice, 1.0))))


def _in_label(data: np.ndarray, value) -> np.ndarray:
    """Where ``data`` holds the label ``value`` (one value, or a tuple of them), as float32 0 and 1."""
    if isinstance(value, tuple):
        return np.isin(data, value).astype(np.float32)
    return (data == value).astype(np.float32)


def _values(data: np.ndarray, value, fill: float) -> np.ndarray:
    """The source values to resample, as float32, minus ``fill`` (so that outside the image adds nothing): a label's
    indicator for a label."""
    if value is not None:
        return _in_label(data, value) - np.float32(fill)
    return np.ascontiguousarray(data, dtype=np.float32) - np.float32(fill)


def _axis_weights(n_source: int, positions: np.ndarray, length: float, average: bool) -> np.ndarray:
    """Weights [n_target, n_source] of one axis: each target voxel's value as a combination of the source voxels'.
    ``positions`` are the target voxel centres and ``length`` the target voxel's length, in source voxels. With
    ``average``, the mean of the source over the target voxel (the overlap of each source voxel's extent with it);
    otherwise linear interpolation at the centre, the edge voxels extended to the image's edge. Rows sum to the part
    of the target inside the image."""
    if average:
        lo, hi = positions[:, None] - length / 2, positions[:, None] + length / 2
        edges = np.arange(n_source)[None, :]
        overlap = np.clip(np.minimum(hi, edges + 0.5) - np.maximum(lo, edges - 0.5), 0, None)
        return overlap / length
    W = np.zeros((len(positions), n_source))
    inside = (positions >= -0.5) & (positions <= n_source - 0.5)
    p = np.clip(positions, 0, n_source - 1)
    i0 = np.minimum(np.floor(p).astype(int), max(n_source - 2, 0))
    w = p - i0
    rows = np.nonzero(inside)[0]
    W[rows, i0[rows]] += 1 - w[rows]
    if n_source > 1:
        W[rows, i0[rows] + 1] += w[rows]
    return W


def _resample(data: np.ndarray, M: np.ndarray, shape: tuple, mode: str, value, fill: float) -> torch.Tensor:
    """``data`` sampled at the target grid of ``shape``, whose voxel (i, j, k) is at source index M @ (i, j, k, 1)."""
    L = M[:3, :3]
    separable = np.allclose(L, np.diag(np.diag(L)), atol=1e-9 * np.abs(L).max())
    if mode == 'nearest':
        return _nearest(data, M, shape, value, fill)
    if separable:
        return _separable(data, M, shape, value, fill)
    return _general(data, M, shape, value, fill)


def _separable(data, M, shape, value, fill) -> torch.Tensor:
    """Axis-aligned grids: one weight matrix per axis, applied one after another, a few target slices at a time."""
    weights = []
    for axis in range(3):
        scale = abs(M[axis, axis])
        positions = M[axis, axis] * np.arange(shape[axis]) + M[axis, 3]
        W = _axis_weights(data.shape[axis], positions, scale, average=scale > AVERAGE_ABOVE)
        weights.append(torch.from_numpy(W.astype(np.float32)))
    Wx, Wy, Wz = weights
    nx, ny, nz = data.shape
    per_slice = 4.0 * (nx * ny * (max(1.0, abs(M[2, 2])) + 2) + shape[1] * nx + shape[0] * shape[1])
    step = _chunk(per_slice)
    out = torch.empty(shape, dtype=torch.float32)
    for z0 in range(0, shape[2], step):
        z1 = min(shape[2], z0 + step)
        Wzc = Wz[z0:z1]
        used = torch.nonzero(Wzc.abs().sum(dim=0)).flatten()
        if len(used) == 0:
            out[:, :, z0:z1] = 0
            continue
        s0, s1 = int(used[0]), int(used[-1]) + 1
        slab = torch.from_numpy(_values(data[:, :, s0:s1], value, fill))
        t = torch.matmul(slab, Wzc[:, s0:s1].T)                              # (nx, ny, cz)
        t = torch.tensordot(Wy, t, dims=([1], [1]))                           # (Ty, nx, cz)
        t = torch.tensordot(Wx, t, dims=([1], [1]))                           # (Tx, Ty, cz)
        out[:, :, z0:z1] = t
    return out + fill


def _subsamples(M, mode_average: bool) -> list:
    """Offsets (in target voxels) of the points averaged along each target axis."""
    offsets = []
    for t in range(3):
        length = float(np.abs(M[:3, t]).max())       # a target voxel step in source voxels, along the main axis
        n = math.ceil(length) if mode_average and length > AVERAGE_ABOVE else 1
        offsets.append((np.arange(n) + 0.5) / n - 0.5)
    return offsets


def _general(data, M, shape, value, fill) -> torch.Tensor:
    """Oblique grids: trilinear interpolation at points spread over each target voxel (one point at the centre along
    axes not averaged), averaged, a few target slices at a time."""
    from scipy.ndimage import map_coordinates
    offsets = _subsamples(M, True)
    n_sub = int(np.prod([len(o) for o in offsets]))
    step = _chunk(shape[0] * shape[1] * n_sub * 48.0)
    sub = np.stack(np.meshgrid(*offsets, indexing='ij'), axis=-1).reshape(-1, 3)        # (n_sub, 3)
    n_source = np.array(data.shape)
    out = torch.empty(shape, dtype=torch.float32)
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    for z0 in range(0, shape[2], step):
        z1 = min(shape[2], z0 + step)
        kk = np.arange(z0, z1)
        grid = np.stack(np.broadcast_arrays(ii[:, :, None], jj[:, :, None], kk[None, None, :]), axis=-1)
        points = grid[..., None, :] + sub                                    # (Tx, Ty, cz, n_sub, 3)
        coords = points @ M[:3, :3].T + M[:3, 3]
        inside = np.all((coords >= -0.5) & (coords <= n_source - 0.5), axis=-1)
        lo = int(np.clip(np.floor(coords[..., 2].min()), 0, n_source[2] - 1))
        hi = int(np.clip(np.ceil(coords[..., 2].max()), 0, n_source[2] - 1)) + 1
        slab = _values(data[:, :, lo:hi], value, fill)
        c = np.clip(coords, 0, n_source - 1)
        c[..., 2] -= lo
        sampled = map_coordinates(slab, c.reshape(-1, 3).T, order=1, mode='nearest').reshape(inside.shape)
        out[:, :, z0:z1] = torch.from_numpy((sampled * inside).mean(axis=-1).astype(np.float32))
    return out + fill


def _nearest(data, M, shape, value, fill) -> torch.Tensor:
    step = _chunk(shape[0] * shape[1] * 40.0)
    n_source = np.array(data.shape)
    out = torch.empty(shape, dtype=torch.float32)
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    for z0 in range(0, shape[2], step):
        z1 = min(shape[2], z0 + step)
        grid = np.stack(np.broadcast_arrays(ii[:, :, None], jj[:, :, None], np.arange(z0, z1)[None, None, :]), axis=-1)
        index = np.rint(grid @ M[:3, :3].T + M[:3, 3]).astype(np.int64)
        inside = np.all((index >= 0) & (index < n_source), axis=-1)
        index = np.clip(index, 0, n_source - 1)
        sampled = data[index[..., 0], index[..., 1], index[..., 2]]
        sampled = _in_label(sampled, value) if value is not None else sampled.astype(np.float32)
        out[:, :, z0:z1] = torch.from_numpy(np.where(inside, sampled, np.float32(fill)).astype(np.float32))
    return out
