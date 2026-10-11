"""An image read from a file (or made from an array), with its place in the patient."""
from __future__ import annotations

import numpy as np

from pytomography.metadata import ObjectMeta
from pytomography.metadata.metadata import FRAMES


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

    Attributes:
        centre (np.ndarray): the position of the image's centre in LPS mm.
        object_meta (ObjectMeta): the image's own grid as an :class:`~pytomography.metadata.ObjectMeta` (for an image
            whose axes run along +x, +y and +z).
    """
    def __init__(self, data, affine, units: str | None = None, modality: str | None = None, meta: dict | None = None,
                 frame: str = 'patient') -> None:
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

    @property
    def shape(self) -> tuple:
        return tuple(int(n) for n in self.data.shape)

    @property
    def voxel_size(self) -> tuple:
        """The distance between neighbouring voxels along i, j and k, in mm."""
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
