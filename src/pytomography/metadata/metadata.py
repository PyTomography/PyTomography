from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

#: The values ``ObjectMeta.frame`` can take (see :class:`ObjectMeta`).
FRAMES = ('scanner', 'patient')


def _three(values, kind, name: str) -> tuple:
    """``values`` (one number, or three numbers from a list, tuple, numpy array or torch tensor) as a tuple of three
    Python numbers of type ``kind``."""
    if hasattr(values, 'detach'):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 1:
        values = np.repeat(values, 3)
    if values.size != 3:
        raise ValueError(f'{name} needs 3 values (x, y, z), got {values.size}')
    if kind is int:
        if not np.all(values == np.round(values)):
            raise ValueError(f'{name} must be whole numbers, got {tuple(values)}')
        return tuple(int(v) for v in values)
    return tuple(float(v) for v in values)


class ObjectMeta():
    r"""The grid of an object (an image or a reconstruction), and where that grid is in the patient.

    PyTomography keeps every object as a tensor with axes ``(x, y, z)`` on a regular grid, in one frame for every
    modality: DICOM's patient coordinates, LPS, in mm. Index ``i`` runs towards the patient's left (+x), ``j`` towards
    posterior (+y) and ``k`` towards superior (+z), so ``k = 0`` is the most inferior slice. Readers turn the
    scanner's geometry into these axes; arrays are never flipped or permuted against them.

    Args:
        dr (Sequence[float]): voxel size along x, y and z, in mm.
        shape (Sequence[int]): number of voxels along x, y and z.
        centre (Sequence[float], optional): the position of the grid's centre in LPS mm (the centre of the middle
            voxel, or the point between the middle voxels when a size is even). Projectors place the grid centred on
            the scanner's centre (the SPECT rotation axis, the PET ring centre, the CT isocentre), so this is where the
            scanner's centre is in the patient. Defaults to (0, 0, 0).
        frame (str, optional): where the origin of the coordinates is. ``'patient'``: the study's DICOM patient
            coordinates, shared with its CT and other images (the readers of DICOM data set it). ``'scanner'``: LPS
            axes with the origin at the scanner's centre (simulations, and data without a patient frame). Defaults to
            ``'scanner'``.
        units (str, optional): the units of the object's values, e.g. ``'counts'``, ``'Bq/mL'``, ``'1/mm'``,
            ``'HU'``. Writers use them. Defaults to None (unknown).
        modality (str, optional): ``'NM'``, ``'PT'`` or ``'CT'``. Writers use it. Defaults to None (unknown).
    """
    #: mm per unit of ``dr``.
    _mm_per_unit = 1.0

    def __init__(self, dr: Sequence[float], shape: Sequence[int], centre: Sequence[float] = (0.0, 0.0, 0.0),
                 frame: str = 'scanner', units: str | None = None, modality: str | None = None) -> None:
        self.dr = dr
        self.shape = shape
        self.centre = centre
        self.frame = frame
        self.units = units
        self.modality = modality

    # ---------- the grid ----------

    @property
    def dr(self) -> tuple:
        return self._dr

    @dr.setter
    def dr(self, value) -> None:
        dr = _three(value, float, 'dr')
        if min(dr) <= 0:
            raise ValueError(f'voxel sizes must be positive, got dr = {dr}')
        self._dr = dr

    @property
    def dx(self) -> float:
        return self._dr[0]

    @property
    def dy(self) -> float:
        return self._dr[1]

    @property
    def dz(self) -> float:
        return self._dr[2]

    @property
    def shape(self) -> tuple:
        return self._shape

    @shape.setter
    def shape(self, value) -> None:
        shape = _three(value, int, 'shape')
        if min(shape) < 1:
            raise ValueError(f'the shape must be at least 1 voxel along each axis, got {shape}')
        self._shape = shape

    @property
    def centre(self) -> tuple:
        return self._centre

    @centre.setter
    def centre(self, value) -> None:
        self._centre = _three(value, float, 'centre')

    @property
    def frame(self) -> str:
        return self._frame

    @frame.setter
    def frame(self, value: str) -> None:
        if value not in FRAMES:
            raise ValueError(f"frame must be one of {FRAMES}, not {value!r}")
        self._frame = value

    @property
    def modality(self) -> str | None:
        return self._modality

    @modality.setter
    def modality(self, value: str | None) -> None:
        self._modality = None if value is None else str(value).upper()

    # ---------- where the grid is ----------

    @property
    def affine(self) -> np.ndarray:
        """4 x 4 float64: voxel index (i, j, k) to the voxel's centre in LPS mm. Derived from ``dr``, ``shape`` and
        ``centre``, so it is always up to date: ``diag(dx, dy, dz)`` with translation ``centre - (shape - 1) / 2 * dr``."""
        d = np.asarray(self._dr) * self._mm_per_unit
        A = np.diag([*d, 1.0])
        A[:3, 3] = np.asarray(self._centre) - (np.asarray(self._shape) - 1) / 2 * d
        return A

    def with_grid(self, dr: Sequence[float] | None = None, shape: Sequence[int] | None = None) -> 'ObjectMeta':
        """A new grid with the same centre, frame, units and modality (and, for a subclass, its other settings).

        Args:
            dr (Sequence[float], optional): the new voxel size. Defaults to this grid's.
            shape (Sequence[int], optional): the new shape. Defaults to this grid's.

        Returns:
            ObjectMeta: the new grid, of the same class as this one.
        """
        new = copy.copy(self)
        if dr is not None:
            new.dr = dr
        if shape is not None:
            new.shape = shape
        return new

    def subsample(self, factor: int | Sequence[int]) -> 'ObjectMeta':
        """The grid with voxels ``factor`` times larger and ``factor`` times fewer of them along each axis (a
        remainder is dropped), centred where this one is.

        Args:
            factor (int | Sequence[int]): one factor for every axis, or one per axis (x, y, z).

        Returns:
            ObjectMeta: the coarser grid, of the same class as this one.
        """
        f = _three(factor, int, 'factor')
        if min(f) < 1:
            raise ValueError(f'subsampling factors must be at least 1, got {f}')
        shape = tuple(n // k for n, k in zip(self._shape, f))
        if min(shape) < 1:
            raise ValueError(f'a grid of shape {self._shape} cannot be subsampled by {f}')
        return self.with_grid(dr=tuple(d * k for d, k in zip(self._dr, f)), shape=shape)

    def __repr__(self):
        return (f'{type(self).__name__}(dr={self.dr}, shape={self.shape}, centre={self.centre}, '
                f'frame={self.frame!r}, units={self.units!r}, modality={self.modality!r})')


class ProjMeta():
    """Parent class for all different types of Projection Space Metadata. Implementation and required parameters will differ significantly between different imaging modalities.
    """
    def __init__(self, angles) -> None:
        self.angles = angles
        self.num_projections = len(angles)

    def __repr__(self):
        attributes = [f"{attr} = {getattr(self, attr)}\n" for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        return ''.join(attributes)
