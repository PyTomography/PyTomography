"""SPECT metadata: the object grid, the projection space, the energy windows and the collimator PSF.

The readers in :mod:`pytomography.io.SPECT` make the first three; :func:`~pytomography.io.SPECT.collimator_psf` and
:func:`~pytomography.io.SPECT.simind_psf` make the PSF. All lengths are in mm, energies in keV and times in s.
"""
from __future__ import annotations
import copy
import datetime
import warnings
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pytomography
from pytomography.utils import compute_pad_size
import torch
from..metadata import ObjectMeta, ProjMeta


@dataclass(frozen=True)
class EnergyWindow:
    """An energy window of a SPECT acquisition (immutable).

    Args:
        name (str): Its name, as the source gives it.
        ranges (tuple): Every (lower, upper) range of the window in keV; most windows have one, some (Ga-67)
            several.
        photon_energy (float): The photon energy the window stands for, in keV: the nuclide's strongest line inside it,
            else the window's centre. Attenuation maps and collimator PSFs for this window are made at this energy.
    """
    name: str
    ranges: tuple
    photon_energy: float

    def __post_init__(self):
        object.__setattr__(self, 'ranges', tuple((float(lo), float(hi)) for lo, hi in self.ranges))
        object.__setattr__(self, 'photon_energy', float(self.photon_energy))

    @property
    def width(self) -> float:
        """The total width of the window's ranges, in keV, as the energy window scatter estimates use."""
        return float(sum(hi - lo for lo, hi in self.ranges))

    @property
    def lower(self) -> float:
        """The lowest bound of the window's ranges, in keV."""
        return min(lo for lo, _ in self.ranges)

    @property
    def upper(self) -> float:
        """The highest bound of the window's ranges, in keV."""
        return max(hi for _, hi in self.ranges)

    def __repr__(self):
        ranges = ', '.join(f'{lo:g}-{hi:g}' for lo, hi in self.ranges)
        return f'EnergyWindow({self.name!r}, {ranges} keV, photon {self.photon_energy:g} keV)'


class SPECTObjectMeta(ObjectMeta):
    """Object space of a SPECT reconstruction: an ObjectMeta with the padding the rotate-and-sum projectors need.

    It is an :class:`~pytomography.metadata.ObjectMeta` whose x and y are padded for rotations (so that nothing is cut
    off when the object is rotated). A warning says when the voxel sizes look like cm (all below 1): they are in mm
    since PyTomography 4.0.

    Args:
        dr (Sequence[float]): Voxel size along x, y and z, in mm.
        shape (Sequence[int]): Number of voxels [Lx, Ly, Lz]; Lx = Ly.
        centre (Sequence[float], optional): Where the rotation axis and the middle projection row are, in LPS mm.
            Defaults to (0, 0, 0).
        frame (str, optional): ``'patient'`` or ``'scanner'`` (see :class:`~pytomography.metadata.ObjectMeta`).
            Defaults to ``'scanner'``.
        units (str, optional): The units of the object's values. Defaults to ``'counts'``.
        modality (str, optional): The modality, for the writers. Defaults to ``'NM'``.
    """
    def __init__(self, dr: Sequence[float], shape: Sequence[int], centre: Sequence[float] = (0.0, 0.0, 0.0),
                 frame: str = 'scanner', units: str | None = 'counts', modality: str | None = 'NM') -> None:
        super().__init__(dr, shape, centre=centre, frame=frame, units=units, modality=modality)
        if max(self.dr) < 1.0:
            warnings.warn(f'SPECT voxel sizes are in mm (since PyTomography 4.0); dr = {self.dr} looks like cm',
                          stacklevel=2)
        self.recon_method = None

    @property
    def pad_size(self) -> int:
        """Number of voxels added on each side of x and y for rotations."""
        return int(compute_pad_size(self.shape[0]))

    @property
    def padded_shape(self) -> tuple:
        """The shape of the padded object, (Lx + 2 pad_size, Ly + 2 pad_size, Lz)."""
        return (self.shape[0] + 2 * self.pad_size, self.shape[1] + 2 * self.pad_size, self.shape[2])


class SPECTProjMeta(ProjMeta):
    r"""Projection space of a parallel-hole SPECT acquisition: the projections are [θ, r, z] (one energy window).

    The geometry is PyTomography's: the angle β of view θ is in degrees, β = 0 puts the detector anterior and β = 90
    at the patient's right, β grows counter-clockwise seen from the feet; r runs along (cos β, -sin β, 0), with the
    centre column on the rotation axis; z rows run foot to head.

    Args:
        projection_shape (Sequence[int]): (Lr, Lz), the pixels of each projection.
        dr (Sequence[float]): (dr, dz), the pixel size in mm.
        angles (Sequence[float]): β of every view, in degrees.
        radii (Sequence[float], optional): The distance from the rotation axis to the detector face of every view, in
            mm (needed for PSF modelling). Defaults to None.
        energy_windows (Sequence[EnergyWindow], optional): Every energy window of the acquisition (all of them, not
            only the one reconstructed), its energies in keV. Defaults to None.
        frame_duration (float | Sequence[float], optional): The acquisition time of each view, in s (one number is
            used for every view). Defaults to None.
        acquisition_start (datetime.datetime, optional): When the acquisition started. Defaults to None.
        nuclide (str, optional): The radionuclide, as the source names it. Defaults to None.
        units (str, optional): The units of the projections' values. Defaults to ``'counts'``.

    Attributes:
        angles (torch.Tensor): β of every view, in degrees, on ``pytomography.device``.
        radii (np.ndarray | None): The radius of every view, in mm.
        dr (tuple): (dr, dz), the pixel size in mm.
        shape (tuple): (θ, Lr, Lz), the shape of the projections of one window.
        num_projections (int): Number of views.
        energy_windows (list[EnergyWindow] | None): Every energy window of the acquisition.
        frame_duration (np.ndarray | None): The acquisition time of every view, in s.
        acquisition_start (datetime.datetime | None): When the acquisition started.
        nuclide (str | None): The radionuclide.
        units (str | None): The units of the projections' values.
        pad_size (int): Number of pixels added on each side of r for rotations.
        padded_shape (tuple): (θ, Lr + 2 pad_size, Lz).
    """
    def __init__(
        self,
        projection_shape: Sequence,
        dr: Sequence[float],
        angles: Sequence,
        radii=None,
        *,
        energy_windows: Sequence[EnergyWindow] | None = None,
        frame_duration: float | Sequence[float] | None = None,
        acquisition_start: datetime.datetime | None = None,
        nuclide: str | None = None,
        units: str | None = 'counts',
    ) -> None:
        self.angles = torch.as_tensor(np.asarray(angles, dtype=np.float64)).to(pytomography.dtype).to(pytomography.device)
        self.dr = tuple(float(d) for d in dr)
        self.radii = None if radii is None else np.asarray(radii, dtype=np.float64)
        self.num_projections = len(angles)
        self.shape = (self.num_projections, int(projection_shape[0]), int(projection_shape[1]))
        self.energy_windows = list(energy_windows) if energy_windows is not None else None
        if frame_duration is not None and np.ndim(frame_duration) == 0:
            frame_duration = np.full(self.num_projections, float(frame_duration))
        self.frame_duration = None if frame_duration is None else np.asarray(frame_duration, dtype=np.float64)
        self.acquisition_start = acquisition_start
        self.nuclide = nuclide
        self.units = units
        self.compute_padded_shape()

    def compute_padded_shape(self) -> list:
        """Sets ``pad_size`` and ``padded_shape``, the projection shape padded along r for rotations.

        The readers and :meth:`subsample` call it; call it again after changing ``shape`` by hand.
        """
        self.pad_size = compute_pad_size(self.shape[1])
        theta_padded = self.shape[0]
        r_padded = self.shape[1] + 2*self.pad_size
        z_padded = self.shape[2]
        self.padded_shape =  (int(theta_padded), int(r_padded), int(z_padded))

    def subsample(self, factor: int = 1, angle_factor: int = 1, angle_start: int = 0) -> 'SPECTProjMeta':
        """The projection space of subsampled projections: pixels ``factor`` times larger, fewer views.

        It goes with :func:`pytomography.io.SPECT.shared.subsample_projections` called with the same arguments, and
        with ``object_meta.subsample(factor)``. The angles, radii and frame durations of the views kept are kept.

        Args:
            factor (int, optional): Pixel reduction factor along r and z. Defaults to 1.
            angle_factor (int, optional): Keep every ``angle_factor``-th view (a factor, no units). Defaults to 1.
            angle_start (int, optional): Index of the first view kept. Defaults to 0.

        Returns:
            SPECTProjMeta: A new projection space; this one is unchanged.
        """
        new = copy.copy(self)
        views = slice(angle_start, None, angle_factor)
        new.angles = self.angles[views]
        new.radii = None if self.radii is None else self.radii[views]
        new.frame_duration = None if self.frame_duration is None else self.frame_duration[views]
        new.num_projections = len(new.angles)
        new.dr = tuple(d * factor for d in self.dr)
        new.shape = (new.num_projections, self.shape[1] // factor, self.shape[2] // factor)
        new.compute_padded_shape()
        return new

class SPECTPSFMeta():
    r"""The collimator-detector response of a SPECT camera: the PSF's width as a function of the distance, in mm.

    The PSF at distance :math:`d` from the detector has width :math:`\sigma(d) = f(d, p)`, with :math:`f` the
    ``sigma_fit`` function and :math:`p` its ``sigma_fit_params``. With ``shape='gaussian'`` the PSF is the Gaussian
    :math:`e^{-x^2/(2\sigma(d)^2)}` along r and z (normalised); with ``shape='square'`` (square holes) it is a
    triangle that reaches zero at :math:`\sigma(d)`, the geometric resolution :math:`D (L + d) / L` of the
    collimator, which is also the triangle's full width at half maximum.
    :class:`~pytomography.transforms.SPECT.SPECTPSFTransform` applies it, with convolutions whose kernel size is fixed
    by ``min_sigmas``. Usually made by :func:`~pytomography.io.SPECT.collimator_psf` or
    :func:`~pytomography.io.SPECT.simind_psf`.

    Args:
        sigma_fit_params (Sequence[float]): Parameters of ``sigma_fit``, in mm (slopes are dimensionless).
        sigma_fit (Callable, optional): Function ``sigma_fit(d, *sigma_fit_params)`` that gives the width at the
            distances ``d`` (mm). Defaults to the linear model ``a * d + b``.
        kernel_dimensions (str, optional): ``'2D'``: blur along r and along z (two separable 1D convolutions).
            ``'1D'``: blur along r only, each plane of constant z on its own. Defaults to ``'2D'``.
        min_sigmas (float, optional): Half-width of the convolution kernels, in standard deviations (widths) at the
            largest distance in the
            object: the Gaussian is infinite and must be cut off, and the kernel size is the same at every distance, so
            every distance gets at least ``min_sigmas`` widths. Ignored (set to 1, the whole triangle) for
            ``shape='square'``. Defaults to 3.
        shape (str, optional): ``'gaussian'`` or ``'square'`` (see above). Defaults to ``'gaussian'``.

    Raises:
        ValueError: If ``shape`` is not ``'gaussian'`` or ``'square'``.
    """
    def __init__(
        self,
        sigma_fit_params: Sequence[float, float],
        sigma_fit : function = lambda r, a, b: a*r+b,
        kernel_dimensions: str = '2D',
        min_sigmas: float | None = 3,
        shape: str = 'gaussian'
    ) -> None:
        if shape not in ('gaussian', 'square'):
            raise ValueError(f"shape must be 'gaussian' or 'square', not {shape!r}")
        self.sigma_fit_params = sigma_fit_params
        self.sigma_fit = sigma_fit
        self.kernel_dimensions = kernel_dimensions
        if shape == 'square':
            self.min_sigmas = 1 # will include whole PSF
        else:
            self.min_sigmas = min_sigmas
        self.shape = shape

    def __repr__(self):
        attributes = [f"{attr} = {getattr(self, attr)}\n" for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        return ''.join(attributes)
