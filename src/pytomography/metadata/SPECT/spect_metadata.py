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
    """An energy window of a SPECT acquisition.

    Args:
        name (str): its name, as the source gives it.
        ranges_keV (tuple): every (lower, upper) range of the window in keV; most windows have one, some (Ga-67)
            several.
        photon_keV (float): the photon energy the window stands for: the nuclide's strongest line inside it, else the
            window's centre. Attenuation maps and collimator PSFs for this window are made at this energy.
    """
    name: str
    ranges_keV: tuple
    photon_keV: float

    def __post_init__(self):
        object.__setattr__(self, 'ranges_keV', tuple((float(lo), float(hi)) for lo, hi in self.ranges_keV))
        object.__setattr__(self, 'photon_keV', float(self.photon_keV))

    @property
    def width_keV(self) -> float:
        """The total width of the window's ranges (keV), as the energy window scatter estimates use."""
        return float(sum(hi - lo for lo, hi in self.ranges_keV))

    @property
    def lower_keV(self) -> float:
        return min(lo for lo, _ in self.ranges_keV)

    @property
    def upper_keV(self) -> float:
        return max(hi for _, hi in self.ranges_keV)

    def __repr__(self):
        ranges = ', '.join(f'{lo:g}-{hi:g}' for lo, hi in self.ranges_keV)
        return f'EnergyWindow({self.name!r}, {ranges} keV, photon {self.photon_keV:g} keV)'


class SPECTObjectMeta(ObjectMeta):
    """Object space of a SPECT reconstruction: an :class:`~pytomography.metadata.ObjectMeta` with the padding the
    rotate-and-sum projectors need (so that nothing is cut off when the object is rotated).

    Args:
        dr (Sequence[float]): voxel size along x, y and z, in mm.
        shape (Sequence[int]): number of voxels [Lx, Ly, Lz]; Lx = Ly.
        centre (Sequence[float], optional): where the rotation axis and the middle projection row are, in LPS mm.
            Defaults to (0, 0, 0).
        frame (str, optional): ``'patient'`` or ``'scanner'`` (see :class:`~pytomography.metadata.ObjectMeta`).
            Defaults to ``'scanner'``.
        units (str, optional): the units of the object's values. Defaults to ``'counts'``.
        modality (str, optional): Defaults to ``'NM'``.
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
        """Voxels added on each side of x and y for rotations."""
        return int(compute_pad_size(self.shape[0]))

    @property
    def padded_shape(self) -> tuple:
        """The shape of the padded object."""
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
        radii (Sequence[float], optional): the distance from the rotation axis to the detector face of every view, in
            mm (needed for PSF modelling). Defaults to None.
        energy_windows (Sequence[EnergyWindow], optional): every energy window of the acquisition (all of them, not
            only the one reconstructed). Defaults to None.
        frame_duration_s (float | Sequence[float], optional): the acquisition time of each view, in seconds.
        acquisition_start (datetime.datetime, optional): when the acquisition started.
        nuclide (str, optional): the radionuclide, as the source names it.
        units (str, optional): the units of the projections' values. Defaults to ``'counts'``.
    """
    def __init__(
        self,
        projection_shape: Sequence,
        dr: Sequence[float],
        angles: Sequence,
        radii=None,
        *,
        energy_windows: Sequence[EnergyWindow] | None = None,
        frame_duration_s: float | Sequence[float] | None = None,
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
        if frame_duration_s is not None and np.ndim(frame_duration_s) == 0:
            frame_duration_s = np.full(self.num_projections, float(frame_duration_s))
        self.frame_duration_s = None if frame_duration_s is None else np.asarray(frame_duration_s, dtype=np.float64)
        self.acquisition_start = acquisition_start
        self.nuclide = nuclide
        self.units = units
        self.compute_padded_shape()

    def compute_padded_shape(self) -> list:
        """Computes the padded shape of an object required when rotating the object (to avoid anything getting cut off).
        """
        self.pad_size = compute_pad_size(self.shape[1])
        theta_padded = self.shape[0]
        r_padded = self.shape[1] + 2*self.pad_size
        z_padded = self.shape[2]
        self.padded_shape =  (int(theta_padded), int(r_padded), int(z_padded))

    def subsample(self, factor: int = 1, angle_factor: int = 1, angle_start: int = 0) -> 'SPECTProjMeta':
        """The projection space of projections subsampled ``factor`` times in r and z and keeping every
        ``angle_factor``-th view from ``angle_start`` (see :func:`pytomography.io.SPECT.shared.subsample_projections`).

        Returns:
            SPECTProjMeta: a new projection space; this one is unchanged.
        """
        new = copy.copy(self)
        views = slice(angle_start, None, angle_factor)
        new.angles = self.angles[views]
        new.radii = None if self.radii is None else self.radii[views]
        new.frame_duration_s = None if self.frame_duration_s is None else self.frame_duration_s[views]
        new.num_projections = len(new.angles)
        new.dr = tuple(d * factor for d in self.dr)
        new.shape = (new.num_projections, self.shape[1] // factor, self.shape[2] // factor)
        new.compute_padded_shape()
        return new

class SPECTPSFMeta():
    r"""Metadata for PSF correction. PSF blurring is implemented using Gaussian blurring with :math:`\sigma(r) = f(r,p)` where :math:`r` is the distance from the detector, :math:`\sigma` is the width of the Gaussian blurring at that location, and :math:`f(r,p)` is the ``sigma_fit`` function which takes in additional parameters :math:`p` called ``sigma_fit_params``. (By default, ``sigma_fit`` is a linear curve). As such, :math:`\frac{1}{\sigma\sqrt{2\pi}}e^{-r^2/(2\sigma(r)^2)}` is the point spread function. Blurring is implemented using convolutions with a specified kernel size. Distances and widths are in mm, like the object's voxels.

     Args:
        sigma_fit_params (float): Parameters to the sigma fit function, in mm (slopes are dimensionless).
        sigma_fit (function): Function used to model blurring as a function of radial distance. Defaults to a 2 parameter linear model.
        kernel_dimensions (str): If '1D', blurring is done seperately in each axial plane (so only a 1 dimensional convolution is used). If '2D', blurring is mixed between axial planes (so a 2D convolution is used). Defaults to '2D'.
        min_sigmas (float, optional): This is the number of sigmas to consider in PSF correction. PSF are modelled by Gaussian functions whose extension is infinite, so we need to crop the Gaussian when computing this operation numerically. Note that the blurring width is depth dependent, but the kernel size used for PSF blurring is constant. As such, this parameter is used to fix the kernel size such that all locations have at least ``min_sigmas`` of a kernel size.
        shape (str, optional): Shape of the PSF. Defaults to 'gaussian', in which case sigma is the sigma of the Gaussian. Can also be 'square', for square holes: the PSF along r and along z is then a triangle that reaches zero at ``sigma``, the geometric resolution D (L + d) / L of the collimator, which is also the triangle's full width at half maximum.
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
