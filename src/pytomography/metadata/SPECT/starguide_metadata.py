"""Projection metadata of the GE StarGuide, whose every view has its own angle, radius and transaxial offset."""
from __future__ import annotations
import datetime
from typing import Sequence
from pytomography.metadata import ProjMeta


class StarGuideProjMeta(ProjMeta):
    """Projection space of the GE StarGuide: the projections are [view, r, z], z rows running foot to head.

    The StarGuide has 12 swivelling CZT detector heads, 16 pixels wide, so every view has its own angle, radius and
    transaxial offset. :func:`~pytomography.io.SPECT.read_starguide` makes it.

    Args:
        projection_shape (Sequence): Shape of the projections of one energy window: (views, Lr, Lz).
        angles (Sequence): Angle β of each view, in degrees (PyTomography's convention, see
            :class:`~pytomography.metadata.SPECT.SPECTProjMeta`).
        frame_duration (Sequence, optional): Acquisition time of each view, in s. Defaults to None.
        offsets (Sequence, optional): Transaxial offset of each view's centre along r, in mm. Defaults to None.
        radii (Sequence, optional): Distance from the centre of rotation to each view's detector face, in mm; needed
            for PSF modelling. Defaults to None.
        energy_windows (Sequence[EnergyWindow], optional): Every energy window of the acquisition, its energies in
            keV. Defaults to None.
        acquisition_start (datetime.datetime, optional): When the acquisition started. Defaults to None.
        nuclide (str, optional): The radionuclide, as the source names it. Defaults to None.
        units (str, optional): The units of the projections' values. Defaults to ``'counts'``.
    """
    def __init__(
        self,
        projection_shape: Sequence,
        angles: Sequence,
        frame_duration=None,
        offsets=None,
        radii: Sequence | None = None,
        *,
        energy_windows: Sequence | None = None,
        acquisition_start: datetime.datetime | None = None,
        nuclide: str | None = None,
        units: str | None = 'counts',
    ) -> None:
        self.angles = angles
        self.radii = radii
        self.offsets = offsets
        self.frame_duration = frame_duration
        self.num_projections = len(angles)
        self.shape = tuple(projection_shape)
        self.energy_windows = list(energy_windows) if energy_windows is not None else None
        self.acquisition_start = acquisition_start
        self.nuclide = nuclide
        self.units = units
