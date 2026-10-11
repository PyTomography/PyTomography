from __future__ import annotations
import datetime
from typing import Sequence
from pytomography.metadata import ProjMeta


class StarGuideProjMeta(ProjMeta):
    def __init__(
        self,
        projection_shape: Sequence,
        angles: Sequence,
        frame_duration_s=None,
        offsets=None,
        radii: Sequence | None = None,
        *,
        energy_windows: Sequence | None = None,
        acquisition_start: datetime.datetime | None = None,
        nuclide: str | None = None,
        units: str | None = 'counts',
    ) -> None:
        """Projection space of the GE StarGuide: 12 swivelling CZT detector heads, 16 pixels wide, so every view has
        its own angle, radius and transaxial offset. The projections are [view, r, z], z rows running foot to head.

        Args:
            projection_shape (Sequence): Shape of the projection data (number of views, width, height).
            angles (Sequence): Angle β of each view (degrees, PyTomography's convention).
            frame_duration_s (Sequence, optional): Acquisition time of each view, in seconds.
            offsets (Sequence, optional): Transaxial offset of each view's centre along r, in mm.
            radii (Sequence, optional): Distance from the centre of rotation to each view's detector face, in mm; needed
                for PSF modelling. Defaults to None.
            energy_windows (Sequence[EnergyWindow], optional): every energy window of the acquisition.
            acquisition_start (datetime.datetime, optional): when the acquisition started.
            nuclide (str, optional): the radionuclide, as the source names it.
            units (str, optional): the units of the projections' values. Defaults to ``'counts'``.
        """
        self.angles = angles
        self.radii = radii
        self.offsets = offsets
        self.frame_duration_s = frame_duration_s
        self.num_projections = len(angles)
        self.shape = tuple(projection_shape)
        self.energy_windows = list(energy_windows) if energy_windows is not None else None
        self.acquisition_start = acquisition_start
        self.nuclide = nuclide
        self.units = units
