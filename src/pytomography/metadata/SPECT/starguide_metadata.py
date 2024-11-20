from __future__ import annotations
from pytomography.metadata import ProjMeta
from typing import Sequence

class StarGuideProjMeta(ProjMeta):
    def __init__(
        self,
        projection_shape: Sequence,
        angles: Sequence,
        times=None,
        offsets=None,
        radii: Sequence | None = None
    ) -> None:
        """Metadtata for the StarGuide SPECT imaging system.

        Args:
            projection_shape (Sequence): Shape of the projection data (number of angles, width, height).
            angles (Sequence): Angle of each projection.
            times (Sequence, optional): Acquisition time for each projection. 
            offsets (Sequence, optional): Offset of each projection (in axial direction).
            radii (Sequence, optional): Radial distance from center of each projection; needed for PSF modeling. Defaults to None.
        """ 
        self.angles = angles
        self.radii = radii
        self.offsets = offsets
        self.times = times
        self.num_projections = len(angles)
        self.shape = projection_shape