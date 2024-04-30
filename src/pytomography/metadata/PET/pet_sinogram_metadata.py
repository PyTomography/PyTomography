from __future__ import annotations
import pytomography
from pytomography.io.PET import shared
from .pet_tof_metadata import PETTOFMeta

class PETSinogramPolygonProjMeta():
    """PET Sinogram metadata class for polygonal scanner geometry

    Args:
        info (dict): PET geometry information dictionary
        tof_meta (PETTOFMeta | None, optional): PET time of flight metadata. If None, then assumes no time of flight. Defaults to None.
    """
    def __init__(
        self,
        info: dict,
        tof_meta: PETTOFMeta | None = None
    ):  
        self.info = info
        scanner_LUT = shared.get_scanner_LUT(info)
        self.scanner_lut = scanner_LUT
        self.detector_coordinates, self.ring_coordinates = shared.sinogram_to_spatial(info)
        self.detector_coordinates = self.detector_coordinates.to(pytomography.device)
        self.ring_coordinates = self.ring_coordinates.to(pytomography.device)
        self.shape = [self.detector_coordinates.shape[0], self.detector_coordinates.shape[1], self.ring_coordinates.shape[0]]
        self.N_angles = self.detector_coordinates.shape[0]
        self.tof_meta = tof_meta