import pytomography
from pytomography.io.PET import shared

class PETSinogramPolygonProjMeta():
    def __init__(
        self,
        info,
    ):  
        self.info = info
        scanner_LUT = shared.get_scanner_LUT(info)
        self.detector_coordinates, self.ring_coordinates = shared.sinogram_to_spatial(info, scanner_LUT)
        self.detector_coordinates = self.detector_coordinates.to(pytomography.device)
        self.ring_coordinates = self.ring_coordinates.to(pytomography.device)
        self.shape = [self.detector_coordinates.shape[0], self.detector_coordinates.shape[1], self.ring_coordinates.shape[0]]
        self.N_angles = self.detector_coordinates.shape[0]