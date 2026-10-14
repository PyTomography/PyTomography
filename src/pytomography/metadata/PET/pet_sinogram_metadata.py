"""Projection metadata of PET sinograms."""
from __future__ import annotations
import pytomography
from .pet_tof_metadata import PETTOFMeta

class PETSinogramPolygonProjMeta():
    """Projection space of a PET sinogram of a polygonal scanner (rings of flat detector modules).

    The sinogram is [angles, radial bins, planes] (and TOF bins): ``N / 2`` angles and ``N + 1`` radial bins for
    ``N`` crystals per ring, and ``R**2`` planes (every pair of rings) for ``R`` rings, as
    :func:`~pytomography.io.PET.listmode_to_sinogram` bins them.

    Args:
        info (dict): The scanner's description (:func:`~pytomography.io.PET.scanner_info`, or ``proj_meta.info`` of a
            reader).
        tof_meta (PETTOFMeta | None, optional): Time of flight binning. Defaults to None (no time of flight).

    Attributes:
        info (dict): The scanner's description.
        scanner_lut (torch.Tensor): Position (x, y, z) of every crystal in mm, [N_crystals, 3]
            (:func:`~pytomography.io.PET.scanner_lut`).
        detector_coordinates (torch.Tensor): (x, y) in mm of the two crystals of every (angle, radial bin),
            [N / 2, N + 1, 2, 2], on ``pytomography.device``.
        ring_coordinates (torch.Tensor): z in mm of the two rings of every plane, [R**2, 2], on
            ``pytomography.device``.
        shape (list): [N / 2, N + 1, R**2], the shape of a sinogram without TOF.
        N_angles (int): Number of angles, N / 2.
        tof_meta (PETTOFMeta | None): Time of flight binning.
    """
    def __init__(
        self,
        info: dict,
        tof_meta: PETTOFMeta | None = None
    ):  
        from pytomography.io.PET.scanner import scanner_lut
        from pytomography.io.PET.shared import sinogram_to_spatial
        self.info = info
        self.scanner_lut = scanner_lut(info)
        self.detector_coordinates, self.ring_coordinates = sinogram_to_spatial(info)
        self.detector_coordinates = self.detector_coordinates.to(pytomography.device)
        self.ring_coordinates = self.ring_coordinates.to(pytomography.device)
        self.shape = [self.detector_coordinates.shape[0], self.detector_coordinates.shape[1], self.ring_coordinates.shape[0]]
        self.N_angles = self.detector_coordinates.shape[0]
        self.tof_meta = tof_meta