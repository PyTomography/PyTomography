from __future__ import annotations
import torch
from pytomography.metadata import ProjMeta

class CTGen3ProjMeta(ProjMeta):
    def __init__(
        self,
        source_phis: torch.Tensor,
        source_rhos: torch.Tensor,
        source_zs: torch.Tensor,
        source_phi_offsets: torch.Tensor,
        source_rho_offsets: torch.Tensor,
        source_z_offsets: torch.Tensor,
        detector_centers_col_idx: float,
        detector_centers_row_idx: float,
        col_det_spacing: float,
        row_det_spacing: float,
        DSD: float,
        shape: tuple
    ) -> None:
        r"""Metadata for 3rd generation clinical CT scanners. For more information, see the DICOM-CT-PD user manual in the PyTomography tutorial files. Currently only supports cylindrical detectors.

        Args:
            source_phis (torch.Tensor): Angle of detectors in cylindrical coordinates
            source_rhos (torch.Tensor): Radius of detectors in cylindrical coordinates
            source_zs (torch.Tensor): Z coordinate of detectors (cylindrical coordinates)
            source_phi_offsets (torch.Tensor): :math:`\phi` offset if flying focal spot used
            source_rho_offsets (torch.Tensor): :math:`\rho` offset if flying focal spot used
            source_z_offsets (torch.Tensor): :math:`z` offset if flying focal spot used
            detector_centers_col_idx (float): Detector element (in column) that aligns with detectors focal center and isocenter
            detector_centers_row_idx (float): Detector element (in row) that aligns with detectors focal center and isocenter
            col_det_spacing (float): Spacing between columns of detector data (in mm)
            row_det_spacing (float): Spacing between rows of detector data (in mm)
            DSD (float): Distance between focal spot and detector center.
            shape (tuple): Shape of projection data
        """
        self.source_phis = source_phis
        self.source_rhos = source_rhos
        self.source_zs = source_zs
        # make (0,0,0) at center
        self.source_zs -= self.source_zs.mean()
        self.source_phi_offsets = source_phi_offsets
        self.source_z_offsets = source_z_offsets
        self.source_rho_offsets = source_rho_offsets
        self.detector_centers_col_idx = detector_centers_col_idx
        self.detector_centers_row_idx = detector_centers_row_idx
        self.col_det_spacing = col_det_spacing
        self.row_det_spacing = row_det_spacing
        self.N_angles = len(self.source_phis)
        self.DSD = DSD
        self.shape = shape
        # Reorient for DICOM system
        self.source_zs = - self.source_zs
        self.source_z_offsets = - self.source_z_offsets
        self.source_phis -= torch.pi/2
        # Compute things that are required
        self.source_focal_centers = torch.stack([
            self.source_rhos*torch.cos(self.source_phis),
            self.source_rhos*torch.sin(self.source_phis),
            self.source_zs
        ], dim=-1)
        self.source_focal_spots = torch.stack([
            (self.source_rhos+self.source_rho_offsets)*torch.cos((self.source_phis+self.source_phi_offsets)),
            (self.source_rhos+self.source_rho_offsets)*torch.sin((self.source_phis+self.source_phi_offsets)),
            self.source_zs + self.source_z_offsets
        ], dim=-1)
        phis_det = (torch.arange(1,shape[0]+1) - detector_centers_col_idx[0]) * col_det_spacing
        zs_det = (torch.arange(1,shape[1]+1) - detector_centers_row_idx[0]) * row_det_spacing
        #zs_det = torch.flip(zs_det, dims=(0,))
        self.phis_det, self.zs_det = torch.meshgrid(phis_det, zs_det, indexing='ij')
        
    def get_detector_coordinates(self, idxs: torch.Tensor[int]) -> torch.Tensor:
        """Obtain detector coordinates and the angles corresponding to idxs

        Args:
            idxs (torch.Tensor[int]): Angle indices

        Returns:
            torch.Tensor: Detector coordinates (in XYZ) at all angle indices.
        """
        return torch.stack([
            -self.DSD*torch.cos(self.source_phis[idxs][:,None,None] + self.phis_det[None]),
            -self.DSD*torch.sin(self.source_phis[idxs][:,None,None] + self.phis_det[None]),
            0*self.source_zs[idxs][:,None,None] + self.zs_det[None]
        ], dim=-1) + self.source_focal_centers[idxs,None,None]
        