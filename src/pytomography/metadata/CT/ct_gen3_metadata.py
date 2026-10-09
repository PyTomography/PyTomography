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
        shape: tuple,
        patient_position: str | None = None
    ) -> None:
        r"""Metadata for 3rd generation clinical CT scanners. For more information, see the DICOM-CT-PD user manual in the PyTomography tutorial files. Currently only supports cylindrical detectors.

        The object is centred on the isocentre in x and y and on the middle of the focal spot path in z. The original axial position of that middle is kept in ``z_center``, and :meth:`get_patient_affine` maps object voxels to DICOM patient coordinates.

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
            patient_position (str | None, optional): DICOM PatientPosition of the scan (e.g. ``'FFS'``), used by :meth:`get_patient_affine`. Defaults to None.
        """
        # copies: the geometry is shifted and rotated below, which must not change the caller's tensors
        self.source_phis = source_phis.clone()
        self.source_rhos = source_rhos.clone()
        self.source_zs = source_zs.clone()
        # make (0,0,0) at center, keeping where the center was
        self.z_center = float(self.source_zs.double().mean())
        self.source_zs -= self.source_zs.mean()
        self.patient_position = patient_position
        # set by the DICOM-CT-PD reader when the files carry them: incident photons per detector column of every view
        # (views, columns), the water attenuation per mm, the preprocessing flags, and the spiral pitch
        self.photon_counts = None
        self.water_attenuation = None
        self.correction_flags = None
        self.spiral_pitch = None
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

    def get_patient_affine(self, object_meta) -> torch.Tensor:
        r"""Affine matrix mapping the voxel index :math:`(i, j, k, 1)` of an object reconstructed with this metadata to DICOM patient coordinates :math:`(x, y, z, 1)` in mm (LPS, the frame of ``ImagePositionPatient``). In-plane the object is centred on the isocentre, with :math:`i` along :math:`+x` and :math:`j` along :math:`+y`; along the axis :math:`z` decreases with :math:`k`, from ``z_center``, the mean focal spot position, at the middle slice.

        This was checked against the scanner's own reconstruction of a feet first supine scan (TCIA LDCT-and-Projection-data case C145, GE): resampled with this affine, the two agree to within 0.5 mm in-plane and 0.34 mm axially, and every other choice of axis directions correlates worse. Other patient positions have not been checked, so they raise rather than return an untested mapping.

        Args:
            object_meta (ObjectMeta): Object space the reconstruction is on.

        Returns:
            torch.Tensor: 4x4 affine matrix (float64).
        """
        if self.patient_position != 'FFS':
            raise NotImplementedError(f'The mapping to patient coordinates has only been verified for feet first supine (FFS) scans; this scan is {self.patient_position!r}. Compare a reconstruction with the scanner images to extend it.')
        (Nx, Ny, Nz), (dx, dy, dz) = object_meta.shape, object_meta.dr
        return torch.tensor([
            [dx, 0, 0, -(Nx - 1) / 2 * dx],
            [0, dy, 0, -(Ny - 1) / 2 * dy],
            [0, 0, -dz, self.z_center + (Nz - 1) / 2 * dz],
            [0, 0, 0, 1]
        ], dtype=torch.float64)

    def get_detector_coordinates(self, idxs: torch.Tensor[int]) -> torch.Tensor:
        """Obtain detector coordinates and the angles corresponding to idxs. They are computed on the device ``idxs`` is on, so a projector running on the GPU need not build them on the host and copy them across.

        Args:
            idxs (torch.Tensor[int]): Angle indices

        Returns:
            torch.Tensor: Detector coordinates (in XYZ) at all angle indices.
        """
        device = idxs.device
        source_phis = self.source_phis.to(device)[idxs][:,None,None]
        phis_det, zs_det = self.phis_det.to(device), self.zs_det.to(device)
        return torch.stack([
            -self.DSD*torch.cos(source_phis + phis_det[None]),
            -self.DSD*torch.sin(source_phis + phis_det[None]),
            0*source_phis + zs_det[None]
        ], dim=-1) + self.source_focal_centers.to(device)[idxs,None,None]
        