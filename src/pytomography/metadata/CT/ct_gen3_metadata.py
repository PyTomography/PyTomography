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
        detector_centers_col_idx: torch.Tensor,
        detector_centers_row_idx: torch.Tensor,
        col_angle_spacing: float,
        row_det_spacing: float,
        DSD: float,
        shape: tuple,
        patient_position: str | None = None
    ) -> None:
        r"""Metadata of a third generation CT scan (a cylindrical detector opposite a focal spot that turns, and moves along
        the axis for a helical scan), as DICOM-CT-PD files give it: the focal spot of every view in cylindrical
        coordinates. For more information, see the DICOM-CT-PD user manual in the PyTomography tutorial files.

        The geometry is put in PyTomography's object frame, the patient's LPS axes (k runs foot to head), in mm and
        radians, centred in x and y on the isocentre and along z on the middle of the focal spot path:

        * focal spot centre of a view: :math:`(\rho\cos\phi', \rho\sin\phi', z - z_c)`, with :math:`\phi' = \phi - \pi/2`
          and :math:`z_c` (``z_center``) the mean focal spot position, which is the patient z of the grid's centre;
        * flying focal spot: the focal spot itself is at :math:`(\rho + \Delta\rho, \phi' + \Delta\phi, z - z_c + \Delta z)`;
          the detector stays where the focal spot centre puts it;
        * detector element (column :math:`c`, row :math:`r`, counted from 1): focal spot centre :math:`-
          \mathrm{DSD}(\cos(\phi' + \gamma_c), \sin(\phi' + \gamma_c), 0) + (0, 0, -(r - r_0)\,\Delta z)`, with
          :math:`\gamma_c = (c - c_0)` ``col_angle_spacing``.

        The patient frame (FFS) was checked against the scanner's own images of TCIA LDCT-and-Projection-data case C145
        (GE): within 0.006 mm in x, y and z. Before PyTomography 4 the frame's z was the patient's -z (k ran head to
        foot), which put the same rays in the mirrored frame.

        Args:
            source_phis (torch.Tensor): Angle of the focal spot centre of each view, DICOM-CT-PD's (rad)
            source_rhos (torch.Tensor): Its distance from the axis (mm)
            source_zs (torch.Tensor): Its axial position, the patient's z (mm)
            source_phi_offsets (torch.Tensor): :math:`\Delta\phi` of a flying focal spot (rad)
            source_rho_offsets (torch.Tensor): :math:`\Delta\rho` of a flying focal spot (mm)
            source_z_offsets (torch.Tensor): :math:`\Delta z` of a flying focal spot (mm)
            detector_centers_col_idx (torch.Tensor): Column :math:`c_0` (from 1, fractional) of each view through which
                the ray from the focal spot centre passes the isocentre
            detector_centers_row_idx (torch.Tensor): Row :math:`r_0` (from 1, fractional) of that ray
            col_angle_spacing (float): Angle between neighbouring columns, seen from the focal spot (rad): the
                transverse spacing at the detector over ``DSD``. (Before PyTomography 4, ``col_det_spacing``,
                documented wrongly as mm.)
            row_det_spacing (float): Spacing between rows at the detector (mm)
            DSD (float): Distance from the focal spot to the detector (mm)
            shape (tuple): Shape of the projections of a view: (columns, rows)
            patient_position (str | None, optional): DICOM PatientPosition of the scan (e.g. ``'FFS'``). Defaults to None.
        """
        # copies: the geometry is shifted and rotated below, which must not change the caller's tensors
        self.source_phis = source_phis.clone()
        self.source_rhos = source_rhos.clone()
        self.source_zs = source_zs.clone()
        # the patient z of the grid's centre, then the axis centred on it
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
        self.col_angle_spacing = col_angle_spacing
        self.row_det_spacing = row_det_spacing
        self.N_angles = len(self.source_phis)
        self.DSD = DSD
        self.shape = shape
        # the object frame: x and y of DICOM-CT-PD's cylinder turned by -pi/2; z is the patient's
        self.source_phis -= torch.pi/2
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
        phis_det = (torch.arange(1,shape[0]+1) - detector_centers_col_idx[0]) * col_angle_spacing
        # rows run towards -z: row 1 is the most superior
        zs_det = -(torch.arange(1,shape[1]+1) - detector_centers_row_idx[0]) * row_det_spacing
        self.phis_det, self.zs_det = torch.meshgrid(phis_det, zs_det, indexing='ij')

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
