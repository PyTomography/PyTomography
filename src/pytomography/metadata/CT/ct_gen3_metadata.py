"""Projection metadata of third generation (fan beam, cylindrical detector) CT scans."""
from __future__ import annotations
import torch
from pytomography.metadata import ProjMeta

class CTGen3ProjMeta(ProjMeta):
    r"""Projection metadata of a third generation CT scan, with the geometry of every view as DICOM-CT-PD gives it.

    A third generation scanner has a cylindrical detector opposite a focal spot; they turn together about the axis and,
    in a helical scan, also move along it relative to the patient. Every view is given by its focal spot in cylindrical
    coordinates (see the DICOM-CT-PD user manual). :func:`~pytomography.io.CT.read_dicom_ct_pd` builds this metadata
    from the files, and also sets ``photon_counts``, ``water_attenuation``, ``correction_flags`` and
    ``spiral_pitch``, which are None otherwise.

    The geometry is put in PyTomography's object frame, the patient's LPS axes (k runs foot to head), in mm and
    degrees (DICOM-CT-PD's radians are converted by the reader), centred in x and y on the isocentre and along z on
    the middle of the focal spot path:

    * focal spot centre of a view: :math:`(\rho\cos\phi', \rho\sin\phi', z - z_c)`, with :math:`\phi' = \phi - 90°`
      and :math:`z_c` (``z_center``) the mean focal spot position, which is the patient z of the grid's centre;
    * flying focal spot: the focal spot itself is at :math:`(\rho + \Delta\rho, \phi' + \Delta\phi, z - z_c + \Delta z)`
      in cylindrical coordinates; the detector stays where the focal spot centre puts it;
    * detector element (column :math:`c`, row :math:`r`, counted from 1): the focal spot centre plus
      :math:`-\mathrm{DSD}\,(\cos(\phi' + \gamma_c), \sin(\phi' + \gamma_c), 0) + (0, 0, -(r - r_0)\,\delta_r)`, with
      the fan angle :math:`\gamma_c = (c - c_0)\,\delta_\gamma`, :math:`\delta_\gamma` ``col_angle_spacing`` and
      :math:`\delta_r` ``row_det_spacing``. Row 1 is the most superior.

    The patient frame (feet first supine) was checked against the scanner's own images of TCIA
    LDCT-and-Projection-data case C145 (GE): within 0.006 mm in x, y and z. Before PyTomography 4 the frame's z was the
    patient's -z (k ran head to foot), which put the same rays in the mirrored frame.

    Args:
        source_phis (torch.Tensor): Angle :math:`\phi` of the focal spot centre of each view, as DICOM-CT-PD gives
            it, in degrees, [views].
        source_rhos (torch.Tensor): Distance :math:`\rho` of the focal spot centre from the axis, in mm, [views].
        source_zs (torch.Tensor): Axial position :math:`z` of the focal spot centre, the patient's z, in mm, [views].
        source_phi_offsets (torch.Tensor): :math:`\Delta\phi` of a flying focal spot, in degrees, [views].
        source_rho_offsets (torch.Tensor): :math:`\Delta\rho` of a flying focal spot, in mm, [views].
        source_z_offsets (torch.Tensor): :math:`\Delta z` of a flying focal spot, in mm, [views].
        detector_centers_col_idx (torch.Tensor): Column :math:`c_0` (counted from 1, fractional) that the ray from the
            focal spot centre through the isocentre reaches, [views]; the first view's value is used for all.
        detector_centers_row_idx (torch.Tensor): Row :math:`r_0` (counted from 1, fractional) of that ray, [views];
            the first view's value is used for all.
        col_angle_spacing (float): Angle :math:`\delta_\gamma` between neighbouring columns, seen from the focal
            spot, in degrees: the transverse spacing at the detector over ``DSD``, as an angle.
        row_det_spacing (float): Spacing :math:`\delta_r` between rows at the detector, in mm.
        DSD (float): Distance from the focal spot to the detector, in mm.
        shape (tuple[int, int]): Shape of the projections of one view: (columns, rows).
        patient_position (str | None, optional): DICOM PatientPosition of the scan (e.g. ``'FFS'``). Defaults to None.

    Attributes:
        N_angles (int): Number of views.
        z_center (float): Mean axial position :math:`z_c` of the focal spot centre, the patient z of the grid's
            centre, in mm.
        source_phis (torch.Tensor): :math:`\phi'` of each view, the given angle minus 90, in degrees.
        source_zs (torch.Tensor): :math:`z - z_c` of each view, in mm.
        source_focal_centers (torch.Tensor): Focal spot centre of each view in the object frame, [views, 3], in mm.
        source_focal_spots (torch.Tensor): Focal spot of each view with the flying focal spot offsets, [views, 3], in
            mm.
        phis_det (torch.Tensor): Fan angle :math:`\gamma_c` of every detector element, [columns, rows], in degrees.
        zs_det (torch.Tensor): Axial offset :math:`-(r - r_0)\,\delta_r` of every detector element from the focal
            spot centre, [columns, rows], in mm.
        photon_counts (torch.Tensor | None): Incident photons per detector column of every view, [views, columns]
            (DICOM-CT-PD's PhotonStatistics), or None.
        water_attenuation (float | None): Water's linear attenuation coefficient at the scan's effective energy, in
            1/mm (DICOM-CT-PD's WaterAttenuationCoefficient), or None.
        correction_flags (dict | None): The corrections the files record as applied to the projections, e.g.
            ``{'beam hardening': True, ...}``, or None.
        spiral_pitch (float | None): DICOM SpiralPitchFactor of the scan, or None.
    """
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
        # the object frame: x and y of DICOM-CT-PD's cylinder turned by -90 degrees; z is the patient's
        self.source_phis -= 90.0
        phi, dphi = torch.deg2rad(self.source_phis), torch.deg2rad(self.source_phi_offsets)
        self.source_focal_centers = torch.stack([
            self.source_rhos*torch.cos(phi),
            self.source_rhos*torch.sin(phi),
            self.source_zs
        ], dim=-1)
        self.source_focal_spots = torch.stack([
            (self.source_rhos+self.source_rho_offsets)*torch.cos(phi + dphi),
            (self.source_rhos+self.source_rho_offsets)*torch.sin(phi + dphi),
            self.source_zs + self.source_z_offsets
        ], dim=-1)
        phis_det = (torch.arange(1,shape[0]+1) - detector_centers_col_idx[0]) * col_angle_spacing
        # rows run towards -z: row 1 is the most superior
        zs_det = -(torch.arange(1,shape[1]+1) - detector_centers_row_idx[0]) * row_det_spacing
        self.phis_det, self.zs_det = torch.meshgrid(phis_det, zs_det, indexing='ij')

    def get_detector_coordinates(self, idxs: torch.Tensor[int]) -> torch.Tensor:
        """Compute the position of every detector element in the views ``idxs``.

        They are computed on the device ``idxs`` is on, so a projector running on the GPU need not build them on the
        host and copy them across.

        Args:
            idxs (torch.Tensor): Indices of the views, [n] (integers).

        Returns:
            torch.Tensor: Centres of the detector elements in the object frame, [n, columns, rows, 3] (x, y, z), in mm,
            on the device of ``idxs``.
        """
        device = idxs.device
        source_phis = torch.deg2rad(self.source_phis.to(device)[idxs][:,None,None])
        phis_det, zs_det = torch.deg2rad(self.phis_det.to(device)), self.zs_det.to(device)
        return torch.stack([
            -self.DSD*torch.cos(source_phis + phis_det[None]),
            -self.DSD*torch.sin(source_phis + phis_det[None]),
            0*source_phis + zs_det[None]
        ], dim=-1) + self.source_focal_centers.to(device)[idxs,None,None]
