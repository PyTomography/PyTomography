"""Projection metadata of cone-beam CT with a flat panel detector."""
from __future__ import annotations
import torch
import pytomography
from pytomography.metadata import ProjMeta
import numpy as np

class CTConeBeamFlatPanelProjMeta(ProjMeta):
    r"""Projection metadata of a cone-beam CT scan with a flat panel detector, the source and panel turning about z.

    In the object frame (mm), the view at angle :math:`\theta` has its source at
    :math:`(-R_s\cos\theta, -R_s\sin\theta, z)` and its detector centre at :math:`(R_d\cos\theta, R_d\sin\theta, z)`,
    with :math:`R_s` ``beam_radius``, :math:`R_d` ``detector_radius`` and :math:`z` the view's ``z_locations``. The
    detector faces the rotation axis, its normal in the axial plane at every z; its u axis runs along
    :math:`(-\sin\theta, \cos\theta, 0)` and its v axis along -z (v = 0 is the top row). With ``COR``, the source and detector of every view are shifted so that they turn about the
    axis at (0, -COR) instead of the z axis; the view at :math:`\theta = 0` does not move.
    :func:`~pytomography.io.CT.read_nikon` builds this metadata from a Nikon XTek scan.

    Args:
        angles (torch.Tensor): Angle :math:`\theta` of each view, in radians, [views].
        z_locations (torch.Tensor): Axial position :math:`z` of the source and detector centre at each view, in mm,
            [views].
        detector_radius (float): Distance from the rotation axis to the detector along the central ray, in mm.
        beam_radius (float): Distance from the source to the rotation axis along the central ray, in mm.
        shape (tuple[int, int]): Number of detector pixels along u and v.
        dr (tuple[float, float]): Detector pixel size along u and v, in mm.
        COR (float | None, optional): Offset of the rotation axis from the central ray, in mm: the axis is at
            (0, -COR) in the object frame. Defaults to None: no offset.

    Attributes:
        N_angles (int): Number of views.
        beam_locations (torch.Tensor): Source position of each view, [views, 3], in mm.
        detector_locations (torch.Tensor): Detector centre of each view, [views, 3], in mm.
        detector_orientations (torch.Tensor): Unit normal of the detector at each view, [views, 3], in the axial plane,
            pointing from the detector centre towards the z axis (computed before the ``COR`` shift).
        DSD (float): Distance from the source to the detector, ``|beam_radius| + |detector_radius|``, in mm.
        DSO (float): Distance from the source to the rotation axis, ``|beam_radius|``, in mm.
    """
    def __init__(self,
                 angles,
                 z_locations,
                 detector_radius,
                 beam_radius,
                 shape,
                 dr,
                 COR = None
                 ):
        self.detector_radius = detector_radius
        self.beam_radius = beam_radius
        if COR is None:
            self.COR = torch.tensor([0.,0.,0.]).to(pytomography.device).to(pytomography.dtype)
        else:
            self.COR = torch.tensor([0.,-COR,0.]).to(pytomography.device).to(pytomography.dtype)
        self.angles = angles.to(pytomography.device).to(pytomography.dtype)
        self.z_locations = z_locations.to(pytomography.device).to(pytomography.dtype)
        self.detector_locations =  torch.stack([
            detector_radius * torch.cos(self.angles),
            detector_radius * torch.sin(self.angles),
            self.z_locations
        ], dim=-1)
        self.beam_locations =  torch.stack([
            - beam_radius* torch.cos(self.angles),
            - beam_radius*torch.sin(self.angles),
            self.z_locations
        ], dim=-1)
        # the panel faces the rotation axis, perpendicular to it at every z (the detector's position projected on the
        # axial plane: its position itself tilted the panel off z = 0)
        in_plane = self.detector_locations.clone()
        in_plane[:, 2] = 0
        self.detector_orientations = - in_plane / torch.norm(in_plane, dim=-1).unsqueeze(-1)
        # Rotation offset
        offsetx, offsety = self._get_CORs()
        self.detector_locations[:,0] += offsetx
        self.detector_locations[:,1] += offsety 
        self.beam_locations[:,0] += offsetx
        self.beam_locations[:,1] += offsety 
        # Other attributes
        self.shape = shape
        self.dr = dr
        self.N_angles = len(self.detector_locations)
        self.s, self.v = self._get_detector_pixel_s_v()
        self.DSD = np.abs(beam_radius) + np.abs(detector_radius)
        self.DSO = np.abs(beam_radius)
        self.DSOs = torch.norm(self.beam_locations, dim=-1)
        
    def _get_CORs(self):
        offsetx = - self.COR[0]*torch.cos(self.angles)+self.COR[1]*torch.sin(self.angles) + self.COR[0]
        offsety = - self.COR[0]*torch.sin(self.angles)-self.COR[1]*torch.cos(self.angles) + self.COR[1]
        return offsetx, offsety
        
    def _get_detector_pixel_s_v(self, device=None):
        Nx, Ny = self.shape
        dx, dy = self.dr
        s, v = torch.meshgrid(torch.arange(-Nx/2+0.5, Nx/2+0.5, 1)*dx, torch.arange(-Ny/2+0.5, Ny/2+0.5, 1)*dy, indexing='ij')
        if device is not None:
            s = s.to(device)
            v = v.to(device)
        return s, v

    def _get_detector_coordinates(self, idx):
        o = self.detector_orientations[idx].cpu() # perp to detector
        r = torch.tensor([o[1], -o[0], 0]) # top aligned in axial plane
        p = torch.cross(o,r) # other perpendicular direction
        center = self.detector_locations[idx]
        return (r*self.s.unsqueeze(-1) + p*self.v.unsqueeze(-1) + center.cpu()).to(pytomography.device)