from __future__ import annotations
import torch
import pytomography
from pytomography.metadata import ProjMeta
import numpy as np

class CTConeBeamFlatPanelProjMeta(ProjMeta):
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
        self.detector_orientations = - self.detector_locations / torch.norm(self.detector_locations, dim=-1).unsqueeze(-1)
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