from __future__ import annotations
import torch
import pytomography
import numpy as np
from pytomography.projectors import SystemMatrix
from pytomography.metadata import ObjectMeta
from pytomography.metadata.CT import CTConeBeamFlatPanelProjMeta
from torch.nn.functional import pad
try:
    import parallelproj
except:
    pass

# TODO:Place these functions in a utilities file and support more filter types
def get_discrete_ramp_FFT(n):
    nn = torch.arange(-n / 2, n / 2)
    h = torch.zeros(nn.shape, dtype=torch.float32)
    h[n//2] = 1 / 4
    odd = nn % 2 == 1
    h[odd] = -1 / (np.pi * nn[odd]) ** 2
    return torch.abs(torch.fft.fft(h))
def FBP_filter(proj, device=pytomography.device):
    pad_size = proj.shape[0] // 2
    ramp_filter = get_discrete_ramp_FFT(proj.shape[0]+2*pad_size).to(device).reshape((-1,1))
    proj_fft = pad(proj, [0,0,pad_size,pad_size])
    # filter projections
    proj_fft = torch.fft.fft(proj_fft, dim=0)
    proj_fft = proj_fft * ramp_filter
    proj_filtered = torch.fft.ifft(proj_fft, dim=0).real[pad_size:-pad_size]
    return proj_filtered

class CTConeBeamFlatPanelSystemMatrix(SystemMatrix):
    """System matrix for a cone beam CT system with a flat detector panel. Backprojection supports FBP, but only for non-helical (i.e. fixed z) geometries.

        Args:
            object_meta (ObjectMeta): Metadata for object space
            proj_meta (CTConeBeamFlatPanelProjMeta): Projection metadata for the CT system
            N_splits (int, optional): Splits up computation of forward/back projection to save GPU memory. Defaults to 1.
            device (str, optional): Device on which projections are output. Defaults to pytomography.device.
        """
    def __init__(
        self,
        object_meta: ObjectMeta,
        proj_meta: CTConeBeamFlatPanelProjMeta,
        N_splits: int = 1,
        device: str = pytomography.device
    ) -> None:
        super(CTConeBeamFlatPanelSystemMatrix, self).__init__(object_meta, proj_meta)
        # Used for parallelproj projectors
        self.origin = -(torch.tensor(object_meta.shape).to(pytomography.device)/2-0.5) * torch.tensor(object_meta.dr).to(pytomography.dtype).to(pytomography.device) # + proj_meta.COR
        self.voxel_size = torch.tensor(object_meta.dr).to(pytomography.dtype).to(pytomography.device)
        self.N_splits = N_splits
        self.device = device
        self._FBP_postweight_component1 = None
        self._FBP_preweight = None
    
    def _get_FBP_scale(self):
        return 0.5 * (2 * np.pi/ self.proj_meta.N_angles) * (self.proj_meta.DSD/self.proj_meta.DSO) / self.proj_meta.dr[0]
    
    def _get_FBP_preweight(self, idx):
        if self._FBP_preweight is None:
            s, v = self.proj_meta._get_detector_pixel_s_v(self.device)
            self._FBP_preweight = (self.proj_meta.DSD / torch.sqrt(s**2 + v**2 + self.proj_meta.DSD**2)).to(self.device)
        return self._FBP_preweight
    
    def _get_FBP_postweight(self, idx):
        # Postweight put on pytomography.device, not self.device (otherwise too slow)
        Nx, Ny, Nz = self.object_meta.shape
        dx, dy, dz = self.object_meta.dr
        du, dv = self.proj_meta.dr
        ox, oy =  self.proj_meta.COR[:2].to(pytomography.device)
        x = (torch.arange(-Nx/2+0.5, Nx/2+0.5, 1)*dx).to(pytomography.device) + ox
        y = (torch.arange(-Ny/2+0.5, Ny/2+0.5, 1)*dy).to(pytomography.device) + oy
        z = (torch.arange(-Nz/2+0.5, Nz/2+0.5, 1)*dz).to(pytomography.device)
        # Typical post-weight from FDK algorithm
        if self._FBP_postweight_component1 is None:
            xv, yv = torch.meshgrid(x, y, indexing='ij')
            post_weight = (self.proj_meta.DSO / (self.proj_meta.DSO + yv.unsqueeze(0) * torch.sin(self.proj_meta.angles.to(pytomography.device)).reshape((-1,1,1)) + xv.unsqueeze(0) * torch.cos(self.proj_meta.angles.to(pytomography.device)).reshape((-1,1,1))))**2
            self._FBP_postweight_component1 = post_weight.unsqueeze(-1)
        # Weight that removes length scaling Joseph projector to make projector "unmatched" (see Ander Biguri thesis chapter 4)
        d = -self.proj_meta.detector_orientations[idx].to(pytomography.device)
        source_pos = self.proj_meta.beam_locations[idx].to(pytomography.device)
        lx = x - source_pos[0]
        ly = y - source_pos[1]
        lz = z - source_pos[2]
        l_vec = torch.stack(torch.meshgrid(lx,ly,lz, indexing='ij'), dim=-1)
        l = torch.norm(l_vec, dim=-1)
        w = self.proj_meta.DSD**2 * l / ((l_vec*d).sum(dim=-1))**3 * dx*dy*dz / (du*dv)
        return self._FBP_postweight_component1[idx] / w
    
    def set_n_subsets(self, n_subsets: int) -> list:
        """Returns a list where each element consists of an array of indices corresponding to a partitioned version of the projections. 

        Args:
            n_subsets (int): Number of subsets to partition the projections into

        Returns:
            list: List of arrays where each array corresponds to the projection indices of a particular subset.
        """
        indices = torch.arange(self.proj_meta.N_angles).to(torch.long).to(self.device)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        self.subset_indices_array = subset_indices_array
        
    def get_projection_subset(self, projections: torch.Tensor, subset_idx: int | None) -> torch.tensor:
        """Obtains subsampled projections :math:`g_m` corresponding to subset index :math:`m`. CT conebeam flat panel partitions projections based on angle.

        Args:
            projections (torch.Tensor): total projections :math:`g`
            subset_idx (int): subset index :math:`m`

        Returns:
            torch.Tensor: subsampled projections :math:`g_m`.
        """
        if subset_idx is None:
            return projections
        else:
            subset_indices = self.subset_indices_array[subset_idx]
            proj_subset = projections[subset_indices]
            return proj_subset
        
    def get_weighting_subset(
        self,
        subset_idx: int
    ) -> float:
        r"""Computes the relative weighting of a given subset (given that the projection space is reduced). This is used for scaling parameters relative to :math:`\tilde{H}_m^T 1` in reconstruction algorithms, such as prior weighting :math:`\beta`

        Args:
            subset_idx (int): Subset index

        Returns:
            float: Weighting for the subset.
        """
        if subset_idx is None:
            return 1
        else:
            return len(self.subset_indices_array[subset_idx]) / self.proj_meta.N_angles
        
    def compute_normalization_factor(self, subset_idx: int = None):
        r"""Computes the normalization factor :math:`H^T 1`

        Args:
            subset_idx (int, optional): Subset index for ths sinogram. If None, considers all elements. Defaults to None..

        Returns:
            torch.Tensor: Normalization factor.
        """
        return self.backward(torch.ones(self.proj_meta.N_angles, *self.proj_meta.shape).to(self.device), subset_idx)
    
    def forward(
        self, object: torch.Tensor,
        subset_idx: int | None = None,
        FBP_post_weight: torch.Tensor = None,
        projection_type='matched'
    ) -> torch.Tensor:
        """Computes forward projection

        Args:
            object (torch.Tensor): Object to be forward projected
            subset_idx (int | None, optional): Subset index :math:`m` of the projection. If None, then projects to entire projection space. Defaults to None.
            FBP_post_weight (torch.Tensor, optional): _description_. Defaults to None.
            projection_type (str): Type of forward projection to use; defaults to mathced. (For implementing the adjoint of FBP, we need the option of using FBP weights in the forward projection).

        Returns:
            torch.Tensor: Projections corresponding to :math:`\int \mu dx` along all LORs.
        """
        if subset_idx is not None:
            angle_subset = self.subset_indices_array[subset_idx]
        angle_indices = torch.arange(self.proj_meta.N_angles).to(pytomography.device) if subset_idx is None else angle_subset
        proj_total = []
        for i in range(len(angle_indices)):
            idx = angle_indices[i] # index of angle
            detector_coordinates = self.proj_meta._get_detector_coordinates(idx).flatten(end_dim=1)
            beam_coordinate = self.proj_meta.beam_locations[idx].unsqueeze(0).repeat(detector_coordinates.shape[0], 1)
            if FBP_post_weight is None:
                object_i = object
            else:
                object_i = object * FBP_post_weight
            proj = parallelproj.joseph3d_fwd(
                beam_coordinate,
                detector_coordinates,
                object_i,
                self.origin,
                self.voxel_size
            ).reshape(self.proj_meta.shape)
            proj_total.append(proj.to(self.device))
        return torch.stack(proj_total)
    
    def backward(
        self,
        proj: torch.Tensor,
        subset_idx: int | None = None,
        projection_type='matched'
    ) -> torch.Tensor:
        """Computes back projection.

        Args:
            proj (torch.Tensor): Projections to be back projected
            subset_idx (int | None, optional): Subset index :math:`m` of the projection. Defaults to None.
            projection_type (str, optional): Type of back projection to use. To use with filtered back projection, use ``'FBP'``, which weights all LORs accordingly for this geometry. Defaults to ``'matched'``.

        Returns:
            torch.Tensor: _description_
        """
        if subset_idx is not None:
            angle_subset = self.subset_indices_array[subset_idx]
        angle_indices = torch.arange(self.proj_meta.N_angles).to(pytomography.device) if subset_idx is None else angle_subset
        BP = 0
        for i in range(len(angle_indices)):
            idx = angle_indices[i]
            detector_coordinates_i = self.proj_meta._get_detector_coordinates(idx).flatten(end_dim=1)
            beam_coordinate_i = self.proj_meta.beam_locations[idx].unsqueeze(0).repeat(detector_coordinates_i.shape[0], 1)
            proj_i = proj[i]
            # If FBP projection, preweight using FBP weighting and filter
            if projection_type=='FBP':
                proj_i = proj_i * self._get_FBP_preweight(idx)
                proj_i = FBP_filter(proj_i, self.device)
            # Now back project
            proj_i = proj_i.flatten().to(pytomography.device)
            BP_i = 0
            for detector_coordinates_i_s, beam_coordinate_i_s, proj_i_s in zip(torch.tensor_split(detector_coordinates_i, self.N_splits), torch.tensor_split(beam_coordinate_i, self.N_splits), torch.tensor_split(proj_i, self.N_splits)):
                BP_i = BP_i + parallelproj.joseph3d_back(
                    beam_coordinate_i_s,
                    detector_coordinates_i_s,
                    self.object_meta.shape,
                    self.origin,
                    self.voxel_size,
                    proj_i_s
                )
            if projection_type=='FBP':
                BP_i = BP_i * self._get_FBP_postweight(idx) * self._get_FBP_scale()
            BP += BP_i
        return BP