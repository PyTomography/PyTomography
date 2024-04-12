from __future__ import annotations
import torch
import pytomography
from pytomography.metadata import ObjectMeta, PETSinogramPolygonProjMeta
import numpy as np
from pytomography.projectors import SystemMatrix
from pytomography.transforms import Transform
try:
    import parallelproj
except:
    Exception('The PETSinogramSystemMatrix requires the parallelproj package to be installed. Please install it at https://parallelproj.readthedocs.io/en/stable/')

class PETSinogramSystemMatrix(SystemMatrix):
    def __init__(
        self,
        object_meta: ObjectMeta,
        proj_meta: PETSinogramPolygonProjMeta,
        obj2obj_transforms: list[Transform] = [],
        attenuation_map: torch.tensor | None = None,
        norm_sinogram: torch.tensor | None = None,
        N_splits: int = 1,
        device: str = pytomography.device,
    ) -> None:
        super(PETSinogramSystemMatrix, self).__init__(
            obj2obj_transforms=obj2obj_transforms,
            proj2proj_transforms=[],
            object_meta=object_meta,
            proj_meta=proj_meta
            )
        self.object_origin = (- np.array(object_meta.shape) / 2 + 0.5) * (np.array(object_meta.dr))
        self.obj2obj_transforms = obj2obj_transforms
        self.proj_meta = proj_meta
        # In case they get put on another device
        self.attenuation_map = attenuation_map
        self.norm_sinogram = norm_sinogram
        if norm_sinogram is not None:
            self.norm_sinogram = self.norm_sinogram.to(pytomography.device)
        self.N_splits = N_splits
        self._get_xyz_sinogram_coordinates()
        
    def set_n_subsets(self, n_subsets: int) -> list:
        indices = torch.arange(self.proj_meta.N_angles).to(torch.long).to(pytomography.device)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        self.subset_indices_array = subset_indices_array
        
    def get_projection_subset(self, projections: torch.Tensor, subset_idx: int | None) -> torch.tensor:
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
        if subset_idx is None:
            return 1
        else:
            return len(self.subset_indices_array[subset_idx]) / self.proj_meta.N_angles
    
    def _get_xyz_sinogram_coordinates(self):
        xy1 = torch.flatten(torch.tensor(self.proj_meta.detector_coordinates), start_dim=0, end_dim=1)[:,0].cpu()
        xy2 = torch.flatten(torch.tensor(self.proj_meta.detector_coordinates), start_dim=0, end_dim=1)[:,1].cpu()
        z1, z2 = torch.tensor(self.proj_meta.ring_coordinates).T.cpu()
        xyz1 = torch.concatenate([
            xy1.unsqueeze(1).repeat(1,z1.shape[0],1),
            z1.unsqueeze(0).unsqueeze(-1).repeat(xy1.shape[0],1,1)
        ], dim=-1).flatten(start_dim=0,end_dim=1)
        xyz2 = torch.concatenate([
            xy2.unsqueeze(1).repeat(1,z2.shape[0],1),
            z2.unsqueeze(0).unsqueeze(-1).repeat(xy2.shape[0],1,1)
        ], dim=-1).flatten(start_dim=0,end_dim=1)
        self.xyz1 = xyz1.reshape((*self.proj_meta.shape, 3))
        self.xyz2 = xyz2.reshape((*self.proj_meta.shape, 3))
    
    def compute_atteunation_probability_projection(self, subset_idx: torch.tensor) -> torch.tensor:
        if subset_idx is not None:
            xyz1 = self.xyz1[self.subset_indices_array[subset_idx].cpu()].flatten(start_dim=0,end_dim=2)
            xyz2 = self.xyz2[self.subset_indices_array[subset_idx].cpu()].flatten(start_dim=0,end_dim=2)
        else:
            xyz1 = self.xyz1.flatten(start_dim=0,end_dim=2)
            xyz2 = self.xyz2.flatten(start_dim=0,end_dim=2)
        
        proj = torch.zeros(xyz1.shape[0]).to(pytomography.device)
        for idx_partial in torch.tensor_split(torch.arange(xyz1.shape[0]), self.N_splits):
            proj[idx_partial] += torch.exp(-parallelproj.joseph3d_fwd(
                xyz1[idx_partial].to(pytomography.device),
                xyz2[idx_partial].to(pytomography.device),
                self.attenuation_map[0].to(pytomography.device),
                self.object_origin,
                self.object_meta.dr
            ))
        N_angles = self.proj_meta.N_angles if subset_idx is None else self.subset_indices_array[subset_idx].shape[0]
        proj = proj.reshape((N_angles, *self.proj_meta.shape[1:]))
        return proj
    
    def compute_sensitivity_sinogram(self, subset_idx=None):
        if self.norm_sinogram is not None:
            sinogram_sensitivity = self.norm_sinogram
        else:
            sinogram_sensitivity = torch.ones(self.proj_meta.shape).to(pytomography.device)
        if subset_idx is not None:
            sinogram_sensitivity = self.get_projection_subset(sinogram_sensitivity, subset_idx)
        # Scale the weights by attenuation image if its provided in the system matrix
        if self.attenuation_map is not None:
            sinogram_sensitivity *= self.compute_atteunation_probability_projection(subset_idx)
        return sinogram_sensitivity
        
    def compute_normalization_factor(self, subset_idx=None):
        return self.backward(self.compute_sensitivity_sinogram(subset_idx), subset_idx)
    
    def forward(
        self,
        object: torch.tensor,
        subset_idx: int = None,
        scale_by_sensitivity = False
    ) -> torch.tensor:
        # Apply object space transforms
        object = object.to(pytomography.device)
        for transform in self.obj2obj_transforms:
            object = transform.forward(object)
        # Project
        if subset_idx is not None:
            xyz1 = self.xyz1[self.subset_indices_array[subset_idx].cpu()].flatten(start_dim=0,end_dim=2)
            xyz2 = self.xyz2[self.subset_indices_array[subset_idx].cpu()].flatten(start_dim=0,end_dim=2)
        else:
            xyz1 = self.xyz1.flatten(start_dim=0,end_dim=2)
            xyz2 = self.xyz2.flatten(start_dim=0,end_dim=2)
        proj = torch.zeros(xyz1.shape[0]).to(pytomography.device)
        for idx_partial in torch.tensor_split(torch.arange(xyz1.shape[0]), self.N_splits):
            proj[idx_partial] += parallelproj.joseph3d_fwd(
                xyz1[idx_partial].to(pytomography.device),
                xyz2[idx_partial].to(pytomography.device),
                object[0].to(pytomography.device),
                self.object_origin,
                self.object_meta.dr
            )
        N_angles = self.proj_meta.N_angles if subset_idx is None else self.subset_indices_array[subset_idx].shape[0]
        proj = proj.reshape((N_angles, *self.proj_meta.shape[1:]))
        if scale_by_sensitivity:
            proj = proj * self.compute_sensitivity_sinogram(subset_idx)
        return proj
    
    def backward(
        self,
        proj: torch.tensor,
        subset_idx: list[int] = None,
        scale_by_sensitivity = False
    ) -> torch.tensor:
        # sensitivity scaling
        if scale_by_sensitivity:
            proj = proj * self.compute_sensitivity_sinogram(subset_idx)
        # Project
        if subset_idx is not None:
            xyz1 = self.xyz1[self.subset_indices_array[subset_idx].cpu()].flatten(start_dim=0,end_dim=2)
            xyz2 = self.xyz2[self.subset_indices_array[subset_idx].cpu()].flatten(start_dim=0,end_dim=2)
        else:
            xyz1 = self.xyz1.flatten(start_dim=0,end_dim=2)
            xyz2 = self.xyz2.flatten(start_dim=0,end_dim=2)
        BP = 0
        for idx_partial in torch.tensor_split(torch.arange(xyz1.shape[0]), self.N_splits):
            BP += parallelproj.joseph3d_back(
                xyz1[idx_partial].to(pytomography.device),
                xyz2[idx_partial].to(pytomography.device),
                self.object_meta.shape,
                self.object_origin,
                self.object_meta.dr,
                proj.flatten()[idx_partial], # flattens to planes,r,theta
            ).unsqueeze(0)
        # Apply object transforms
        for transform in self.obj2obj_transforms[::-1]:
            BP  = transform.backward(BP)
        return BP