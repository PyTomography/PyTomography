from __future__ import annotations
import torch
import pytomography
from pytomography.transforms import Transform
from pytomography.transforms.shared import RotationTransform
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.utils import pad_object, unpad_object, pad_proj, unpad_proj, rotate_detector_z
import numpy as np
from ..system_matrix import SystemMatrix
try:
    import parallelproj
except:
    pass
    #Warning('parallelproj not installed. The SPECTCompleteSystemMatrix class requires parallelproj to be installed.')

class SPECTSystemMatrix(SystemMatrix):
    r"""System matrix for SPECT imaging implemented using the rotate+sum technique.
    
    Args:
            obj2obj_transforms (Sequence[Transform]): Sequence of object mappings that occur before forward projection.
            proj2proj_transforms (Sequence[Transform]): Sequence of proj mappings that occur after forward projection.
            object_meta (SPECTObjectMeta): SPECT Object metadata.
            proj_meta (SPECTProjMeta): SPECT projection metadata.
            object_initial_based_on_camera_path (bool): Whether or not to initialize the object estimate based on the camera path; this sets voxels to zero that are outside the SPECT camera path. Defaults to False.
    """
    def __init__(
        self,
        obj2obj_transforms: list[Transform],
        proj2proj_transforms: list[Transform],
        object_meta: SPECTObjectMeta,
        proj_meta: SPECTProjMeta,
        object_initial_based_on_camera_path: bool = False
    ) -> None:
        super(SPECTSystemMatrix, self).__init__(object_meta, proj_meta, obj2obj_transforms, proj2proj_transforms)
        self.object_initial_based_on_camera_path = object_initial_based_on_camera_path
        self.rotation_transform = RotationTransform()
        
    def _get_object_initial(self, device=None):
        """Returns an initial object estimate used in reconstruction algorithms. By default, this is a tensor of ones with the same shape as the object metadata.

        Returns:
            torch.Tensor: Initial object used in reconstruction algorithm.
        """
        object_initial = torch.ones(self.object_meta.shape).to(device)
        if self.object_initial_based_on_camera_path:
            for i in range(len(self.proj_meta.angles)):
                cutoff_idx = int(np.ceil(self.object_meta.shape[0]/ 2 - self.proj_meta.radii[i]/self.object_meta.dr[0]))
                if cutoff_idx<0:
                    continue
                img_cutoff = torch.ones(self.object_meta.shape).to(device)
                img_cutoff[:cutoff_idx, :, :] = 0
                img_cutoff = pad_object(img_cutoff, mode='replicate')
                img_cutoff = rotate_detector_z(img_cutoff, -self.proj_meta.angles[i])
                img_cutoff = unpad_object(img_cutoff)
                object_initial *= img_cutoff
        return object_initial
    
    def set_n_subsets(
        self,
        n_subsets: int
    ) -> list:
        """Sets the subsets for this system matrix given ``n_subsets`` total subsets.
        
        Args:
            n_subsets (int): number of subsets used in OSEM 
        """
        indices = torch.arange(self.proj_meta.shape[0]).to(torch.long).to(pytomography.device)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        self.subset_indices_array = subset_indices_array
        
    def get_projection_subset(
        self,
        projections: torch.tensor,
        subset_idx: int
    ) -> torch.tensor: 
        """Gets the subset of projections :math:`g_m` corresponding to index :math:`m`.

        Args:
            projections (torch.tensor): full projections :math:`g`
            subset_idx (int): subset index :math:`m`

        Returns:
            torch.tensor: subsampled projections :math:`g_m`
        """
        return projections[...,self.subset_indices_array[subset_idx],:,:]
    
    def get_weighting_subset(
        self,
        subset_idx: int
    ) -> float:
        r"""Computes the relative weighting of a given subset (given that the projection space is reduced). This is used for scaling parameters relative to :math:`H_m^T 1` in reconstruction algorithms, such as prior weighting :math:`\beta`

        Args:
            subset_idx (int): Subset index

        Returns:
            float: Weighting for the subset.
        """
        if subset_idx is None:
            return 1
        else:
            return len(self.subset_indices_array[subset_idx]) / self.proj_meta.num_projections

    def compute_normalization_factor(self, subset_idx : int | None = None) -> torch.tensor:
        """Function used to get normalization factor :math:`H^T_m 1` corresponding to projection subset :math:`m`.

        Args:
            subset_idx (int | None, optional): Index of subset. If none, then considers all projections. Defaults to None.

        Returns:
            torch.Tensor: normalization factor :math:`H^T_m 1`
        """
        norm_proj = torch.ones(self.proj_meta.shape).to(pytomography.device)
        if subset_idx is not None:
            norm_proj = self.get_projection_subset(norm_proj, subset_idx)
        return self.backward(norm_proj, subset_idx)
    
    def forward(
        self,
        object: torch.tensor,
        subset_idx: int | None = None,
    ) -> torch.tensor:
        r"""Applies forward projection to ``object`` for a SPECT imaging system.

        Args:
            object (torch.tensor[Lx, Ly, Lz]): The object to be forward projected
            subset_idx (int, optional): Only uses a subset of angles :math:`g_m` corresponding to the provided subset index :math:`m`. If None, then defaults to the full projections :math:`g`.

        Returns:
            torch.tensor: forward projection estimate :math:`g_m=H_mf`
        """
        # Deal with subset stuff
        if subset_idx is not None:
            angle_subset = self.subset_indices_array[subset_idx]
        N_angles = self.proj_meta.num_projections if subset_idx is None else len(angle_subset)
        angle_indices = torch.arange(N_angles).to(pytomography.device) if subset_idx is None else angle_subset
        # Start projection
        object = object.to(pytomography.device)
        proj = torch.zeros(
            (N_angles,*self.proj_meta.padded_shape[1:])
            ).to(pytomography.device)
        # Loop through all angles (or groups of angles in parallel)
        for i in range(0, len(angle_indices)):
            angle_indices_i = angle_indices[i]
            # Format Object
            object_i = pad_object(object)
            # beta = 270 - phi, and backward transform called because projection should be at +beta (requires inverse rotation of object)
            object_i = self.rotation_transform.backward(object_i, 270-self.proj_meta.angles[angle_indices_i])
            # Apply object 2 object transforms
            for transform in self.obj2obj_transforms:
                object_i = transform.forward(object_i, angle_indices_i)
            proj[i] = object_i.sum(axis=0)
        for transform in self.proj2proj_transforms:
            proj = transform.forward(proj)
        return unpad_proj(proj)
    
    def backward(
        self,
        proj: torch.tensor,
        subset_idx: int | None = None,
    ) -> torch.tensor:
        r"""Applies back projection to ``proj`` for a SPECT imaging system.

        Args:
            proj (torch.tensor): projections :math:`g` which are to be back projected
            subset_idx (int, optional): Only uses a subset of angles :math:`g_m` corresponding to the provided subset index :math:`m`. If None, then defaults to the full projections :math:`g`.
            return_norm_constant (bool): Whether or not to return :math:`H_m^T 1` along with back projection. Defaults to 'False'.

        Returns:
            torch.tensor: the object :math:`\hat{f} = H_m^T g_m` obtained via back projection.
        """
        # Deal with subset stuff
        if subset_idx is not None:
            angle_subset = self.subset_indices_array[subset_idx]
        N_angles = self.proj_meta.num_projections if subset_idx is None else len(angle_subset)
        angle_indices = torch.arange(N_angles).to(pytomography.device) if subset_idx is None else angle_subset
        # Box used to perform back projection
        boundary_box_bp = pad_object(torch.ones(self.object_meta.shape).to(pytomography.device), mode='back_project')
        # Pad proj and norm_proj (norm_proj used to compute sum_j H_ij)
        proj = pad_proj(proj)
        # First apply proj transforms before back projecting
        for transform in self.proj2proj_transforms[::-1]:
            proj = transform.backward(proj)
        # Setup for back projection
        object = torch.zeros(self.object_meta.padded_shape).to(pytomography.device)
        for i in range(0, len(angle_indices)):
            angle_indices_i = angle_indices[i]
            # Perform back projection
            object_i = proj[i].unsqueeze(0) * boundary_box_bp
            # Apply object mappings
            for transform in self.obj2obj_transforms[::-1]:
                object_i  = transform.backward(object_i, angle_indices_i)
            # Rotate all objects by by their respective angle
            object_i = self.rotation_transform.forward(object_i, 270-self.proj_meta.angles[angle_indices_i])
            # Add to total 
            object += object_i
        # Unpad
        object = unpad_object(object)
        return object
        
class SPECTCompleteSystemMatrix(SPECTSystemMatrix):
    """Class presently under construction. 
    """
    def __init__(
        self,
        object_meta,
        proj_meta,
        attenuation_map,
        psf_kernel,
        store_system_matrix = None,
        object_mask = None,
        projections_mask = None,
    ) -> None:
        super(SPECTCompleteSystemMatrix, self).__init__([],[], object_meta, proj_meta)
        self.dimension_single_proj = (*self.proj_meta.shape, *self.object_meta.shape)
        self.attenuation_map = attenuation_map
        self.psf_kernel = psf_kernel
        self.psf_kernel._configure(object_meta)
        self.X_obj = self._get_object_positions()
        self.system_matrices = None
        if store_system_matrix is not None:
            self.system_matrix_device = store_system_matrix
        else:
            self.system_matrix_device = pytomography.device
        self.system_matrix_dtype = torch.float16
        self.origin_amap = -(torch.tensor(object_meta.shape).to(pytomography.device)/2-0.5) * torch.tensor(object_meta.dr).to(pytomography.dtype).to(pytomography.device)
        self.voxel_size_amap = torch.tensor(object_meta.dr).to(pytomography.dtype).to(pytomography.device)
        if projections_mask is not None:
            self.projections_mask = projections_mask
        else:
            self.projections_mask = torch.ones(self.proj_meta.shape).to(pytomography.device).to(torch.bool)
        self.valid_proj_pixel_mask = self.projections_mask.reshape(self.proj_meta.num_projections,-1)
        if object_mask is not None:
            self.object_mask = object_mask
            self.valid_obj_voxel_mask= object_mask.ravel()
        else:
            self.object_mask = torch.ones(self.object_meta.shape).to(pytomography.device).to(torch.bool)
        self.valid_obj_voxel_mask = self.object_mask.ravel()
        if store_system_matrix is not None:
            print(self._compute_system_matrix_components(0).to(torch.float16).to(self.system_matrix_device).shape)
            self.system_matrices = [self._compute_system_matrix_components(i).to(torch.float16).to(self.system_matrix_device) for i in range(self.proj_meta.num_projections)]
            
    def _get_object_initial(self, device=None):
        return self.object_mask.to(pytomography.dtype)
        
    def _get_proj_positions(self, idx):
        Ny = self.proj_meta.shape[1]
        Nz = self.proj_meta.shape[2]
        dy = self.proj_meta.dr[0]
        dz = self.proj_meta.dr[1]
        angle = (270-self.proj_meta.angles[idx]) * torch.pi / 180
        radius = self.proj_meta.radii[idx]
        yv, zv = torch.meshgrid(torch.arange(-Ny/2+0.5, Ny/2+0.5, 1)*dy, torch.arange(-Nz/2+0.5, Nz/2+0.5, 1)*dz, indexing='ij')
        xv = torch.ones(yv.shape) * radius
        X = torch.stack([xv,yv,zv], dim=-1).to(pytomography.device)
        rotation_matrix = torch.tensor([
                [torch.cos(angle), -torch.sin(angle), 0],
                [torch.sin(angle), torch.cos(angle), 0],
                [0, 0, 1]
            ]).to(pytomography.device)
        return torch.flatten(torch.matmul(rotation_matrix.unsqueeze(0).unsqueeze(0), X.unsqueeze(-1)).squeeze(), end_dim=-2)

    def _get_object_positions(self):
        Nx, Ny, Nz = self.object_meta.shape
        dx, dy, dz = self.object_meta.dr
        xv, yv, zv = torch.meshgrid(
            [torch.arange(-Nx/2+0.5, Nx/2+0.5, 1)*dx, torch.arange(-Ny/2+0.5, Ny/2+0.5, 1)*dy, torch.arange(-Nz/2+0.5, Nz/2+0.5, 1)*dz], indexing='ij')
        X = torch.stack([xv,yv,zv], dim=-1).to(pytomography.device)
        return torch.flatten(X, end_dim=-2)
        
    def _compute_system_matrix_components(self, idx):
        if self.system_matrices is not None:
            return self.system_matrices[idx].to(self.system_matrix_dtype)
        valid_proj_idx = self.valid_proj_pixel_mask[idx]
        valid_obj_idx = self.valid_obj_voxel_mask
        X_proj = self._get_proj_positions(idx)[valid_proj_idx]
        X_obj = self.X_obj[valid_obj_idx]
        # Assume 0 contribution outside of valid regions (requires cropping object initial and projection data)
        system_matrix_proj_i = torch.einsum(
            'i,j->ij',
            torch.ones(valid_proj_idx.sum()).to(pytomography.device),
            torch.ones(valid_obj_idx.sum()).to(pytomography.device)
        )
        angle = (270-self.proj_meta.angles[idx]) * torch.pi / 180
        N_splits = 64
        # PSF
        for X_proj_sub, indices in zip(torch.tensor_split(X_proj, N_splits), torch.tensor_split(torch.arange(X_proj.shape[0]).to(pytomography.device), N_splits)):
            delta_r = (X_proj_sub[:,None] - X_obj)
            d = torch.abs(delta_r[:,:,0]*torch.cos(angle) + delta_r[:,:,1]*torch.sin(angle))
            x = delta_r[:,:,1]*torch.cos(angle) - delta_r[:,:,0]*torch.sin(angle)
            y = delta_r[:,:,2]
            system_matrix_proj_i[indices] *= self.psf_kernel(x.ravel(),y.ravel(),d.ravel()).reshape(x.shape)
        # Attenuation
        for X_proj_sub, indices in zip(torch.tensor_split(X_proj, N_splits), torch.tensor_split(torch.arange(X_proj.shape[0]).to(pytomography.device), N_splits)):
            X_start = X_obj[None].repeat((X_proj_sub.shape[0],1,1))
            X_end = X_proj_sub[:,None].repeat((1,X_obj.shape[0],1))
            system_matrix_proj_i[indices] *= torch.exp(-parallelproj.joseph3d_fwd(
                torch.flatten(X_start, end_dim=-2),
                torch.flatten(X_end, end_dim=-2),
                self.attenuation_map,
                self.origin_amap,
                self.voxel_size_amap # TODO: adjust for CT,
            )).reshape(X_proj_sub.shape[0], X_obj.shape[0])
        return system_matrix_proj_i.to(torch.float16)
        
    def compute_normalization_factor(self, subset_idx : int | None = None) -> torch.tensor:
        norm_proj = self.projections_mask.to(self.system_matrix_dtype)
        if subset_idx is not None:
            norm_proj = norm_proj[self.subset_indices_array[subset_idx]]
        return self.backward(norm_proj, subset_idx)
    
    def forward(
        self,
        object,
        subset_idx: int | None = None
    ):
        if subset_idx is not None:
            angle_subset = self.subset_indices_array[subset_idx]
        N_angles = self.proj_meta.num_projections if subset_idx is None else len(angle_subset)
        angle_indices = torch.arange(N_angles) if subset_idx is None else angle_subset
        proj = torch.zeros(
            (N_angles,self.proj_meta.shape[1]*self.proj_meta.shape[2])
            ).to(pytomography.device)
        for i, idx in enumerate(angle_indices):
            system_matrix_proj_i = self._compute_system_matrix_components(idx)
            proj[i,self.valid_proj_pixel_mask[idx]] = torch.einsum(
                'ij,j->i',
                system_matrix_proj_i,
                object.ravel()[self.valid_obj_voxel_mask].to(self.system_matrix_device).to(self.system_matrix_dtype)
            ).to(pytomography.device).to(pytomography.dtype)
        return proj.reshape(N_angles,self.proj_meta.shape[1],self.proj_meta.shape[2])
    
    def backward(
        self,
        proj,
        subset_idx: int | None = None
    ):
        if subset_idx is not None:
            angle_subset = self.subset_indices_array[subset_idx]
        N_angles = self.proj_meta.num_projections if subset_idx is None else len(angle_subset)
        angle_indices = torch.arange(N_angles) if subset_idx is None else angle_subset
        object = torch.flatten(torch.zeros(*self.object_meta.shape).to(pytomography.device))
        for i, idx in enumerate(angle_indices):
            system_matrix_proj_i = self._compute_system_matrix_components(idx)
            object[self.valid_obj_voxel_mask] += torch.einsum(
                'ij,i->j',
                system_matrix_proj_i,
                proj.flatten(start_dim=1)[i,self.valid_proj_pixel_mask[idx]].to(self.system_matrix_device).to(self.system_matrix_dtype)
            ).to(pytomography.device).to(pytomography.dtype)
        return object.reshape(self.object_meta.shape)