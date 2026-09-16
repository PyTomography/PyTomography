from __future__ import annotations
import torch
import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETSinogramPolygonProjMeta
import numpy as np
from pytomography.projectors import SystemMatrix
from pytomography.transforms import Transform
from pytomography.io.PET.shared import listmode_to_sinogram
import parallelproj

class PETSinogramSystemMatrix(SystemMatrix):
    r"""System matrix for sinogram-based PET reconstruction. 

        Args:
            object_meta (ObjectMeta): Metadata of object space, containing information on voxel size and dimensions.
            proj_meta (PETSinogramPolygonProjMeta): PET sinogram projection space metadata. This information contains the scanner lookup table and time-of-flight metadata. 
            obj2obj_transforms (list[Transform], optional): Object to object space transforms applied before forward projection and after back projection. These are typically used for PSF modeling in PET imaging. Defaults to [].
            attenuation_map (torch.tensor | None, optional): Attenuation map used for attenuation modeling. If provided, all weights will be scaled by detection probabilities derived from this map. Note that this scales on top of ``sinogram_sensitivity``, so if attenuation is already accounted for there, this is not needed. Defaults to None.
            sinogram_sensitivity (torch.tensor | None, optional): Normalization sinogram used to scale projections after forward projection. This factor may include detector normalization :math:`\eta` and/or attenuation modeling :math:`\mu`. The attenuation modeling :math:`\mu` should not be included if ``attenuation_map`` is provided as an argument to the function. Defaults to None.
            scale_projection_by_sensitivity (bool, optional): Whether or not to scale the projections by :math:`\mu \eta`. This is not needed in reconstruction algorithms using a PoissonLogLikelihood. Defaults to False.
            N_splits (int, optional): Splits up computation of forward/back projection to save GPU memory. Defaults to 1.
            device (str, optional): The device for any objects in projection space projection space (what it outputs in forward projection and what it expects for back projection). This is seperate from ``pytomography.device`` since the internal functionality may still use GPU even if this is CPU. This is used to save GPU memory since sinograms are often very large. Defaults to pytomography.device.
        """
    def __init__(
        self,
        object_meta: ObjectMeta,
        proj_meta: PETSinogramPolygonProjMeta,
        obj2obj_transforms: list[Transform] = [],
        attenuation_map: torch.tensor | None = None,
        sinogram_sensitivity: torch.tensor | None = None,
        scale_projection_by_sensitivity: bool = False,
        N_splits: int = 1,
        device: str = pytomography.device,
    ) -> None:
        super(PETSinogramSystemMatrix, self).__init__(
            obj2obj_transforms=obj2obj_transforms,
            proj2proj_transforms=[],
            object_meta=object_meta,
            proj_meta=proj_meta
            )
        self.output_device = device
        self.object_origin = (- np.array(object_meta.shape) / 2 + 0.5) * (np.array(object_meta.dr))
        self.obj2obj_transforms = obj2obj_transforms
        self.proj_meta = proj_meta
        # In case they get put on another device
        self.attenuation_map = attenuation_map
        self.sinogram_sensitivity = sinogram_sensitivity
        self.scale_projection_by_sensitivity = scale_projection_by_sensitivity
        if sinogram_sensitivity is not None:
            self.sinogram_sensitivity = self.sinogram_sensitivity.to(self.output_device)
        self.N_splits = N_splits
        self.TOF = self.proj_meta.tof_meta is not None
    
    def _get_xyz_sinogram_coordinates(self, subset_idx: int = None):
        """Get the XYZ coordinates corresponding to the pair of crystals of the projection angle

        Args:
            subset_idx (int, optional): Subset index for ths sinogram. If None, considers all elements. Defaults to None.

        Returns:
            Sequence[torch.Tensor, torch.Tensor]: XYZ coordinates of crystal 1 and XYZ coordinates of crystal 2 corresponding to all elements in the sinogram.
        """
        if subset_idx is not None:
            idx = self.subset_indices_array[subset_idx].cpu()
            detector_coordinates = self.proj_meta.detector_coordinates[idx]
            N_angles = idx.shape[0]
        else:
            detector_coordinates = self.proj_meta.detector_coordinates
            N_angles = self.proj_meta.shape[0]
        xy1 = torch.flatten(detector_coordinates, start_dim=0, end_dim=1)[:,0].cpu()
        xy2 = torch.flatten(detector_coordinates, start_dim=0, end_dim=1)[:,1].cpu()
        z1, z2 = self.proj_meta.ring_coordinates.T.cpu()
        xyz1 = torch.concatenate([
            xy1.unsqueeze(1).repeat(1,z1.shape[0],1),
            z1.unsqueeze(0).unsqueeze(-1).repeat(xy1.shape[0],1,1)
        ], dim=-1).flatten(start_dim=0,end_dim=1)
        xyz2 = torch.concatenate([
            xy2.unsqueeze(1).repeat(1,z2.shape[0],1),
            z2.unsqueeze(0).unsqueeze(-1).repeat(xy2.shape[0],1,1)
        ], dim=-1).flatten(start_dim=0,end_dim=1)
        xyz1 = xyz1.reshape((N_angles, *self.proj_meta.shape[1:], 3))
        xyz2 = xyz2.reshape((N_angles, *self.proj_meta.shape[1:], 3))
        return xyz1.flatten(start_dim=0,end_dim=2), xyz2.flatten(start_dim=0,end_dim=2)
    
    def _compute_atteunation_probability_projection(self, subset_idx: torch.tensor) -> torch.tensor:
        """Compute the probability of a photon not being attenuated for a certain sinogram element.

        Args:
            subset_idx (torch.tensor): Subset index for ths sinogram.

        Returns:
            torch.tensor: Probability sinogram
        """
        xyz1, xyz2 = self._get_xyz_sinogram_coordinates(subset_idx=subset_idx)
        proj = torch.zeros(xyz1.shape[0]).to(self.output_device)
        for idx_partial in torch.tensor_split(torch.arange(xyz1.shape[0]), self.N_splits):
            proj[idx_partial] += torch.exp(-parallelproj.joseph3d_fwd(
                xyz1[idx_partial].to(pytomography.device),
                xyz2[idx_partial].to(pytomography.device),
                self.attenuation_map.to(pytomography.device),
                self.object_origin,
                self.object_meta.dr
            )).to(self.output_device)
        N_angles = self.proj_meta.N_angles if subset_idx is None else self.subset_indices_array[subset_idx].shape[0]
        proj = proj.reshape((N_angles, *self.proj_meta.shape[1:]))
        return proj
    
    def _compute_sensitivity_sinogram(self, subset_idx: int = None):
        r"""Computes the sensitivity sinogram :math:`\mu \eta` that accounts for attenuation effects and normalization effects.

        Args:
            subset_idx (int, optional): Subset index for ths sinogram. If None, considers all elements. Defaults to None..

        Returns:
            torch.Tensor: Sensitivity sinogram.
        """
        if self.sinogram_sensitivity is not None:
            sinogram_sensitivity = self.sinogram_sensitivity
        else:
            sinogram_sensitivity = torch.ones(self.proj_meta.shape).to(self.output_device)
        if subset_idx is not None:
            sinogram_sensitivity = self.get_projection_subset(sinogram_sensitivity, subset_idx)
        # Scale the weights by attenuation image if its provided in the system matrix
        if self.attenuation_map is not None:
            sinogram_sensitivity = sinogram_sensitivity * self._compute_atteunation_probability_projection(subset_idx)
        if self.TOF:
            sinogram_sensitivity = sinogram_sensitivity.unsqueeze(-1)
        return sinogram_sensitivity
    
    def set_n_subsets(self, n_subsets: int) -> list:
        """Returns a list where each element consists of an array of indices corresponding to a partitioned version of the projections. 

        Args:
            n_subsets (int): Number of subsets to partition the projections into

        Returns:
            list: List of arrays where each array corresponds to the projection indices of a particular subset.
        """
        indices = torch.arange(self.proj_meta.N_angles).to(torch.long).to(self.output_device)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        self.subset_indices_array = subset_indices_array
        
    def get_projection_subset(self, projections: torch.Tensor, subset_idx: int | None) -> torch.tensor:
        """Obtains subsampled projections :math:`g_m` corresponding to subset index :math:`m`. Sinogram PET partitions projections based on angle.

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
        r"""Computes the normalization factor :math:`H^T \mu \eta`

        Args:
            subset_idx (int, optional): Subset index for ths sinogram. If None, considers all elements. Defaults to None..

        Returns:
            torch.Tensor: Normalization factor.
        """
        return self.backward(1, subset_idx, force_nonTOF=True, force_scale_by_sensitivity=True)
    
    def forward(
        self,
        object: torch.tensor,
        subset_idx: int = None,
    ) -> torch.tensor:
        r"""PET Sinogram forward projection

        Args:
            object (torch.tensor): Object to be forward projected
            subset_idx (int, optional): Subset index for ths sinogram. If None, considers all elements. Defaults to None.
            scale_by_sensitivity (bool, optional): Whether or not to scale the projections by :math:`\mu \eta`. This is not necessarily needed in reconstruction algorithms. Defaults to False.

        Returns:
            torch.tensor: Forward projection
        """
        # Apply object space transforms
        object = object.to(pytomography.device)
        for transform in self.obj2obj_transforms:
            object = transform.forward(object)
        # Project
        xyz1, xyz2 = self._get_xyz_sinogram_coordinates(subset_idx=subset_idx)
        if self.TOF:
            proj = torch.zeros((xyz1.shape[0], self.proj_meta.tof_meta.num_bins)).to(pytomography.dtype).to(self.output_device)
        else:
            proj = torch.zeros((xyz1.shape[0])).to(pytomography.dtype).to(self.output_device)
        for idx_partial in torch.tensor_split(torch.arange(xyz1.shape[0]), self.N_splits):
            if self.TOF:
                proj[idx_partial] += parallelproj.joseph3d_fwd_tof_sino(
                    xyz1[idx_partial].to(pytomography.device),
                    xyz2[idx_partial].to(pytomography.device),
                    object.to(pytomography.device),
                    self.object_origin,
                    self.object_meta.dr,
                    self.proj_meta.tof_meta.bin_width,
                    self.proj_meta.tof_meta.sigma,
                    self.proj_meta.tof_meta.center_offset,
                    self.proj_meta.tof_meta.n_sigmas,
                    self.proj_meta.tof_meta.num_bins
                ).to(self.output_device)
            else:
                proj[idx_partial] += parallelproj.joseph3d_fwd(
                    xyz1[idx_partial].to(pytomography.device),
                    xyz2[idx_partial].to(pytomography.device),
                    object.to(pytomography.device),
                    self.object_origin,
                    self.object_meta.dr
                ).to(self.output_device)
        N_angles = self.proj_meta.N_angles if subset_idx is None else self.subset_indices_array[subset_idx].shape[0]
        proj = proj.reshape((N_angles, *self.proj_meta.shape[1:], -1))
        if self.scale_projection_by_sensitivity:
            proj = proj * self._compute_sensitivity_sinogram(subset_idx)
        proj = proj.squeeze() # will remove first dim if nonTOF
        return proj
    
    def backward(
        self,
        proj: torch.tensor,
        subset_idx: int = None,
        force_scale_by_sensitivity = False,
        force_nonTOF = False,
    ) -> torch.tensor:
        """PET Sinogram back projection

        Args:
            proj (torch.tensor): Sinogram to be back projected
            subset_idx (int, optional): Subset index for ths sinogram. If None, considers all elements. Defaults to None.
            scale_by_sensitivity (bool, optional): Whether or not to scale the projections by :math:`\mu \eta`. This is not necessarily needed in reconstruction algorithms. Defaults to False.
            force_nonTOF (bool, optional): Force non-TOF projection, even if TOF metadata is contained in the projection metadata. This is used for computing normalization factors (which don't depend on TOF). Defaults to False.

        Returns:
            torch.tensor: Back projection.
        """
        # sensitivity scaling
        if force_scale_by_sensitivity or self.scale_projection_by_sensitivity:
            proj = proj * self._compute_sensitivity_sinogram(subset_idx)
        # Project
        xyz1, xyz2 = self._get_xyz_sinogram_coordinates(subset_idx=subset_idx)
        BP = 0
        for idx_partial in torch.tensor_split(torch.arange(xyz1.shape[0]), self.N_splits):
            if self.TOF*(not force_nonTOF):
                BP += parallelproj.joseph3d_back_tof_sino(
                    xyz1[idx_partial].to(pytomography.device),
                    xyz2[idx_partial].to(pytomography.device),
                    self.object_meta.shape,
                    self.object_origin,
                    self.object_meta.dr,
                    proj.flatten(end_dim=-2)[idx_partial].to(pytomography.device), # flattens to planes,r,theta
                    self.proj_meta.tof_meta.bin_width,
                    self.proj_meta.tof_meta.sigma,
                    self.proj_meta.tof_meta.center_offset,
                    self.proj_meta.tof_meta.n_sigmas,
                    self.proj_meta.tof_meta.num_bins
                )
            else:
                BP += parallelproj.joseph3d_back(
                    xyz1[idx_partial].to(pytomography.device),
                    xyz2[idx_partial].to(pytomography.device),
                    self.object_meta.shape,
                    self.object_origin,
                    self.object_meta.dr,
                    proj.flatten()[idx_partial].to(pytomography.device), # flattens to planes,r,theta
                )
        # Apply object transforms
        for transform in self.obj2obj_transforms[::-1]:
            BP  = transform.backward(BP)
        return BP
    
def create_sinogramSM_from_LMSM(lm_system_matrix: SystemMatrix, device='cpu'):
    """Generates a sinogram system matrix from a listmode system matrix. This is used in the single scatter simulation algorithm.   

    Args:
        lm_system_matrix (SystemMatrix): A listmode PET system matrix
        device (str, optional): The device for any objects in projection space projection space (what it outputs in forward projection and what it expects for back projection). This is seperate from ``pytomography.device`` since the internal functionality may still use GPU even if this is CPU. This is used to save GPU memory since sinograms are often very large. Defaults to pytomography.device.

    Returns:
        SystemMatrix: PET sinogram system matrix generated via a corresponding PET listmode system matrix.
    """
    lm_proj_meta = lm_system_matrix.proj_meta
    sino_proj_meta = PETSinogramPolygonProjMeta(
        lm_proj_meta.info,
        lm_proj_meta.tof_meta
    )
    if lm_proj_meta.weights_sensitivity is not None:
        idxs = torch.arange(lm_proj_meta.scanner_lut.shape[0]).cpu()
        detector_ids_sensitivity = torch.combinations(idxs, 2)
        sinogram_sensitivity = listmode_to_sinogram(
            detector_ids_sensitivity,
            lm_proj_meta.info,
            lm_proj_meta.weights_sensitivity.cpu()
        )
    else:
        sinogram_sensitivity = None
    sino_system_matrix = PETSinogramSystemMatrix(
        lm_system_matrix.object_meta,
        sino_proj_meta,
        obj2obj_transforms = lm_system_matrix.obj2obj_transforms,
        N_splits=20,
        attenuation_map=lm_system_matrix.attenuation_map,
        sinogram_sensitivity=sinogram_sensitivity,
        device=device
    )
    return sino_system_matrix