from __future__ import annotations
import torch
import pytomography
from pytomography.transforms import Transform
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta
from pytomography.projectors import SystemMatrix
import numpy as np
import parallelproj

class PETLMSystemMatrix(SystemMatrix):
    r"""System matrix of PET list mode data. Forward projections corresponds to computing the expected counts along all LORs specified: in particular it approximates :math:`g_i = \int_{\text{LOR}_i} h(r) f(r) dr` where index :math:`i` corresponds to a particular detector pair and :math:`h(r)` is a Gaussian function that incorporates time-of-flight information (:math:`h(r)=1` for non-time-of-flight). The integral is approximated in the discrete object space using Joseph3D projections. In general, the system matrix implements two different projections, the quantity :math:`H` which projects to LORs corresponding to all detected events, and the quantity :math:`\tilde{H}` which projects to all valid LORs. The quantity :math:`H` is used for standard forward/back projection, while :math:`\tilde{H}` is used to compute the sensitivity image.

        Args:
            object_meta (SPECTObjectMeta): Metadata of object space, containing information on voxel size and dimensions.
            proj_meta (PETLMProjMeta): PET listmode projection space metadata. This information contains the detector ID pairs of all detected events, as well as a scanner lookup table and time-of-flight metadata. In addition, this metadata contains all information regarding event weights, typically corresponding to the effects of attenuation :math:`\mu` and :math:`\eta`. 
            obj2obj_transforms (Sequence[Transform]): Object to object space transforms applied before forward projection and after back projection. These are typically used for PSF modeling in PET imaging.
            attenuation_map (torch.tensor[float] | None, optional): Attenuation map used for attenuation modeling. If provided, all weights will be scaled by detection probabilities derived from this map. Note that this scales on top of any weights provided in ``proj_meta``, so if attenuation is already accounted for there, this is not needed. Defaults to None.
            scale_projection_by_sensitivity (bool, optional): Whether or not to scale the projections by :math:`\mu \eta`. This is not needed in reconstruction algorithms using a PoissonLogLikelihood. Defaults to False.
            N_splits (int): Splits up computation of forward/back projection to save GPU memory. Defaults to 1.
            device (str): The device on which forward/back projection tensors are output. This is seperate from ``pytomography.device``, which handles internal computations. The reason for having the option of a second device is that the projection space may be very large, and certain GPUs may not have enough memory to store the projections. If ``device`` is not the same as ``pytomography.device``, then one must also specify the same ``device`` in any reconstruction algorithm used. Defaults to ``pytomography.device``.

    """
    def __init__(
        self,
        object_meta: ObjectMeta,
        proj_meta: PETLMProjMeta,
        obj2obj_transforms: list[Transform] = [],
        attenuation_map: torch.tensor[float] | None = None,
        scale_projection_by_sensitivity: bool = False,
        N_splits: int = 1,
        FOV_scale_enabled: bool = True,
        device: str = pytomography.device,
    ) -> None:
        super(PETLMSystemMatrix, self).__init__(
            obj2obj_transforms=obj2obj_transforms,
            proj2proj_transforms=[],
            object_meta=object_meta,
            proj_meta=proj_meta
            )
        self.output_device = device
        if self.proj_meta.tof_meta is not None:
            self.TOF = True
        else:
            self.TOF = False
        self.object_origin = (- np.array(object_meta.shape) / 2 + 0.5) * (np.array(object_meta.dr))
        self.obj2obj_transforms = obj2obj_transforms
        # In case they get put on another device
        self.proj_meta.detector_ids = self.proj_meta.detector_ids.cpu()
        self.proj_meta.scanner_lut = self.proj_meta.scanner_lut.cpu()
        self.attenuation_map = attenuation_map
        self.N_splits = N_splits
        self.scale_projection_by_sensitivity = scale_projection_by_sensitivity
        self.norm_BP = self._backward_full()
        # replace zeros (outside FOV) with small value to avoid NaNs
        self.norm_BP[self.norm_BP < 1e-7] = 1e7
        self.FOV_scale_enabled = FOV_scale_enabled
        
    def _get_object_initial(self, device=pytomography.device):
        # Only consider the space within the FOV
        zmin = (self.object_meta.shape[-1]-1)/2 + self.proj_meta.scanner_lut[:,2].min() /self.object_meta.dr[-1]
        zmax = (self.object_meta.shape[-1]-1)/2 + self.proj_meta.scanner_lut[:,2].max() /self.object_meta.dr[-1]
        zmin = max(0, zmin)
        zmax = max(0,zmax)
        object_initial = torch.ones(self.object_meta.shape).to(device)
        object_initial[:,:,:int(np.ceil(zmin))] = 0
        object_initial[:,:,int(np.floor(zmax)):] = 0
        return object_initial
    
    def _get_prior_FOV_scale(self):
        """Sets scaling for the prior within the FOV.

        Returns:
            torch.Tensor: Prior scaling
        """
        if self.FOV_scale_enabled:
            zmin = (self.object_meta.shape[-1]-1)/2 + self.proj_meta.scanner_lut[:,2].min() /self.object_meta.dr[-1]
            zmax = (self.object_meta.shape[-1]-1)/2 + self.proj_meta.scanner_lut[:,2].max() /self.object_meta.dr[-1]
            zmid = (zmin + zmax) / 2
            zmin = max(0, zmin)
            zmax = max(0,zmax)
            # Set axial FOV scaling
            z = torch.arange(self.object_meta.shape[-1]).to(pytomography.device)
            FOV_scale = (zmid - torch.abs(z - zmid)) / zmid
            FOV_scale[FOV_scale<0] = 0
            FOV_scale = torch.ones(self.object_meta.shape).to(pytomography.device) * FOV_scale
        else:
            FOV_scale = torch.ones(self.object_meta.shape).to(pytomography.device)
        return FOV_scale
    
    def _compute_attenuation_probability_projection(self, idx: torch.tensor) -> torch.tensor:
        """Computes probabilities of photons being detected along an LORs corresponding to ``idx``.

        Args:
            idx (torch.tensor): Indices of the detector pairs.

        Returns:
            torch.Tensor: The probabilities of photons being detected along the detector pairs.
        """
        proj = torch.tensor([]).cpu()
        for idx_partial in torch.tensor_split(idx, self.N_splits):
            proj_i = torch.exp(-parallelproj.joseph3d_fwd(
                self.proj_meta.scanner_lut[idx_partial[:,0]].to(pytomography.device),
                self.proj_meta.scanner_lut[idx_partial[:,1]].to(pytomography.device),
                self.attenuation_map,
                self.object_origin,
                self.object_meta.dr,
                num_chunks=4)).cpu()
            proj = torch.concatenate([proj, proj_i])
        return proj.to(self.output_device)
    
    def _compute_sensitivity_projection(self, all_ids: bool = True) -> torch.Tensor:
        """Computes the sensitivty projection (when back projected, gives normalization factor)

        Args:
            all_ids (bool, optional): Compute for all detector IDs. Defaults to True.

        Returns:
            torch.Tensor: Sesitivity factor for detector IDs
        """
        # If detector_ids is None, use all detector ids
        if all_ids:
            if self.proj_meta.detector_ids_sensitivity is not None:
                detector_ids = self.proj_meta.detector_ids_sensitivity
            else:
                # Assumes all possible pairs are used
                idxs = torch.arange(self.proj_meta.scanner_lut.shape[0]).to(pytomography.device).to(torch.int32)
                detector_ids = torch.combinations(idxs, 2).cpu()
        else:
            detector_ids = self.proj_meta.detector_ids
        proj = torch.ones(detector_ids.shape[0])
        # Load normalization weights for the specific detector IDs
        if self.proj_meta.weights_sensitivity is not None:
            # If using all detector IDs (assumes weights_sensitivity is same shape as detector_ids)
            if all_ids:
                proj *= self.proj_meta.weights_sensitivity.cpu()
            # Otherwise need to grab norm factor specific to the detector_ids used
            else: # otherwise grab specific IDs (maybe move this somewhere else)
                ids_sorted, _ = torch.sort(detector_ids[:,:2], 1)
                norm_factor_idxs = ((self.proj_meta.info['NrCrystalsPerRing'] * self.proj_meta.info['NrRings']-1)*ids_sorted[:,0] + ids_sorted[:,1] - ids_sorted[:,0]*(ids_sorted[:,0]+1)/2 - 1).to(torch.int)
                proj *= self.proj_meta.weights_sensitivity.cpu()[norm_factor_idxs]    
        # Scale the weights by attenuation image if its provided in the system matrix
        if self.attenuation_map is not None:
            proj *= self._compute_attenuation_probability_projection(detector_ids).cpu()
        return proj
        
    def _backward_full(self, N_splits: int = 10):
        r"""Computes full back projection :math:`\tilde{H}^T w g` where :math:`w` is the weighting specified in the projection metadata that accounts for attenuation/normalization correction. If ``proj`` ($g$) is not provided, then uses a tensor of all ones (this is used to compute the normalization factor).

        Args:
            N_splits (int, optional): Optionally splits up computation to save memory on GPU. Defaults to 10.
        """
        proj = self._compute_sensitivity_projection()
        # All detector IDs
        if self.proj_meta.detector_ids_sensitivity is not None:
            detector_ids_sensitivity = self.proj_meta.detector_ids_sensitivity
        else:
            idxs = torch.arange(self.proj_meta.scanner_lut.shape[0]).to(pytomography.device).to(torch.int32)
            detector_ids_sensitivity = torch.combinations(idxs, 2).cpu()
        norm_BP = 0
        for proj_subset, detector_ids_sensitivity_subset in zip(torch.tensor_split(proj, N_splits), torch.tensor_split(detector_ids_sensitivity, N_splits)):
            # Add tensors to PyTomography device for fast projection
            norm_BP += parallelproj.joseph3d_back(
                self.proj_meta.scanner_lut[detector_ids_sensitivity_subset[:,0]].to(pytomography.device),
                self.proj_meta.scanner_lut[detector_ids_sensitivity_subset[:,1]].to(pytomography.device),
                self.object_meta.shape,
                self.object_origin,
                self.object_meta.dr,
                proj_subset.to(pytomography.device) + pytomography.delta,
                num_chunks=4)
        # Apply object transforms
        for transform in self.obj2obj_transforms[::-1]:
            norm_BP  = transform.backward(norm_BP)
        return norm_BP.cpu()
    
    def set_n_subsets(self, n_subsets: int) -> list:
        """Returns a list where each element consists of an array of indices corresponding to a partitioned version of the projections. 

        Args:
            n_subsets (int): Number of subsets to partition the projections into

        Returns:
            list: List of arrays where each array corresponds to the projection indices of a particular subset.
        """
        indices = torch.arange(self.proj_meta.detector_ids.shape[0]).to(torch.long).cpu()
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        self.subset_indices_array = subset_indices_array
        
    def get_projection_subset(self, projections: torch.Tensor, subset_idx: int) -> torch.tensor:
        """Obtains subsampled projections :math:`g_m` corresponding to subset index :math:`m`. For LM PET, its always the case that :math:`g_m=1`, but this function is still required for subsampling scatter :math:`s_m` as is required in certain reconstruction algorithms

        Args:
            projections (torch.Tensor): total projections :math:`g`
            subset_idx (int): subset index :math:`m`

        Returns:
            torch.Tensor: subsampled projections :math:`g_m`.
        """
        # Needs to consider cases where projection is simply a 1 element tensor in the numerator, but also cases of scatter where it is a longer tensor
        
        if (projections.shape[0]>1)*(subset_idx is not None):
            subset_indices = self.subset_indices_array[subset_idx]
            proj_subset = projections[subset_indices]
        else:
            proj_subset = projections
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
            return len(self.subset_indices_array[subset_idx]) / self.proj_meta.detector_ids.shape[0]

    def compute_normalization_factor(self, subset_idx: int | None = None) -> torch.tensor:
        r"""Function called by reconstruction algorithms to get the sensitivty image :math:`\tilde{H}_m^T w`.

        Args:
            subset_idx (int | None, optional): Subset index :math:`m`. If none, then considers backprojection over all subsets. Defaults to None.

        Returns:
            torch.tensor: Normalization factor.
        """
        
        if subset_idx is None:
            fraction_considered = 1
        else:
            fraction_considered = self.subset_indices_array[subset_idx].shape[0] / self.proj_meta.detector_ids.shape[0] 
        return fraction_considered * self.norm_BP.to(self.output_device)
    
    def forward(
        self,
        object: torch.tensor,
        subset_idx: int = None,
    ) -> torch.tensor:
        """Computes forward projection. In the case of list mode PET, this corresponds to the expected number of detected counts along each LOR corresponding to a particular object.

        Args:
            object (torch.tensor): Object to be forward projected
            subset_idx (int, optional): Subset index :math:`m` of the projection. If None, then assumes projection to the entire projection space. Defaults to None.

        Returns:
            torch.tensor: Projections corresponding to the expected number of counts along each LOR.
        """ 
        # Deal With subset stuff
        if subset_idx is not None:
            idx = self.proj_meta.detector_ids[self.subset_indices_array[subset_idx].cpu()].squeeze()
        else:
            idx = self.proj_meta.detector_ids.squeeze()
        # Apply object space transforms
        object = object.to(pytomography.device)
        for transform in self.obj2obj_transforms:
            object = transform.forward(object)
        # Project
        proj = torch.tensor([]).cpu()
        for idx_partial in torch.tensor_split(idx, self.N_splits):
            if self.TOF:
                proj_i = parallelproj.joseph3d_fwd_tof_lm(
                    self.proj_meta.scanner_lut[idx_partial[:,0].to(torch.int)].to(pytomography.device),
                    self.proj_meta.scanner_lut[idx_partial[:,1].to(torch.int)].to(pytomography.device),
                    object,
                    self.object_origin,
                    self.object_meta.dr,
                    self.proj_meta.tof_meta.bin_width,
                    self.proj_meta.tof_meta.sigma,
                    self.proj_meta.tof_meta.center_offset,
                    self.proj_meta.tof_meta.n_sigmas,
                    idx_partial[:,2].squeeze().to(pytomography.device) - (self.proj_meta.tof_meta.num_bins - 1) // 2
                    )
            else:
                proj_i = parallelproj.joseph3d_fwd(
                    self.proj_meta.scanner_lut[idx_partial[:,0].to(torch.int)].to(pytomography.device),
                    self.proj_meta.scanner_lut[idx_partial[:,1].to(torch.int)].to(pytomography.device),
                    object,
                    self.object_origin,
                    self.object_meta.dr
                    )
            proj = torch.concatenate([proj, proj_i.cpu()])
            
        if self.scale_projection_by_sensitivity:
            if self.proj_meta.weights is None:
                raise Exception('If scaling by sensitivity, then `weights` must be provided in the projection metadata')
            else:
                proj = proj * self.get_projection_subset(self.proj_meta.weights, subset_idx).to(proj.device)
        return proj.to(self.output_device)
            
    def backward(
        self,
        proj: torch.tensor,
        subset_idx: list[int] = None,
        return_norm_constant: bool = False,
    ) -> torch.tensor:
        """Computes back projection. This corresponds to tracing a sequence of LORs into object space.

        Args:
            proj (torch.tensor): Projections to be back projected
            subset_idx (int, optional): Subset index :math:`m` of the projection. If None, then assumes projection to the entire projection space. Defaults to None.
            return_norm_constant (bool, optional): Whether or not to return the normalization constant: useful in reconstruction algorithms that require :math:`H_m^T 1`. Defaults to False.

        Returns:
            torch.tensor: _description_
        """
        # Deal With subset stuff
        if subset_idx is not None:
            idx = self.proj_meta.detector_ids[self.subset_indices_array[subset_idx].cpu()].squeeze()
        else:
            idx = self.proj_meta.detector_ids.squeeze()
        # Normalization/attenuation scaling (if needed)
        if self.scale_projection_by_sensitivity:
            if self.proj_meta.weights is None:
                raise Exception('If scaling by sensitivity, then `weights` must be provided in the projection metadata')
            else:
                proj = proj * self.get_projection_subset(self.proj_meta.weights, subset_idx).to(proj.device)
        BP = 0
        for proj_i, idx_partial in zip(torch.tensor_split(proj, self.N_splits), torch.tensor_split(idx, self.N_splits)):
            proj_i = proj_i.to(pytomography.device)
            if self.TOF:
                BP += parallelproj.joseph3d_back_tof_lm(
                    self.proj_meta.scanner_lut[idx_partial[:,0].to(torch.int)].to(pytomography.device),
                    self.proj_meta.scanner_lut[idx_partial[:,1].to(torch.int)].to(pytomography.device),
                    self.object_meta.shape,
                    self.object_origin,
                    self.object_meta.dr,
                    proj_i,
                    self.proj_meta.tof_meta.bin_width,
                    self.proj_meta.tof_meta.sigma,
                    self.proj_meta.tof_meta.center_offset,
                    self.proj_meta.tof_meta.n_sigmas,
                    idx_partial[:,2].squeeze().to(pytomography.device) - (self.proj_meta.tof_meta.num_bins - 1) // 2
                    )
            else:
                BP += parallelproj.joseph3d_back(
                    self.proj_meta.scanner_lut[idx_partial[:,0].to(torch.int)].to(pytomography.device),
                    self.proj_meta.scanner_lut[idx_partial[:,1].to(torch.int)].to(pytomography.device),
                    self.object_meta.shape,
                    self.object_origin,
                    self.object_meta.dr,
                    proj_i
                    )
        # Apply object transforms
        norm_constant = self.compute_normalization_factor(subset_idx)
        for transform in self.obj2obj_transforms[::-1]:
            if return_norm_constant:
                BP, norm_constant = transform.backward(BP, norm_constant=norm_constant)
            else:
                BP  = transform.backward(BP)
        # Return
        if return_norm_constant:
            return BP.to(self.output_device), norm_constant.to(self.output_device)
        else:
            return BP.to(self.output_device)