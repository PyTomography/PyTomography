from __future__ import annotations
import torch
import pytomography
from pytomography.metadata import ObjectMeta, PETLMProjMeta
from pytomography.projectors import SystemMatrix
import numpy as np
import parallelproj
class PETLMSystemMatrix(SystemMatrix):
    """System matrix used to model forward and back projection of PET list mode data. Projections correspond to lists consisiting of the detector-pair indices for every detected event. The `proj_meta` argument contains a lookup table used to convert these indices to spatial coordinates. This projector is still under development in unison with the PETSIRD datatype; future updates will contain information required for valid-detector pairs when computing normalization factors.

        Args:
            event_detector_1_id (torch.tensor[int]): Indices corresponding to the first detector of coincidence events.
            event_detector_2_id (torch.tensor[int]): Indices corresponding to the second detector of coincidence events.
            object_meta (SPECTObjectMeta): Metadata of object space.
            proj_meta (PETLMProjMeta): PET listmode projection space metadata.
            attenuation_map (torch.tensor[float] | None, optional): Attenuation map used for attenuation modeling. Defaults to None.
            event_det_TOF (torch.tensor[int] | None, optional): Time of flight index corresponding to the detected event. If None, then TOF is not used. Defaults to None.
            include_weighting_in_projection (bool, optional): Includes sensitivty and attenuation maps in forward/back projection. In image reconstruction using OSEM (for example) this is not required due to factors canceling out. It may be required for other reconstruction algorithms analytic modeling. Defaults to False.
    """
    def __init__(
        self,
        event_detector_ids: torch.tensor[int],
        object_meta: ObjectMeta,
        proj_meta: PETLMProjMeta,
        attenuation_map: torch.tensor[float] | None = None,
        obj2obj_transforms = [],
        include_weighting_in_projection: bool = False,
        N_splits: int = 1,
        device: str = pytomography.device,
        geo_sens = False # REMOVE LATER
    ) -> None:
        self.device = device
        self.event_detector_ids = event_detector_ids.to(self.device)
        if self.event_detector_ids.shape[1]==3:
            self.TOF = True
        else:
            self.TOF = False
        self.object_origin = (- np.array(object_meta.shape) / 2 + 0.5) * (np.array(object_meta.dr))
        self.obj2obj_transforms = obj2obj_transforms
        super(PETLMSystemMatrix, self).__init__(
            obj2obj_transforms=self.obj2obj_transforms,
            proj2proj_transforms=[],
            object_meta=object_meta,
            proj_meta=proj_meta
            )
        self.attenuation_map = attenuation_map
        self.include_weighting_in_projection = include_weighting_in_projection
        self.N_splits = N_splits
        self.geo_sens = geo_sens
        self._compute_sens_factor()
        
    def get_subset_splits(self, n_subsets: int) -> list:
        """Returns a list where each element consists of an array of indices corresponding to a partitioned version of the projections. 

        Args:
            n_subsets (int): Number of subsets to partition the projections into

        Returns:
            list: List of arrays where each array corresponds to the projection indices of a particular subset.
        """
        indices = torch.arange(self.event_detector_ids.shape[0]).to(torch.long).to(pytomography.device)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        return subset_indices_array
    
    def compute_atteunation_probability_projection(self, idx):
        """Computes probabilities of photons being attenuated along a collection of LORs

        Args:
            idx_start (torch.tensor): Indices corresponding to detector 1
            idx_end (torch.Tensor): Indices corresponding to detector 2

        Returns:
            torch.Tensor: The probabilities of photons being attenuated along each LOR
        """
        return torch.exp(-parallelproj.joseph3d_fwd(
            self.proj_meta.scanner_lut[idx[:,0]],
            self.proj_meta.scanner_lut[idx[:,1]],
            self.attenuation_map[0],
            self.object_origin,
            self.object_meta.dr,
            num_chunks=4))
        
    def compute_geo_norm(self, combos):
        r1 = self.proj_meta.scanner_lut[combos[:,0]]
        r2 = self.proj_meta.scanner_lut[combos[:,1]]
        d = r2 - r1
        return (-r1[:,0]*d[:,0] - r1[:,1]*d[:,1])/(torch.norm(r1, dim=1)*torch.norm(d, dim=1)) \
        * (r2[:,0]*d[:,0] + r2[:,1]*d[:,1])/(torch.norm(r2, dim=1)*torch.norm(d, dim=1))
    
    def _compute_sens_factor(self, N_splits = 10):
        """Computes the normalization factor :math:`H^T 1`.

        Args:
            N_splits (int, optional): _description_. Defaults to 10.
        """
        idxs = torch.arange(self.proj_meta.scanner_lut.shape[0]).to(pytomography.device)
        combos_total = torch.combinations(idxs, 2)
        self.norm_BP = 0
        for combos in torch.tensor_split(combos_total, N_splits):
            ones = torch.ones(combos.shape[0]).to(pytomography.device)
            if self.attenuation_map is not None:
                ones *= self.compute_atteunation_probability_projection(combos[:,0], combos[:,1])
            if self.geo_sens:
                ones *= self.compute_geo_norm(combos)
            if self.TOF:
                ones_bins = torch.ones(combos.shape[0]).to(pytomography.device)
                offset =  (self.proj_meta.num_tof_bins-1) / 2
                for i in range(self.proj_meta.num_tof_bins):
                    tof_bins = (i-offset)*ones_bins.to(torch.int16)
                    self.norm_BP += parallelproj.joseph3d_back_tof_lm(
                        self.proj_meta.scanner_lut[combos[:,0]],
                        self.proj_meta.scanner_lut[combos[:,1]],
                        self.object_meta.shape,
                        self.object_origin,
                        self.object_meta.dr,
                        ones,
                        self.proj_meta.tofbin_width,
                        self.proj_meta.sigma_tof,
                        self.proj_meta.tofcenter_offset,
                        self.proj_meta.nsigmas,
                        tof_bins
                        ).unsqueeze(0)
            else:
                self.norm_BP += parallelproj.joseph3d_back(
                    self.proj_meta.scanner_lut[combos[:,0]],
                    self.proj_meta.scanner_lut[combos[:,1]],
                    self.object_meta.shape,
                    self.object_origin,
                    self.object_meta.dr,
                    ones,
                    num_chunks=4).unsqueeze(0)
        # Apply object transforms
        # Apply object transforms
        for transform in self.obj2obj_transforms[::-1]:
            self.norm_BP  = transform.backward(self.norm_BP)
        self.norm_BP = self.norm_BP.to(self.device)
    def compute_normalization_factor(self, angle_subset: list[int] = None):
        """Function called by reconstruction algorithms to get :math:`H^T 1`.

        Returns:
           torch.Tensor: Normalization factor :math:`H^T 1`.
        """
        if angle_subset is None:
            fraction_considered = 1
        else:
            fraction_considered = angle_subset.shape[0] / self.event_detector_ids.shape[0] 
        return fraction_considered * self.norm_BP
    
    def forward(
        self,
        object: torch.tensor,
        angle_subset: list[int] = None,
    ) -> torch.tensor:
        """Computes forward projection. In the case of list mode PET, this corresponds to the expected number of detected counts along each LOR corresponding to a particular object.

        Args:
            object (torch.tensor): Object to be forward projected
            angle_subset (list[int], optional): List of indices corresponding to a subset of the defined LORs. Defaults to None.

        Returns:
            torch.tensor: Projections corresponding to the expected number of counts along each LOR.
        """ 
        object = object.to(pytomography.device)
        # Apply object space transforms
        for transform in self.obj2obj_transforms:
            object = transform.forward(object)
        if angle_subset is not None:
            idx = self.event_detector_ids[angle_subset.to(self.device)].squeeze()
        else:
            idx = self.event_detector_ids.squeeze()
        proj = torch.tensor([]).to(self.device)
        for idx_partial in torch.tensor_split(idx, self.N_splits):
            if self.TOF:
                proj_i = parallelproj.joseph3d_fwd_tof_lm(
                    self.proj_meta.scanner_lut[idx_partial[:,0].to(torch.int)].to(pytomography.device),
                    self.proj_meta.scanner_lut[idx_partial[:,1].to(torch.int)].to(pytomography.device),
                    object[0],
                    self.object_origin,
                    self.object_meta.dr,
                    self.proj_meta.tofbin_width,
                    self.proj_meta.sigma_tof,
                    self.proj_meta.tofcenter_offset,
                    self.proj_meta.nsigmas,
                    idx_partial[:,2].squeeze().to(pytomography.device) - (self.proj_meta.num_tof_bins-1) / 2
                    )
            else:
                proj_i = parallelproj.joseph3d_fwd(
                    self.proj_meta.scanner_lut[idx_partial[:,0].to(torch.int)].to(pytomography.device),
                    self.proj_meta.scanner_lut[idx_partial[:,1].to(torch.int)].to(pytomography.device),
                    object[0],
                    self.object_origin,
                    self.object_meta.dr
                    )
            if self.include_weighting_in_projection:
                if self.attenuation_map is not None:
                    proj_i *= self.compute_atteunation_probability_projection(idx_partial)
            proj = torch.concatenate([proj, proj_i.to(self.device)])
        return proj
            
    def backward(
        self,
        proj: torch.tensor,
        angle_subset: list[int] = None,
        return_norm_constant: bool = False,
    ) -> torch.tensor:
        """Computes back projection. This corresponds to tracing a sequence of LORs into object space.

        Args:
            proj (torch.tensor): Projections to be back projected
            angle_subset (list[int], optional): List of indices designating a subset of projections. Defaults to None.
            return_norm_constant (bool, optional): Whether or not to return the normalization constant: useful in reconstruction algorithms that require :math:`H^T 1`. Defaults to False.

        Returns:
            torch.tensor: _description_
        """
        if angle_subset is not None:
            idx = self.event_detector_ids[angle_subset.to(self.device)].squeeze()
        else:
            idx = self.event_detector_ids.squeeze()
        BP = 0
        for proj_i, idx_partial in zip(torch.tensor_split(proj, self.N_splits), torch.tensor_split(idx, self.N_splits)):
            proj_i = proj_i.to(pytomography.device)
            if self.include_weighting_in_projection:
                if self.attenuation_map is not None:
                    proj_i = proj_i * self.compute_atteunation_probability_projection(idx_partial)
            if self.TOF:
                BP += parallelproj.joseph3d_back_tof_lm(
                    self.proj_meta.scanner_lut[idx_partial[:,0].to(torch.int)].to(pytomography.device),
                    self.proj_meta.scanner_lut[idx_partial[:,1].to(torch.int)].to(pytomography.device),
                    self.object_meta.shape,
                    self.object_origin,
                    self.object_meta.dr,
                    proj_i,
                    self.proj_meta.tofbin_width,
                    self.proj_meta.sigma_tof,
                    self.proj_meta.tofcenter_offset,
                    self.proj_meta.nsigmas,
                    idx_partial[:,2].squeeze().to(pytomography.device) - (self.proj_meta.num_tof_bins-1) / 2
                    ).unsqueeze(0)
            else:
                BP += parallelproj.joseph3d_back(
                    self.proj_meta.scanner_lut[idx_partial[:,0].to(torch.int)].to(pytomography.device),
                    self.proj_meta.scanner_lut[idx_partial[:,1].to(torch.int)].to(pytomography.device),
                    self.object_meta.shape,
                    self.object_origin,
                    self.object_meta.dr,
                    proj_i
                    ).unsqueeze(0)
        # Apply object transforms
        norm_constant = self.compute_normalization_factor(angle_subset)
        for transform in self.obj2obj_transforms[::-1]:
            if return_norm_constant:
                BP, norm_constant = transform.backward(BP, norm_constant=norm_constant)
            else:
                BP  = transform.backward(BP)
        # Return
        if return_norm_constant:
            return BP.to(self.device), norm_constant.to(self.device)
        else:
            return BP.to(self.device)