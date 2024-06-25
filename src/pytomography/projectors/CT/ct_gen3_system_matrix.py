from __future__ import annotations
import torch
import pytomography
from pytomography.projectors import SystemMatrix
from pytomography.metadata import ObjectMeta
from pytomography.metadata.CT import CTGen3ProjMeta
try:
    import parallelproj
except:
    pass

class CTGen3SystemMatrix(SystemMatrix):
    """System matrix for 3rd generation clinical DICOM scanners with cylindrical detector panels. For more information, see the DICOM-CTPD user manual.

        Args:
            object_meta (ObjectMeta): Metadata for object space
            proj_meta (CTConeBeamFlatPanelProjMeta): Projection metadata for the CT system
            N_splits (int, optional): Splits up computation of forward/back projection to save GPU memory. Defaults to 1.
            device (str, optional): Device on which projections are output. Defaults to pytomography.device.
    """
    def __init__(
        self,
        object_meta: ObjectMeta,
        proj_meta: CTGen3ProjMeta,
        N_splits: int = 1,
        device: str = pytomography.device
    ) -> None:
        super(CTGen3SystemMatrix, self).__init__(object_meta, proj_meta)
        # Used for parallelproj projectors
        self.origin = -(torch.tensor(object_meta.shape).to(pytomography.device)/2-0.5) * torch.tensor(object_meta.dr).to(pytomography.dtype).to(pytomography.device)
        self.voxel_size = torch.tensor(object_meta.dr).to(pytomography.dtype).to(pytomography.device)
        self.N_splits = N_splits
        self.device = device
        
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
        
    def compute_normalization_factor(self, subset_idx):
        r"""Computes the normalization factor :math:`H^T 1`

        Args:
            subset_idx (int, optional): Subset index for ths sinogram. If None, considers all elements. Defaults to None..

        Returns:
            torch.Tensor: Normalization factor.
        """
        # Put BP on cpu since we could potentially have a lot of them
        return self.backward(torch.ones(self.proj_meta.N_angles, *self.proj_meta.shape).to(self.device), subset_idx).cpu()
        
    def forward(self, object, subset_idx=None, *args, **kwargs):
        """Computes forward projection

        Args:
            object (torch.Tensor): Object to be forward projected
            subset_idx (int | None, optional): Subset index :math:`m` of the projection. If None, then projects to entire projection space. Defaults to None.

        Returns:
            torch.Tensor: Projections corresponding to :math:`\int \mu dx` along all LORs.
        """
        if subset_idx is not None:
            angle_subset = self.subset_indices_array[subset_idx]
        angle_indices = torch.arange(self.proj_meta.N_angles).to(self.device) if subset_idx is None else angle_subset
        # Forward project
        proj_tot = []
        for idxs in torch.tensor_split(angle_indices, self.N_splits):
            detector_coordinates_i = self.proj_meta.get_detector_coordinates(idxs).flatten(end_dim=2)
            beam_coordinate_i = self.proj_meta.source_focal_spots[idxs][:,None,None].repeat(1,self.proj_meta.shape[0],self.proj_meta.shape[1],1).flatten(end_dim=2)
            proj = parallelproj.joseph3d_fwd(
                beam_coordinate_i,
                detector_coordinates_i,
                object,
                self.origin,
                self.voxel_size
            ).reshape(idxs.shape[0], *self.proj_meta.shape).to(self.device)
            proj_tot.append(proj.to(self.device))
        return torch.concatenate(proj_tot)
    
    def backward(self, proj, subset_idx=None, *args, **kwargs):
        """Computes back projection

        Args:
            object (torch.Tensor): Object to be forward projected
            subset_idx (int | None, optional): Subset index :math:`m` of the projection. If None, then projects to entire projection space. Defaults to None.

        Returns:
            torch.Tensor: Projections corresponding to :math:`\int \mu dx` along all LORs.
        """
        if subset_idx is not None:
            angle_subset = self.subset_indices_array[subset_idx]
        angle_indices = torch.arange(self.proj_meta.N_angles).to(self.device) if subset_idx is None else angle_subset
        BP = 0
        for ii, idxs in zip(torch.tensor_split(torch.arange(angle_indices.shape[0]), self.N_splits), torch.tensor_split(angle_indices, self.N_splits)):
            detector_coordinates_i = self.proj_meta.get_detector_coordinates(idxs).flatten(end_dim=2)
            beam_coordinate_i = self.proj_meta.source_focal_spots[idxs][:,None,None].repeat(1,self.proj_meta.shape[0],self.proj_meta.shape[1],1).flatten(end_dim=2)
            proj_i = proj[ii].to(pytomography.device)
            # Preprocessing?
            # ...
            # Now back project
            proj_i = proj_i.flatten().to(pytomography.device)
            BP_i = 0
            BP_i = BP_i + parallelproj.joseph3d_back(
                beam_coordinate_i,
                detector_coordinates_i,
                self.object_meta.shape,
                self.origin,
                self.voxel_size,
                proj_i
                )
            BP += BP_i
            del(proj_i)
            del(detector_coordinates_i)
            del(beam_coordinate_i)
            torch.cuda.empty_cache()
        return BP