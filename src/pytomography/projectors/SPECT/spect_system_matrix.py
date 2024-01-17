from __future__ import annotations
import torch
import pytomography
from pytomography.transforms import Transform, RotationTransform
from pytomography.metadata import SPECTObjectMeta, SPECTProjMeta
from pytomography.priors import Prior
from pytomography.utils import pad_object, unpad_object, pad_proj, unpad_proj
from ..system_matrix import SystemMatrix

class SPECTSystemMatrix(SystemMatrix):
    r"""System matrix for SPECT imaging. By default, this applies to parallel hole collimators, but appropriate use of `proj2proj_transforms` can allow this system matrix to also model converging/diverging collimator configurations as well.
    
    Args:
            obj2obj_transforms (Sequence[Transform]): Sequence of object mappings that occur before forward projection.
            proj2proj_transforms (Sequence[Transform]): Sequence of proj mappings that occur after forward projection.
            object_meta (SPECTObjectMeta): SPECT Object metadata.
            proj_meta (SPECTProjMeta): SPECT projection metadata.
            n_parallel (int): Number of projections to use in parallel when applying transforms. More parallel events may speed up reconstruction time, but also increases GPU usage. Defaults to 1.
    """
    def __init__(
        self,
        obj2obj_transforms: list[Transform],
        proj2proj_transforms: list[Transform],
        object_meta: SPECTObjectMeta,
        proj_meta: SPECTProjMeta,
        n_parallel = 1,
    ) -> None:
        super(SPECTSystemMatrix, self).__init__(obj2obj_transforms, proj2proj_transforms, object_meta, proj_meta)
        self.n_parallel = n_parallel
        self.rotation_transform = RotationTransform()
    
    def compute_normalization_factor(self, subset_idx : int | None = None) -> torch.tensor:
        """Function used to get normalization factor :math:`H^T_m 1` corresponding to projection subset :math:`m`.

        Args:
            subset_idx (int | None, optional): Index of subset. If none, then considers all projections. Defaults to None.

        Returns:
            torch.Tensor: normalization factor :math:`H^T_m 1`
        """
        
        norm_proj = torch.ones((1, *self.proj_meta.shape)).to(pytomography.device)
        if subset_idx is not None:
            norm_proj = norm_proj[:,self.subset_indices_array[subset_idx]]
        return self.backward(norm_proj, subset_idx)
        
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
        return projections[:,self.subset_indices_array[subset_idx]]
    
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
        return len(self.subset_indices_array[subset_idx]) / self.proj_meta.num_projections

    def forward(
        self,
        object: torch.tensor,
        subset_idx: int | None = None,
    ) -> torch.tensor:
        r"""Applies forward projection to ``object`` for a SPECT imaging system.

        Args:
            object (torch.tensor[batch_size, Lx, Ly, Lz]): The object to be forward projected
            subset_idx (int, optional): Only uses a subset of angles :math:`g_m` corresponding to the provided subset index :math:`m`. If None, then defaults to the full projections :math:`g`.

        Returns:
            torch.tensor: forward projection estimate :math:`g_m=H_mf`
        """
        # Deal with subset stuff
        if subset_idx is not None:
            angle_subset = self.subset_indices_array[subset_idx]
        N_angles = self.proj_meta.num_projections if subset_idx is None else len(angle_subset)
        angle_indices = torch.arange(N_angles) if subset_idx is None else angle_subset
        # Start projection
        object = object.to(pytomography.device)
        proj = torch.zeros(
            (object.shape[0],N_angles,*self.proj_meta.padded_shape[1:])
            ).to(pytomography.device)
        # Loop through all angles (or groups of angles in parallel)
        for i in range(0, len(angle_indices), self.n_parallel):
            # Get angle indices
            angle_indices_single_batch_i = angle_indices[i:i+self.n_parallel]
            angle_indices_i = angle_indices_single_batch_i.repeat(object.shape[0])
            # Format Object
            object_i = torch.repeat_interleave(object, len(angle_indices_single_batch_i), 0)
            object_i = pad_object(object_i)
            # beta = 270 - phi, and backward transform called because projection should be at +beta (requires inverse rotation of object)
            object_i = self.rotation_transform.backward(object_i, 270-self.proj_meta.angles[angle_indices_i])
            # Apply object 2 object transforms
            for transform in self.obj2obj_transforms:
                object_i = transform.forward(object_i, angle_indices_i)
            # Reshape to 5D tensor of shape [batch_size, N_parallel, Lx, Ly, Lz]
            object_i = object_i.reshape((object.shape[0], -1, *self.object_meta.padded_shape))
            proj[:,i:i+self.n_parallel] = object_i.sum(axis=2)
        for transform in self.proj2proj_transforms:
            proj = transform.forward(proj)
        return unpad_proj(proj)
    
    def backward(
        self,
        proj: torch.tensor,
        subset_idx: int | None = None,
        return_norm_constant: bool = False,
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
        angle_indices = torch.arange(N_angles) if subset_idx is None else angle_subset
        # Box used to perform back projection
        boundary_box_bp = pad_object(torch.ones((1, *self.object_meta.shape)).to(pytomography.device), mode='back_project')
        # Pad proj and norm_proj (norm_proj used to compute sum_j H_ij)
        norm_proj = torch.ones(proj.shape).to(pytomography.device)
        proj = pad_proj(proj)
        norm_proj = pad_proj(norm_proj)
        # First apply proj transforms before back projecting
        for transform in self.proj2proj_transforms[::-1]:
            if return_norm_constant:
                proj, norm_proj = transform.backward(proj, norm_proj)
            else:
                proj = transform.backward(proj)
        # Setup for back projection
        object = torch.zeros([proj.shape[0], *self.object_meta.padded_shape]).to(pytomography.device)
        norm_constant = torch.zeros([proj.shape[0], *self.object_meta.padded_shape]).to(pytomography.device)
        for i in range(0, len(angle_indices), self.n_parallel):
            angle_indices_i = angle_indices[i:i+self.n_parallel]
            # Perform back projection
            object_i = proj[:,i:i+self.n_parallel].flatten(0,1).unsqueeze(1) * boundary_box_bp
            norm_constant_i = norm_proj[:,i:i+self.n_parallel].flatten(0,1).unsqueeze(1) * boundary_box_bp
            # Apply object mappings
            for transform in self.obj2obj_transforms[::-1]:
                if return_norm_constant:
                    object_i, norm_constant_i = transform.backward(object_i, angle_indices_i, norm_constant=norm_constant_i)
                else:
                    object_i  = transform.backward(object_i, angle_indices_i)
            # Rotate all objects by by their respective angle
            object_i = self.rotation_transform.forward(object_i, 270-self.proj_meta.angles[angle_indices_i])
            norm_constant_i = self.rotation_transform.forward(norm_constant_i, 270-self.proj_meta.angles[angle_indices_i])
            # Reshape to 5D tensor of shape [batch_size, N_parallel, Lx, Ly, Lz]
            object_i = object_i.reshape((object.shape[0], -1, *self.object_meta.padded_shape))
            norm_constant_i = norm_constant_i.reshape((object.shape[0], -1, *self.object_meta.padded_shape))
            # Add to total by summing over the N_parallel dimension (sum over all angles)
            object += object_i.sum(axis=1)
            norm_constant += norm_constant_i.sum(axis=1)
        # Unpad
        norm_constant = unpad_object(norm_constant)
        object = unpad_object(object)
        # Return
        if return_norm_constant:
            return object, norm_constant
        else:
            return object
        
        
class SPECTSystemMatrixMaskedSegments(SPECTSystemMatrix):
    r"""SPECT system matrix where the object space is a vector of length :math:`N` consisting of the mean activities for each masks in ``masks``. This system matrix can be used in reconstruction algorithms to obtain maximum liklihood estimations for the average value of :math:`f` inside each of the masks.
    
    Args:
            obj2obj_transforms (Sequence[Transform]): Sequence of object mappings that occur before forward projection.
            proj2proj_transforms (Sequence[Transform]): Sequence of proj mappings that occur after forward projection.
            object_meta (SPECTObjectMeta): SPECT Object metadata.
            proj_meta (SPECTProjMeta): SPECT proj metadata.
            masks (torch.Tensor): Masks corresponding to each segmented region.
    """
    def __init__(
        self,
        obj2obj_transforms: list[Transform],
        proj2proj_transforms: list[Transform],
        object_meta: SPECTObjectMeta,
        proj_meta: SPECTProjMeta,
        masks: torch.Tensor
        
    ) -> None:
        super(SPECTSystemMatrixMaskedSegments, self).__init__(obj2obj_transforms, proj2proj_transforms, object_meta, proj_meta)
        self.masks = masks.to(pytomography.device)

    def forward(
        self,
        activities: torch.Tensor,
        angle_subset: list[int] = None,
    ) -> torch.Tensor:
        r"""Implements forward projection :math:`HUa` on a vector of activities :math:`a` corresponding to `self.masks`.

        Args:
            activities (torch.tensor[batch_size, n_masks]): Activities in each mask region.
            angle_subset (list, optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.

        Returns:
            torch.tensor[batch_size, Ltheta, Lx, Lz]: Forward projected projections where Ltheta is specified by `self.proj_meta` and `angle_subset`.
        """
        object = 0
        activities = activities.reshape((*activities.shape, 1, 1, 1)).to(pytomography.device)
        object = (activities*self.masks).sum(axis=1)
        return super(SPECTSystemMatrixMaskedSegments, self).forward(object, angle_subset)
    
    def backward(
        self,
        proj: torch.Tensor,
        angle_subset: list | None = None,
        prior: Prior | None = None,
        normalize: bool = False,
        return_norm_constant: bool = False,
    ) -> torch.Tensor:
        r"""Implements back projection :math:`U^T H^T g` on projections :math:`g`, returning a vector of activities for each mask region.

        Args:
            proj (torch.tensor[batch_size, Ltheta, Lr, Lz]): projections which are to be back projected
            angle_subset (list, optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.
            prior (Prior, optional): If included, modifes normalizing factor to :math:`\frac{1}{\sum_j H_{ij} + P_i}` where :math:`P_i` is given by the prior. Used, for example, during in MAP OSEM. Defaults to None.
            normalize (bool): Whether or not to divide result by :math:`\sum_j H_{ij}`
            return_norm_constant (bool): Whether or not to return :math:`1/\sum_j H_{ij}` along with back projection. Defaults to 'False'.

        Returns:
            torch.tensor[batch_size, n_masks]: the activities in each mask region.
        """
        object, norm_constant = super(SPECTSystemMatrixMaskedSegments, self).backward(proj, angle_subset, prior, normalize=False, return_norm_constant = True, delta = pytomography.delta)
        activities = (object.unsqueeze(dim=1) * self.masks).sum(axis=(-1,-2,-3))
        norm_constant = (norm_constant.unsqueeze(dim=1) * self.masks).sum(axis=(-1,-2,-3))
        if normalize:
            activities = (activities+pytomography.delta)/(norm_constant + pytomography.delta)
        if return_norm_constant:
            return activities, norm_constant+pytomography.delta
        else:
            return activities