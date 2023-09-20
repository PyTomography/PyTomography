from __future__ import annotations
import torch
import pytomography
from pytomography.transforms import Transform
from pytomography.metadata import SPECTObjectMeta, SPECTProjMeta
from pytomography.priors import Prior
from pytomography.utils import rotate_detector_z, pad_object, unpad_object, pad_proj, unpad_proj
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

    def forward(
        self,
        object: torch.tensor,
        angle_subset: list[int] = None,
    ) -> torch.tensor:
        r"""Applies forward projection to ``object`` for a SPECT imaging system.

        Args:
            object (torch.tensor[batch_size, Lx, Ly, Lz]): The object to be forward projected
            angle_subset (list, optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.

        Returns:
            torch.tensor[batch_size, Ltheta, Lx, Lz]: Forward projected projections where Ltheta is specified by `self.proj_meta` and `angle_subset`.
        """
        N_angles = self.proj_meta.num_projections
        object = object.to(pytomography.device)
        proj = torch.zeros((object.shape[0],*self.proj_meta.padded_shape)).to(pytomography.device)
        angle_indices = torch.arange(N_angles) if angle_subset is None else angle_subset
        # Loop through all angles (or groups of angles in parallel)
        for i in range(0, len(angle_indices), self.n_parallel):
            # Get angle indices
            angle_indices_single_batch_i = angle_indices[i:i+self.n_parallel]
            angle_indices_i = angle_indices_single_batch_i.repeat(object.shape[0])
            # Format Object
            object_i = torch.repeat_interleave(object, len(angle_indices_single_batch_i), 0)
            object_i = pad_object(object_i)
            object_i = rotate_detector_z(object_i, self.proj_meta.angles[angle_indices_i])
            # Apply object 2 object transforms
            for transform in self.obj2obj_transforms:
                object_i = transform.forward(object_i, angle_indices_i)
            # Reshape to 5D tensor of shape [batch_size, N_parallel, Lx, Ly, Lz]
            object_i = object_i.reshape((object.shape[0], -1, *self.object_meta.padded_shape))
            proj[:,angle_indices_single_batch_i] = object_i.sum(axis=2)
        for transform in self.proj2proj_transforms:
            proj = transform.forward(proj)
        return unpad_proj(proj)
    
    def backward(
        self,
        proj: torch.tensor,
        angle_subset: list | None = None,
        return_norm_constant: bool = False,
    ) -> torch.tensor:
        r"""Applies back projection to ``proj`` for a SPECT imaging system.

        Args:
            proj (torch.tensor[batch_size, Ltheta, Lr, Lz]): projections which are to be back projected
            angle_subset (list, optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.
            return_norm_constant (bool): Whether or not to return :math:`1/\sum_j H_{ij}` along with back projection. Defaults to 'False'.

        Returns:
            torch.tensor[batch_size, Lr, Lr, Lz]: the object obtained from back projection.
        """
        # Box used to perform back projection
        boundary_box_bp = pad_object(torch.ones((1, *self.object_meta.shape)).to(pytomography.device), mode='back_project')
        # Pad proj and norm_proj (norm_proj used to compute sum_j H_ij)
        norm_proj = torch.ones(proj.shape).to(pytomography.device)
        proj = pad_proj(proj)
        norm_proj = pad_proj(norm_proj)
        # First apply proj transforms before back projecting
        for transform in self.proj2proj_transforms[::-1]:
            proj, norm_proj = transform.backward(proj, norm_proj)
        # Setup for back projection
        N_angles = self.proj_meta.num_projections
        object = torch.zeros([proj.shape[0], *self.object_meta.padded_shape]).to(pytomography.device)
        norm_constant = torch.zeros([proj.shape[0], *self.object_meta.padded_shape]).to(pytomography.device)
        angle_indices = torch.arange(N_angles) if angle_subset is None else angle_subset
        for i in range(0, len(angle_indices), self.n_parallel):
            angle_indices_single_batch_i = angle_indices[i:i+self.n_parallel]
            angle_indices_i = angle_indices_single_batch_i.repeat(object.shape[0])
            # Perform back projection
            object_i = proj[:,angle_indices_single_batch_i].flatten(0,1).unsqueeze(1) * boundary_box_bp
            norm_constant_i = norm_proj[:,angle_indices_single_batch_i].flatten(0,1).unsqueeze(1) * boundary_box_bp
            # Apply object mappings
            for transform in self.obj2obj_transforms[::-1]:
                object_i, norm_constant_i = transform.backward(object_i, angle_indices_i, norm_constant=norm_constant_i)
            # Rotate all objects by by their respective angle
            object_i = rotate_detector_z(object_i, self.proj_meta.angles[angle_indices_i], negative=True)
            norm_constant_i = rotate_detector_z(norm_constant_i, self.proj_meta.angles[angle_indices_i], negative=True)
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