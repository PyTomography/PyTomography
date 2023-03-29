from __future__ import annotations
import torch
import torch.nn as nn
import abc
import pytomography
from pytomography.transforms import Transform
from pytomography.metadata import ObjectMeta, ImageMeta
from pytomography.priors import Prior
from pytomography.utils import rotate_detector_z, pad_object, unpad_object, pad_image, unpad_image

class SystemMatrix():
    r"""Update this
    
    Args:
            obj2obj_transforms (Sequence[Transform]): Sequence of object mappings that occur before forward projection.
            im2im_transforms (Sequence[Transform]): Sequence of image mappings that occur after forward projection.
            object_meta (ObjectMeta): Object metadata.
            image_meta (ImageMeta): Image metadata.
    """
    def __init__(
        self,
        obj2obj_transforms: list[Transform],
        im2im_transforms: list[Transform],
        object_meta: ObjectMeta,
        image_meta: ImageMeta,
    ) -> None:
        self.device = pytomography.device
        self.obj2obj_transforms = obj2obj_transforms
        self.im2im_transforms = im2im_transforms
        self.object_meta = object_meta
        self.image_meta = image_meta
        self.initialize_correction_nets()

    def initialize_correction_nets(self):
        """Initializes all mapping networks with the required object and image metadata corresponding to the projection network.
        """
        for net in self.obj2obj_transforms:
            net.configure(self.object_meta, self.image_meta)
        for net in self.im2im_transforms:
            net.configure(self.object_meta, self.image_meta)

    def forward(
        self,
        object: torch.tensor,
        angle_subset: list[int] = None,
    ) -> torch.tensor:
        r"""Implements forward projection :math:`Hf` on an object :math:`f`.

        Args:
            object (torch.tensor[batch_size, Lx, Ly, Lz]): The object to be forward projected
            angle_subset (list, optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.

        Returns:
            torch.tensor[batch_size, Ltheta, Lx, Lz]: Forward projected image where Ltheta is specified by `self.image_meta` and `angle_subset`.
        """
        N_angles = self.image_meta.num_projections
        object = object.to(self.device)
        image = torch.zeros((object.shape[0],*self.image_meta.padded_shape)).to(self.device)
        looper = range(N_angles) if angle_subset is None else angle_subset
        for i in looper:
            object_i = rotate_detector_z(pad_object(object), self.image_meta.angles[i])
            for net in self.obj2obj_transforms:
                object_i = net(object_i, i)
            image[:,i] = object_i.sum(axis=1)
        for net in self.im2im_transforms:
            image = net(image)
        return unpad_image(image)
    
    def backward(
        self,
        image: torch.tensor,
        angle_subset: list | None = None,
        prior: Prior | None = None,
        normalize: bool = False,
        return_norm_constant: bool = False,
        delta: float = 1e-11
    ) -> torch.tensor:
        r"""Implements back projection :math:`H^T g` on an image :math:`g`.

        Args:
            image (torch.tensor[batch_size, Ltheta, Lr, Lz]): image which is to be back projected
            angle_subset (list, optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.
            prior (Prior, optional): If included, modifes normalizing factor to :math:`\frac{1}{\sum_j c_{ij} + P_i}` where :math:`P_i` is given by the prior. Used, for example, during in MAP OSEM. Defaults to None.
            normalize (bool): Whether or not to divide result by :math:`\sum_j c_{ij}`
            return_norm_constant (bool): Whether or not to return :math:`1/\sum_j c_{ij}` along with back projection. Defaults to 'False'.
            delta (float, optional): Prevents division by zero when dividing by normalizing constant. Defaults to 1e-11.

        Returns:
            torch.tensor[batch_size, Lr, Lr, Lz]: the object obtained from back projection.
        """
        # Box used to perform back projection
        boundary_box_bp = pad_object(torch.ones((1, *self.object_meta.shape)).to(self.device), mode='back_project')
        # Pad image and norm_image (norm_image used to compute sum_j c_ij)
        norm_image = torch.ones(image.shape).to(self.device)
        image = pad_image(image)
        norm_image = pad_image(norm_image)
        # First apply image mappings before back projecting
        for net in self.im2im_transforms[::-1]:
            image = net(image, mode='back_project')
            norm_image = net(norm_image, mode='back_project')
        # Setup for back projection
        N_angles = self.image_meta.num_projections
        object = torch.zeros([image.shape[0], *self.object_meta.padded_shape]).to(self.device)
        norm_constant = torch.zeros([image.shape[0], *self.object_meta.padded_shape]).to(self.device)
        looper = range(N_angles) if angle_subset is None else angle_subset
        for i in looper:
            # Perform back projection
            object_i = image[:,i].unsqueeze(dim=1) * boundary_box_bp
            norm_constant_i = norm_image[:,i].unsqueeze(dim=1) * boundary_box_bp
            # Apply object mappings
            for net in self.obj2obj_transforms[::-1]:
                object_i, norm_constant_i = net(object_i, i, norm_constant=norm_constant_i)
            # Add to total
            norm_constant += rotate_detector_z(norm_constant_i, self.image_meta.angles[i], negative=True)
            object += rotate_detector_z(object_i, self.image_meta.angles[i], negative=True)
        # Unpad
        norm_constant = unpad_object(norm_constant)
        object = unpad_object(object)
        # Apply prior 
        if prior:
            norm_constant += prior()
        if normalize:
            object = (object+delta)/(norm_constant + delta)
        # Return
        if return_norm_constant:
            return object, norm_constant+delta
        else:
            return object