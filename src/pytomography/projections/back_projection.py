from __future__ import annotations
import torch
from pytomography.utils.helper_functions import rotate_detector_z, pad_object, unpad_object, pad_image
from .projection import ProjectionNet
from pytomography.priors import Prior

class BackProjectionNet(ProjectionNet):
    r"""Implements a back projection of mathematical form :math:`f_i = \frac{1}{\sum_j c_{ij}}\sum_{j} c_{ij} g_j`. where :math:`f_j` is an object, :math:`g_j` is an image, and :math:`c_{ij}` is the system matrix given by the various phenonemon modeled (e.g. atteunation correction/PSF). Subclass of the ``ProjectionNet`` class."""
    def forward(
        self,
        image: torch.tensor,
        angle_subset: list | None = None,
        prior: Prior | None = None,
        normalize: bool = False,
        return_norm_constant: bool = False,
        delta: float = 1e-11
    ) -> torch.tensor:
        r"""Implements back projection on an image, returning an object.

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
        for net in self.im2im_nets[::-1]:
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
            for net in self.obj2obj_nets[::-1]:
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