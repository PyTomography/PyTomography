import torch
from pytomography.utils import rotate_detector_z, pad_object, unpad_image
from .projection import ProjectionNet

class ForwardProjectionNet(ProjectionNet):
    """Implements a forward projection of mathematical form :math:`g_j = \sum_{i} c_{ij} f_i` where :math:`f_i` is an object, :math:`g_j` is the corresponding image, and :math:`c_{ij}` is the system matrix given by the various phenonemon modeled (e.g. atteunation/PSF).
    """
    def forward(
        self,
        object: torch.tensor,
        angle_subset: list[int] = None,
    ) -> torch.tensor:
        r"""Implements forward projection on an object

        Args:
            object (torch.tensor[batch_size, Lx, Ly, Lz]): The object to be forward projected
            angle_subset (list, optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None,
            which assumes all angles are used.

        Returns:
            torch.tensor[batch_size, Ltheta, Lx, Lz]: Forward projected image where Ltheta is specified by `self.image_meta` and `angle_subset`.
        """
        N_angles = self.image_meta.num_projections
        image = torch.zeros((object.shape[0],*self.image_meta.padded_shape)).to(self.device)
        looper = range(N_angles) if angle_subset is None else angle_subset
        for i in looper:
            object_i = rotate_detector_z(pad_object(object), self.image_meta.angles[i])
            for net in self.obj2obj_nets:
                object_i = net(object_i, i)
            image[:,i] = object_i.sum(axis=1)
        for net in self.im2im_nets:
            image = net(image)
        return unpad_image(image)