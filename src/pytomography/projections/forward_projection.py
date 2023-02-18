import torch
from pytomography.utils import rotate_detector_z, pad_object, unpad_object
from .projection import ProjectionNet

class ForwardProjectionNet(ProjectionNet):
    """Implements a forward projection of mathematical form :math:`g_j = \sum_{i} c_{ij} f_i` where :math:`f_i` is an object, :math:`g_j` is the corresponding image, and :math:`c_{ij}` is the system matrix given by the various phenonemon modeled (atteunation correction/PSF).
    """
    def forward(
        self,
        object: torch.tensor,
        angle_subset: list[int] = None
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
        image = torch.zeros((object.shape[0], N_angles, object.shape[2], object.shape[3])).to(self.device)
        looper = range(N_angles) if angle_subset is None else angle_subset
        for i in looper:
            object_i = rotate_detector_z(pad_object(object), self.image_meta.angles[i])
            for net in self.object_correction_nets:
                object_i = net(object_i, i)
            image[:,i] = unpad_object(object_i, original_shape=self.object_meta.shape).sum(axis=1)
        return image