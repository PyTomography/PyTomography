import torch
import torch.nn as nn
from pytomography.utils.helper_functions import rotate_detector_z


class BackProjectionNet(nn.Module):
    """Implements a back projection of mathematical form $$f_j = \frac{1}{\sum_j c_{ij}}\sum_{j} c_{ij} g_j$$.
    where $f_j$ is an object, $g_j$ is an image, and $c_{ij}$ is the system matrix given
    by the various phenonemon modeled (atteunation correction/PSF). """
    def __init__(self, object_correction_nets, image_correction_nets,
                 object_meta, image_meta, device='cpu'):
        """Initializer

        Args:
            object_correction_nets (list): List of correction networks which operate on an object
            prior to forward projection such that subsequent forward projection leads to the 
            phenomenon being simulated.
            image_correction_nets (list): List of correction networks which operate on an object
            after forward projection such that desired phenoneon are simulated.
            object_meta (ObjectMeta): Object metadata.
            image_meta (ImageMeta): Image metadata.
            device (str, optional): Pytorch device used for computation. Defaults to 'cpu'.
        """
        super(BackProjectionNet, self).__init__()
        self.device = device
        self.object_correction_nets = object_correction_nets.copy()
        self.image_correction_nets = image_correction_nets.copy()
        self.object_correction_nets.reverse()
        self.image_correction_nets.reverse()
        self.object_meta = object_meta
        self.image_meta = image_meta

    def forward(self, image, angle_subset=None, prior=None, delta=1e-11):
        """Implements back projection on an image, returning an object.

        Args:
            image (torch.tensor[batch_size, Ltheta, Lr, Lz]): image which is to be back projected
            angle_subset (list, optional): Only uses a subset of angles (i.e. only certain values of $j$ in
            formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None,
            which assumes all angles are used.
            prior (Prior, optional): If included, modifes normalizing factor to $$\frac{1}{\sum_j c_{ij} + P_i} where
            $P_i$ is given by the prior. Used, for example, during in MAP OSEM. Defaults to None.
            delta (float, optional): Prevents division by zero when dividing by normalizing constant. Defaults to 1e-11.

        Returns:
            torch.tensor[batch_size, Lr, Lr, Lz]: the object obtained from back projection.
        """
        N_angles = self.image_meta.num_projections
        object = torch.zeros([image.shape[0], *self.object_meta.shape]).to(self.device)
        norm_constant = torch.zeros([image.shape[0], *self.object_meta.shape]).to(self.device)
        looper = range(N_angles) if angle_subset is None else angle_subset
        for i in looper:
            object_i = image[:,i].unsqueeze(dim=1)*torch.ones([image.shape[0], *self.object_meta.shape]).to(self.device)
            norm_constant_i = torch.ones([image.shape[0], *self.object_meta.shape]).to(self.device)
            for net in self.object_correction_nets:
                object_i = net(object_i, i, norm_constant=norm_constant_i)
            norm_constant += rotate_detector_z(norm_constant_i, self.image_meta.angles[i], negative=True)
            object += rotate_detector_z(object_i, self.image_meta.angles[i], negative=True)
        if prior:
            norm_constant += prior()
        return object/(norm_constant + delta)