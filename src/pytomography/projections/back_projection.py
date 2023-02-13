import torch
import torch.nn as nn
from pytomography.utils.helper_functions import rotate_detector_z, pad_image, pad_object, unpad_image, unpad_object


class BackProjectionNetOld(nn.Module):
    r"""Implements a back projection of mathematical form :math:`f_i = \frac{1}{\sum_j c_{ij}}\sum_{j} c_{ij} g_j`. where :math:`f_j` is an object, :math:`g_j` is an image, and :math:`c_{ij}` is the system matrix given by the various phenonemon modeled (atteunation correction/PSF). """
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
        super(BackProjectionNetOld, self).__init__()
        self.device = device
        self.object_correction_nets = object_correction_nets.copy()
        self.image_correction_nets = image_correction_nets.copy()
        self.object_correction_nets.reverse()
        self.image_correction_nets.reverse()
        self.object_meta = object_meta
        self.image_meta = image_meta

    def forward(self, image, angle_subset=None, prior=None, delta=1e-11):
        r"""Implements back projection on an image, returning an object.

        Args:
            image (torch.tensor[batch_size, Ltheta, Lr, Lz]): image which is to be back projected
            angle_subset (list, optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.
            prior (Prior, optional): If included, modifes normalizing factor to :math:`\frac{1}{\sum_j c_{ij} + P_i}` where :math:`P_i` is given by the prior. Used, for example, during in MAP OSEM. Defaults to None.
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

class BackProjectionNet(nn.Module):
    r"""Implements a back projection of mathematical form :math:`f_i = \frac{1}{\sum_j c_{ij}}\sum_{j} c_{ij} g_j`. where :math:`f_j` is an object, :math:`g_j` is an image, and :math:`c_{ij}` is the system matrix given by the various phenonemon modeled (atteunation correction/PSF). """
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
        r"""Implements back projection on an image, returning an object.

        Args:
            image (torch.tensor[batch_size, Ltheta, Lr, Lz]): image which is to be back projected
            angle_subset (list, optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.
            prior (Prior, optional): If included, modifes normalizing factor to :math:`\frac{1}{\sum_j c_{ij} + P_i}` where :math:`P_i` is given by the prior. Used, for example, during in MAP OSEM. Defaults to None.
            delta (float, optional): Prevents division by zero when dividing by normalizing constant. Defaults to 1e-11.

        Returns:
            torch.tensor[batch_size, Lr, Lr, Lz]: the object obtained from back projection.
        """
        # First apply any image corrections before back projecting
        for net in self.image_correction_nets:
            image = net(image)
        # Then do back projection
        N_angles = self.image_meta.num_projections
        object = torch.zeros([image.shape[0], *self.object_meta.shape]).to(self.device)
        norm_constant = torch.zeros([image.shape[0], *self.object_meta.shape]).to(self.device)
        looper = range(N_angles) if angle_subset is None else angle_subset
        for i in looper:
            # Pad image so that Lr aligns with extended Lx'
            image_padded = pad_image(image)
            # Define the padded object shape using the padded image
            object_i_shape = (image_padded.shape[0], image_padded.shape[2], image_padded.shape[2], image_padded.shape[3])
            # Perform back projection
            object_i = image_padded[:,i].unsqueeze(dim=1)*torch.ones(object_i_shape).to(self.device)
            norm_constant_i = torch.ones(object_i_shape).to(self.device)
            # Apply any corrections
            for net in self.object_correction_nets:
                object_i = net(object_i, i, norm_constant=norm_constant_i)
            #Rotate and unpad the normalization constant and the object
            norm_constant_after_rotation = rotate_detector_z(norm_constant_i, self.image_meta.angles[i], negative=True)
            norm_constant += unpad_object(norm_constant_after_rotation, object.shape)
            object += unpad_object(rotate_detector_z(object_i, self.image_meta.angles[i], negative=True), object.shape)
        # Apply prior 
        if prior:
            norm_constant += prior()
        return object/(norm_constant + delta)