import torch
import torch.nn as nn
from pytomography.utils.helper_functions import rotate_detector_z


class BackProjectionNet(nn.Module):
    def __init__(self, object_correction_nets, image_correction_nets,
                 object_meta, image_meta, device='cpu'):
        super(BackProjectionNet, self).__init__()
        self.device = device
        self.object_correction_nets = object_correction_nets.copy()
        self.image_correction_nets = image_correction_nets.copy()
        self.object_correction_nets.reverse()
        self.image_correction_nets.reverse()
        self.object_meta = object_meta
        self.image_meta = image_meta

    def forward(self, image, angle_subset=None, prior=None, delta=1e-11):
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