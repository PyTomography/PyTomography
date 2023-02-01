import torch
import torch.nn as nn
from pytomography.utils.helper_functions import rotate_detector_z

# INPUT:
# object [batch_size, Lx, Ly, Lz]
# angles [batch_size, num_projections]
# radii? [ num_projections] b/c all in batch should have same radii
# CT? [batch_size]
# OUTPUT:
# image [batch_size, num_projections, Ly, Lz]
class ForwardProjectionNet(nn.Module):
    def __init__(self, object_correction_nets, image_correction_nets,
                object_meta, image_meta, device='cpu'):
        super(ForwardProjectionNet, self).__init__()
        self.device = device
        self.object_correction_nets = object_correction_nets
        self.image_correction_nets = image_correction_nets
        self.object_meta = object_meta
        self.image_meta = image_meta

    def forward(self, object, angle_subset=None):
        N_angles = self.image_meta.num_projections
        image = torch.zeros((object.shape[0], N_angles, object.shape[2], object.shape[3])).to(self.device)
        looper = range(N_angles) if angle_subset is None else angle_subset
        for i in looper:
            object_i = rotate_detector_z(object, self.image_meta.angles[i])
            for net in self.object_correction_nets:
                object_i = net(object_i, i)
            image[:,i] = object_i.sum(axis=1)
        return image