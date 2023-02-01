import torch
import torch.nn as nn
import numpy as np
from pytomography.projections import ForwardProjectionNet, BackProjectionNet
from pytomography.corrections import CTCorrectionNet, PSFCorrectionNet
from pytomography.io import simind_projections_to_data, simind_CT_to_data

class OSEMNet(nn.Module):
    def __init__(self, 
                 object_initial,
                 forward_projection_net,
                 back_projection_net,
                 prior = None):
        super(OSEMNet, self).__init__()
        self.forward_projection_net = forward_projection_net
        self.back_projection_net = back_projection_net
        self.object_prediction = object_initial
        self.prior = prior

    def get_subset_splits(self, n_subsets, n_angles):
        indices = np.arange(n_angles).astype(int)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        return subset_indices_array

    def set_image(self, image):
        self.image = image

    def set_prior(self, prior):
        self.prior = prior

    def forward(self, n_iters, n_subsets, comparisons=None, delta=1e-11):
        subset_indices_array = self.get_subset_splits(n_subsets, self.image.shape[1])
        for j in range(n_iters):
            for subset_indices in subset_indices_array:
                # Set OSL Prior to have object from previous prediction
                if self.prior:
                    self.prior.set_object(torch.clone(self.object_prediction))
                ratio = self.image / (self.forward_projection_net(self.object_prediction, angle_subset=subset_indices) + delta)
                self.object_prediction = self.object_prediction * self.back_projection_net(ratio, angle_subset=subset_indices, prior=self.prior)
                if comparisons:
                    for key in comparisons.keys():
                        comparisons[key].compare(self.object_prediction)
        return self.object_prediction

def get_osem_net(projections_header, object_initial='ones', CT_header=None, PSF_options=None, file_type='simind', device='cpu'):
    if file_type=='simind':
        object_meta, image_meta, projections = simind_projections_to_data(projections_header)
        if CT_header:
            CT = simind_CT_to_data(CT_header)
    object_correction_nets = []
    image_correction_nets = []
    if CT_header:
        CT_net = CTCorrectionNet(object_meta, image_meta, CT.unsqueeze(dim=0).to(device), device=device)
        object_correction_nets.append(CT_net)
        # fill this in later
    if PSF_options:
        psf_net = PSFCorrectionNet(object_meta, image_meta, PSF_options['collimator_slope'], PSF_options['collimator_intercept'],
                           kernel_size=61, kernel_dimensions = PSF_options['kernel_dimensions'], device=device)
        object_correction_nets.append(psf_net)
        # fill this in later
    fp_net = ForwardProjectionNet(object_correction_nets, image_correction_nets,
                                object_meta, image_meta, device=device)
    bp_net = BackProjectionNet(object_correction_nets, image_correction_nets,
                                object_meta, image_meta, device=device)
    if object_initial == 'ones':
        object_initial_array = torch.ones(object_meta.shape).unsqueeze(dim=0).to(device)
    osem_net = OSEMNet(object_initial_array, fp_net, bp_net)
    osem_net.set_image(projections.to(device))
    return osem_net