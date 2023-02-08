import torch
import torch.nn as nn
import numpy as np
from pytomography.projections import ForwardProjectionNet, BackProjectionNet
from pytomography.corrections import CTCorrectionNet, PSFCorrectionNet
from pytomography.io import simind_projections_to_data, simind_CT_to_data, dicom_projections_to_data, dicom_CT_to_data

class OSEMNet(nn.Module):
    r"""Network used to run OSEM reconstruction: :math:`f_i^{(n+1)} = \frac{f_i^{(n)}}{\sum_j c_{ij} + \beta \frac{\partial V}{\partial f_r}|_{f_i=f_i^{(n)}}} \sum_j c_{ij}\frac{g_j}{\sum_i c_{ij}f_i^{(n)}}`. Initializer initializes the reconstruction algorithm with the initial object guess :math:`f_i^{(0)}`, forward and back projections used (i.e. networks to compute :math:`\sum_i c_{ij} a_i` and :math:`\sum_j c_{ij} b_j`), and prior for Bayesian corrections. Note that OSEMNet uses the one step late (OSL algorithm to compute priors during reconstruction. Once the class is initialized, the number of iterations and subsets are specified at recon time when the forward method is called.

        Args:
            object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): represents the initial object guess :math:`f_i^{(0)}` for the algorithm in object space
            forward_projection_net (ForwardProjectionNet): the forward projection network used to compute :math:`\sum_{i} c_{ij} a_i` where :math:`a_i` is the object being forward projected.
            back_projection_net (BackProjectionNet): the back projection network used to compute :math:`\sum_{j} c_{ij} b_j` where :math:`b_j` is the image being back projected.
            prior (Prior, optional): the Bayesian prior; computes :math:`\beta \frac{\partial V}{\partial f_r}|_{f_i=f_i^{\text{old}}}`. If ``None``, then this term is 0. Defaults to None
        """
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
        """Returns a list of arrays; each array contains indices, corresponding
            to projection numbers, that are used in ordered-subsets. For example,
            ``get_subsets_splits(2, 6)`` would return ``[[0,2,4],[1,3,5]]``.
        Args:
            n_subsets (int): number of subsets used in OSEM 
            n_angles (int): total number of projections

        Returns:
            list: list of index arrays for each subset
        """
        
        indices = np.arange(n_angles).astype(int)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        return subset_indices_array

    def set_image(self, image):
        """Sets the projection data which is to be reconstructed

        Args:
            image (torch.tensor[batch_size, Ltheta, Lr, Lz]): image data
        """
        self.image = image

    def set_prior(self, prior):
        """Sets the prior used for Bayesian modeling

        Args:
            prior (Prior): The prior class corresponding to a particular model
        """
        self.prior = prior
        self.prior.set_kernel(self.forward_projection_net.object_meta)

    def forward(self, n_iters, n_subsets, comparisons=None, delta=1e-11):
        """Performs the reconstruction using `n_iters` iterations and `n_subsets` subsets.

        Args:
            n_iters (int): _description_
            n_subsets (int): _description_
            comparisons (FIX, optional): FIX. Defaults to None.
            delta (_type_, optional): Used to prevent division by zero when calculating ratio, defaults to 1e-11.

        Returns:
            torch.tensor[batch_size, Lx, Ly, Lz]: reconstructed object
        """
        subset_indices_array = self.get_subset_splits(n_subsets, self.image.shape[1])
        # Scale beta by number of subsets
        if self.prior:
            self.prior.set_beta_scale(1/n_subsets)
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

def get_osem_net(projections_header, object_initial='ones', CT_header=None, psf_meta=None, file_type='simind', prior=None, device='cpu'):
    """Function used to obtain an `OSEMNet` given projection data and corrections one wishes to use.

    Args:
        projections_header (str): Path to projection header data (in some modalities, this is also the data path i.e. DICOM). Data from this file is used to set the dimensions of the object [batch_size, Lx, Ly, Lz] and the image [batch_size, Ltheta, Lr, Lz] and the projection data one wants to reconstruct.
        object_initial (str or torch.tensor, optional): Specifies initial object. In the case of `'ones'`, defaults to a tensor of shape [batch_size, Lx, Ly, Lz] containing all ones. Otherwise, takes in a specific initial guess. Defaults to 'ones'.
        CT_header (str or list, optional): File path pointing to CT data file or files. Defaults to None.
        psf_meta (PSFMeta, optional): Metadata specifying PSF correction parameters, such as collimator slope and intercept. Defaults to None.
        file_type (str, optional): The file type of the `projections_header` file. Options include simind output and DICOM. Defaults to 'simind'.
        prior (Prior, optional): The prior used during reconstruction. If `None`, use no prior. Defaults to None.
        device (str, optional): The device used in pytorch for reconstruction. Graphics card can be used. Defaults to 'cpu'.

    Returns:
        OSEMNet: An initialized OSEMNet, ready to perform reconstruction.
    """
    if file_type=='simind':
        object_meta, image_meta, projections = simind_projections_to_data(projections_header)
        if CT_header is not None:
            CT = simind_CT_to_data(CT_header)
    elif file_type=='dicom':
        object_meta, image_meta, projections = dicom_projections_to_data(projections_header)
        if CT_header is not None:
            CT = dicom_CT_to_data(CT_header, projections_header)
    object_correction_nets = []
    image_correction_nets = []
    if CT_header is not None:
        CT_net = CTCorrectionNet(object_meta, image_meta, CT.unsqueeze(dim=0).to(device), device=device)
        object_correction_nets.append(CT_net)
        # fill this in later
    if psf_meta is not None:
        psf_net = PSFCorrectionNet(object_meta, image_meta, psf_meta, device=device)
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
    if prior is not None:
        prior.set_device(device)
        osem_net.set_prior(prior)
    return osem_net


# RENAME ALL THIS LATER
class CompareToNumber():
    def __init__(self, number, mask, norm_factor=None):
        self.number = number
        self.mask = mask
        self.biass = []
        self.vars = []
        self.norm_factor = norm_factor
    def compare(self, prediction):
        prediction = torch.clone(prediction)
        if self.norm_factor:
            prediction *= self.norm_factor 
        self.biass.append(torch.mean(prediction[self.mask] - self.number).item())
        self.vars.append(torch.var(prediction[self.mask] - self.number).item())