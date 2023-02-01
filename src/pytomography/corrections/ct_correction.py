import torch
import torch.nn as nn
from pytomography.utils.helper_functions import rotate_detector_z, rev_cumsum

# CT of size [batch_size, Lx, Ly, Lz]
def get_prob_of_detection_matrix(CT, dx): 
    return torch.exp(-rev_cumsum(CT* dx))

class CTCorrectionNet(nn.Module):
    '''Apply attenuation correction to an object.
    
    Given projection along the x-axis, an object is modified
    using an array of CT values such that subsequent projection
    yields an appropriate image
    '''
    def __init__(self, object_meta, image_meta, CT, store_in_memory=False, device='cpu'):
        '''
        Parameters
        --------
        object_meta : ObjectMeta
          Metadata for the object 
        image_meta : ImageMeta
          Metadata for the corresponding image
        CT : (batch_size, Lx, Ly, Lz) torch.tensor
          Attenuation coefficient in cm^-1 corresponding to the photon energy of the object emission data
        device : str
          The device used by pytorch where the network is placed.
        '''
        super(CTCorrectionNet, self).__init__()
        self.CT = CT
        self.object_meta = object_meta
        self.image_meta = image_meta
        self.device = device
        self.store_in_memory = store_in_memory
        if self.store_in_memory:
            self.probability_matrices = []
            for i, angle in enumerate(self.image_meta.angles):
                self.probability_matrices.append(get_prob_of_detection_matrix(rotate_detector_z(self.CT, angle), self.object_meta.dx).to(self.device))
    @torch.no_grad()
    def forward(self, object_i, i, norm_constant=None):
        '''Modify object using attenuation correction.

        Parameters
        --------
        object_i : (batch_size, Lx, Ly, Lz) torch.tensor
          This object is such that summation along the x-axis yields the ith projection in
          the image
        i: number
          The projection index
        norm_constant: (batch_size, Lx, Ly, Lz) torch.tensor. Default: None
          If true, modify the norm_constant argument by the normalization factor
          used to scale object_i. This is useful during back projection, where division by
          a normalization constant is required.

        Returns:
          Pytorch tensor of size [batch_size, Lx, Ly, Lz] cor
        '''
        if self.store_in_memory:
            norm_factor = self.probability_matrices[i]
        else:
            norm_factor = get_prob_of_detection_matrix(rotate_detector_z(self.CT, self.image_meta.angles[i]), self.object_meta.dx).to(self.device)
        if norm_constant is not None:
            norm_constant*=norm_factor
        return object_i*norm_factor