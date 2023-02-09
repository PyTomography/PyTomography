import torch
import torch.nn as nn
from pytomography.utils.helper_functions import rotate_detector_z, rev_cumsum, pad_object

def get_prob_of_detection_matrix(CT, dx): 
	r"""Converts an attenuation map of :math:`\text{cm}^{-1}` to a probability of photon detection matrix (scanner at +x). Note that this requires the attenuation map to be at the energy of photons being emitted.

    Args:
        CT (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] of attenuation coefficients representing a CT scan
        dx (float): Pixel spacing in the transaxial plane

    Returns:
        torch.tensor: Tensor of size [batch_size, Lx, Ly, Lz] corresponding to probability of photon being detected at detector at +x axis
    """
	return torch.exp(-rev_cumsum(CT* dx))

class CTCorrectionNet(nn.Module):
	r"""Correction network used to correct for attenuation correction in projection operators. In particular, this network is used with other correction networks to model :math:`c` in :math:`\sum_i c_{ij} a_i` (forward projection) and :math:`\sum c_{ij} b_j` (back projection).

		Args:
			object_meta (ObjectMeta): Metadata for object space
			image_meta (ImageMeta): Metadata for image space
			CT (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] corresponding to the attenuation coefficient in :math:`{\text{cm}^{-1}}` at the photon energy corresponding to the particular scan
			device (str, optional): Pytorch computation device. Defaults to 'cpu'.
		"""
	def __init__(self, object_meta, image_meta, CT, device='cpu'):
		super(CTCorrectionNet, self).__init__()
		self.CT = CT.to(device)
		self.object_meta = object_meta
		self.image_meta = image_meta
		self.device = device
                
	@torch.no_grad()
	def forward(self, object_i, i, norm_constant=None):
		"""Applies attenuation correction to an object that's being detected on the right of its first axis

		Args:
			object_i (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] being projected along ``axis=1``.
			i (int): The projection index: used to find the corresponding angle in image space corresponding to ``object_i`` .
			norm_constant (torch.tensor, optional): A tensor used to normalize the output during back projection. Defaults to None.

		Returns:
			torch.tensor: Tensor of size [batch_size, Lx, Ly, Lz] such that projection of this tensor along the first axis corresponds to an attenuation corrected projection.
		"""
		CT = pad_object(self.CT)
		norm_factor = get_prob_of_detection_matrix(rotate_detector_z(CT, self.image_meta.angles[i]), self.object_meta.dx)
		if norm_constant is not None:
			norm_constant*=norm_factor
		return object_i*norm_factor