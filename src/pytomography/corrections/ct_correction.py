import torch
import torch.nn as nn
from pytomography.utils.helper_functions import rotate_detector_z, rev_cumsum

def get_prob_of_detection_matrix(CT, dx): 
	"""Converts a CT scan in units of cm^-1 to a probability of photon detection matrix (scanner at +x)

    Args:
        CT (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] of attenuation coefficients representing a CT scan
        dx (float): Pixel spacing in the transaxial plane

    Returns:
        torch.tensor: Tensor of size [batch_size, Lx, Ly, Lz] corresponding to probability of photon being detected at detector at +x axis
    """
	return torch.exp(-rev_cumsum(CT* dx))

class CTCorrectionNet(nn.Module):
	def __init__(self, object_meta, image_meta, CT, store_in_memory=False, device='cpu'):
		"""Correction network used to correct for attenuation correction in projection operators

		Args:
			object_meta (ObjectMeta): Metadata for object space
			image_meta (ImageMeta): Metadata for image space
			CT (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] corresponding to the attenuation coefficient in cm^-1 at
			the photon energy corresponding to the particular scan
			store_in_memory (bool, optional): Stores all rotated CT scans in memory on computer. May speed up computation. Defaults to False.
			device (str, optional): Pytorch computation device. Defaults to 'cpu'.
		"""
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
		"""Applies attenuation correction to an object that's being detected on the right of its first axis

		Args:
			object_i (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] being projected along its first axis
			i (int): The projection index: used to find the corresponding angle in image space corresponding to object i
			norm_constant (torch.tensor, optional): A tensor used to normalize the output during back projection. Defaults to None.

		Returns:
			torch.tensor: Tensor of size [batch_size, Lx, Ly, Lz] such that projection of this tensor along the first axis corresponds to
			an attenuation corrected projection.
		"""
		if self.store_in_memory:
			norm_factor = self.probability_matrices[i]
		else:
			norm_factor = get_prob_of_detection_matrix(rotate_detector_z(self.CT, self.image_meta.angles[i]), self.object_meta.dx).to(self.device)
		if norm_constant is not None:
			norm_constant*=norm_factor
		return object_i*norm_factor