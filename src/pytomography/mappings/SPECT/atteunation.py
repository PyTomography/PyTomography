from __future__ import annotations
import torch
import torch.nn as nn
from pytomography.utils.helper_functions import rotate_detector_z, rev_cumsum, pad_object
from pytomography.mappings import MapNet


def get_prob_of_detection_matrix(CT: torch.Tensor, dx: float) -> torch.tensor: 
	r"""Converts an attenuation map of :math:`\text{cm}^{-1}` to a probability of photon detection matrix (scanner at +x). Note that this requires the attenuation map to be at the energy of photons being emitted.

    Args:
        CT (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] corresponding to the attenuation coefficient in :math:`{\text{cm}^{-1}}
        dx (float): Axial plane pixel spacing.

    Returns:
        torch.tensor: Tensor of size [batch_size, Lx, Ly, Lz] corresponding to probability of photon being detected at detector at +x axis.
    """
	return torch.exp(-rev_cumsum(CT * dx))

class SPECTAttenuationNet(MapNet):
	r"""obj2obj mapping used to model the effects of attenuation in SPECT.

		Args:
			CT (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] corresponding to the attenuation coefficient in :math:`{\text{cm}^{-1}}` at the photon energy corresponding to the particular scan
			device (str, optional): Pytorch computation device. Defaults to 'cpu'.
		"""
	def __init__(self, CT: torch.Tensor, device: str = 'cpu') -> None:
		super(SPECTAttenuationNet, self).__init__(device)
		self.CT = CT.to(device)
                
	@torch.no_grad()
	def forward(
		self,
		object_i: torch.Tensor,
		i: int, 
		norm_constant: torch.Tensor | None = None,
	) -> torch.tensor:
		"""Applies attenuation modeling to an object that's being detected on the right of its first axis.

		Args:
			object_i (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] being projected along ``axis=1``.
			i (int): The projection index: used to find the corresponding angle in image space corresponding to ``object_i``. In particular, the x axis (tensor `axis=1`) of the object is aligned with the detector at angle i.
			norm_constant (torch.tensor, optional): A tensor used to normalize the output during back projection. Defaults to None.

		Returns:
			torch.tensor: Tensor of size [batch_size, Lx, Ly, Lz] such that projection of this tensor along the first axis corresponds to an attenuation corrected projection.
		"""
		CT = pad_object(self.CT)
		norm_factor = get_prob_of_detection_matrix(rotate_detector_z(CT, self.image_meta.angles[i]), self.object_meta.dx)
		object_i*=norm_factor
		if norm_constant is not None:
			norm_constant*=norm_factor
			return object_i, norm_constant
		else:
			return object_i