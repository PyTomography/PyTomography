from __future__ import annotations
import torch
import torch.nn as nn
import pytomography
from pytomography.utils.helper_functions import rotate_detector_z, rev_cumsum, pad_object
from pytomography.transforms import Transform


def get_prob_of_detection_matrix(CT: torch.Tensor, dx: float) -> torch.tensor: 
	r"""Converts an attenuation map of :math:`\text{cm}^{-1}` to a probability of photon detection matrix (scanner at +x). Note that this requires the attenuation map to be at the energy of photons being emitted.

    Args:
        CT (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] corresponding to the attenuation coefficient in :math:`{\text{cm}^{-1}}
        dx (float): Axial plane pixel spacing.

    Returns:
        torch.tensor: Tensor of size [batch_size, Lx, Ly, Lz] corresponding to probability of photon being detected at detector at +x axis.
    """
	return torch.exp(-rev_cumsum(CT * dx))

class SPECTAttenuationTransform(Transform):
	r"""obj2obj transform used to model the effects of attenuation in SPECT.

		Args:
			CT (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] corresponding to the attenuation coefficient in :math:`{\text{cm}^{-1}}` at the photon energy corresponding to the particular scan
		"""
	def __init__(self, CT: torch.Tensor) -> None:
		super(SPECTAttenuationTransform, self).__init__()
		self.CT = CT.to(self.device)
                
	@torch.no_grad()
	def __call__(
		self,
		object_i: torch.Tensor,
		i: int, 
		norm_constant: torch.Tensor | None = None,
	) -> torch.Tensor:
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