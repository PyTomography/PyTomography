from __future__ import annotations
import torch
from torch.nn.functional import pad
from pytomography.transforms import Transform
from kornia.geometry.transform import rotate
import numpy as np

class RotationTransform(Transform):
	r"""obj2obj transform used to rotate an object to angle :math:`\beta` in the DICOM reference frame. (Note that an angle of )

	Args:
		mode (str): Interpolation mode used in the rotation.
	"""
	def __init__(
		self,
		mode: str = 'bilinear'
		)-> None:
		super(RotationTransform, self).__init__()
		self.mode = mode
				
	@torch.no_grad()
	def forward(
		self,
		object: torch.Tensor,
		angles: torch.Tensor, 
	)-> torch.Tensor:
		r"""Rotates an object to angle :math:`\beta` in the DICOM reference frame. Note that the scanner angle :math:`\beta` is related to :math:`\phi` (azimuthal angle) by :math:`\phi = 3\pi/2 - \beta`. 

		Args:
			object (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] being rotated.
			angles (torch.Tensor):  Tensor of size [batch_size] corresponding to the rotation angles.

		Returns:
			torch.tensor: Tensor of size [batch_size, Lx, Ly, Lz] where each element in the batch dimension is rotated by the corresponding angle.
		"""
		return rotate(object.permute(0,3,1,2), angles, mode=self.mode).permute(0,2,3,1)

	@torch.no_grad()
	def backward(
		self,
		object: torch.Tensor,
		angles: torch.Tensor
	) -> torch.Tensor:
		r"""Forward projection :math:`A:\mathbb{U} \to \mathbb{U}` of attenuation correction.

		Args:
			object (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] being rotated.
			angles (torch.Tensor):  Tensor of size [batch_size] corresponding to the rotation angles.

		Returns:
			torch.tensor: Tensor of size [batch_size, Lx, Ly, Lz] where each element in the batch dimension is rotated by the corresponding angle.
		"""
		return rotate(object.permute(0,3,1,2), -angles, mode=self.mode).permute(0,2,3,1)
