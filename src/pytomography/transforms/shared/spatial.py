"""Rotation of an object about the z axis, as the SPECT system matrix uses to put each view's detector at +x."""
from __future__ import annotations
import torch
from torch.nn.functional import pad
from pytomography.transforms import Transform
from pytomography.utils.spatial import rotate_cached
import numpy as np

class RotationTransform(Transform):
	r"""Object-to-object transform that rotates an object about the z axis (the centre of each x-y plane).

	Every plane of constant z is rotated by the same angle, in degrees, anticlockwise in the (x, y) image as
	``kornia.geometry.transform.rotate`` defines it; :meth:`backward` rotates by the opposite angle. The SPECT system
	matrix rotates the object of view :math:`\beta` with ``backward(object, 270 - beta)`` so that the detector is at +x.

	Args:
		mode (str, optional): Interpolation, ``'bilinear'`` or ``'nearest'``. Defaults to ``'bilinear'``.
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
		r"""Rotates the object by ``angles`` degrees about the z axis.

		Args:
			object (torch.Tensor): The object, [Lx, Ly, Lz].
			angles (float | torch.Tensor): The rotation angle in degrees (a number or a one-element tensor).

		Returns:
			torch.Tensor: The rotated object, [Lx, Ly, Lz].
		"""
		return rotate_cached(object.permute(2,0,1).unsqueeze(0), angles, mode=self.mode).squeeze().permute(1,2,0)

	@torch.no_grad()
	def backward(
		self,
		object: torch.Tensor,
		angles: torch.Tensor
	) -> torch.Tensor:
		r"""Rotates the object by ``-angles`` degrees about the z axis: the inverse of :meth:`forward`.

		Args:
			object (torch.Tensor): The object, [Lx, Ly, Lz].
			angles (float | torch.Tensor): The angle of :meth:`forward` to undo, in degrees.

		Returns:
			torch.Tensor: The rotated object, [Lx, Ly, Lz].
		"""
		return rotate_cached(object.permute(2,0,1).unsqueeze(0), -angles, mode=self.mode).squeeze().permute(1,2,0)
