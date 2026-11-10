"""Rotation of an object about the z axis, as the SPECT system matrix uses to put each view's detector at +x."""
from __future__ import annotations
import torch
from torch.nn.functional import pad
from pytomography.transforms import Transform
from pytomography.utils.spatial import rotate_cached, rotate_cached_transpose
import numpy as np

class RotationTransform(Transform):
	r"""Object-to-object transform that rotates an object about the z axis (the centre of each x-y plane).

	Every plane of constant z is rotated by the same angle, in degrees, as :func:`~pytomography.utils.spatial.rotate_cached`
	rotates an image [x, y] (anti-clockwise as displayed). :meth:`backward` is the exact transpose of :meth:`forward`, as
	for every transform (:func:`~pytomography.utils.spatial.rotate_cached_transpose`): not the rotation by the opposite
	angle, whose interpolation weights differ (to rotate back, call :meth:`forward` with the opposite angle). The SPECT
	system matrix rotates the object of view :math:`\beta` with ``forward(object, beta - 270)`` so that the detector is at
	+x, and back projects with ``backward(view, beta - 270)``.

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
		r"""The exact transpose of :meth:`forward` (``object``, ``angles``), for back projection: close to the rotation
		by ``-angles`` but not equal to it (see :func:`~pytomography.utils.spatial.rotate_cached_transpose`).

		Args:
			object (torch.Tensor): The object, [Lx, Ly, Lz].
			angles (float | torch.Tensor): The angle of the :meth:`forward` call to transpose, in degrees.

		Returns:
			torch.Tensor: The object, [Lx, Ly, Lz].
		"""
		return rotate_cached_transpose(object.permute(2,0,1).unsqueeze(0), angles, mode=self.mode)[0].permute(1,2,0)
