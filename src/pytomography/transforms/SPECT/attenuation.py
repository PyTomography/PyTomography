"""SPECT attenuation: the probability that a photon emitted in each voxel reaches the detector."""
from __future__ import annotations
from typing import Sequence, Callable
import torch
import pytomography
from pytomography.utils import rotate_detector_z, rev_cumsum, pad_object, unpad_object
from pytomography.transforms import Transform
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta

def get_prob_of_detection_matrix(attenuation_map: torch.Tensor, dx: float) -> torch.tensor:
	r"""The probability that a photon from each voxel reaches a detector at +x, from an attenuation map in 1/mm.

	It is :math:`\exp(-\sum \mu \, dx)` over the voxels between the voxel and the +x edge of the map, the voxel itself
	counted by half (the photon starts at its centre). The map must be at the photon energy of the emitted photons.

	Args:
		attenuation_map (torch.Tensor): Attenuation coefficients in 1/mm, [Lx, Ly, Lz].
		dx (float): Voxel size along x, in mm.

	Returns:
		torch.Tensor: The probabilities, [Lx, Ly, Lz].
	"""
	return torch.exp(-rev_cumsum(attenuation_map * dx))

class SPECTAttenuationTransform(Transform):
	r"""Object-to-object transform that models attenuation in SPECT: each voxel times its probability of detection.

	For each view, the object (rotated so that the detector is at +x) is multiplied by the probability that a photon
	from each voxel reaches the detector, :func:`get_prob_of_detection_matrix` of the attenuation map rotated the same
	way. The operator is diagonal, so :meth:`backward` is :meth:`forward`.

	Args:
		attenuation_map (torch.Tensor): Attenuation coefficients in 1/mm, [Lx, Ly, Lz] on the object's grid, at the
			photon energy of the energy window reconstructed (``proj_meta.energy_windows[i].photon_energy``), e.g. from
			:func:`pytomography.io.attenuation_map`.
		assume_padded (bool, optional): The objects given to :meth:`forward` and :meth:`backward` are padded for
			rotation, as they are inside :class:`~pytomography.projectors.SPECT.SPECTSystemMatrix`. Defaults to True.
		cache_probabilities (bool, optional): Compute the rotated probability map of each view once and keep it on the
			device (one padded object per view, e.g. 0.2 GB for a 64x64x64 object and 96 views, 1.6 GB for 128x128x128),
			instead of recomputing it from ``attenuation_map`` at every call. The cache is dropped whenever a new tensor
			is assigned to ``attenuation_map`` or to ``proj_meta.angles`` (changes made in place are not detected).
			Defaults to False.
	"""
	def __init__(
		self,
		attenuation_map: torch.Tensor,
		assume_padded: bool = True,
		cache_probabilities: bool = False,
		)-> None:
		super(SPECTAttenuationTransform, self).__init__()
		self.cache_probabilities = cache_probabilities
		self._prob_cache = {}
		self._prob_cache_src = (None, None)   # (attenuation_map, proj_meta.angles) the cache was built from
		self.attenuation_map = attenuation_map.to(self.device)
		self.assume_padded = assume_padded

	@torch.no_grad()
	def forward(
		self,
		object_i: torch.Tensor,
		ang_idx: torch.Tensor, 
	)-> torch.Tensor:
		r"""Applies attenuation :math:`A:\mathbb{U} \to \mathbb{U}` to the object of one view.

		Args:
			object_i (torch.Tensor): The object rotated for the view, [Lx, Ly, Lz] (padded), projected along x.
			ang_idx (int): Index of the view in ``proj_meta.angles``.

		Returns:
			torch.Tensor: The object times the probability of detection, [Lx, Ly, Lz]; its sum along x is the attenuated
			projection.
		"""
		return object_i*self._prob_of_detection(ang_idx)

	def _prob_of_detection(self, ang_idx: torch.Tensor) -> torch.Tensor:
		"""Probability-of-detection map in the frame rotated to angle ``ang_idx`` (scanner at +x), from the cache when ``cache_probabilities`` is set and ``attenuation_map`` and ``proj_meta.angles`` are the tensors the cache was built from."""
		if self.cache_probabilities:
			src_map, src_angles = self._prob_cache_src
			if src_map is not self.attenuation_map or src_angles is not self.proj_meta.angles:
				self._prob_cache = {}
				self._prob_cache_src = (self.attenuation_map, self.proj_meta.angles)
			key = int(ang_idx)
			norm_factor = self._prob_cache.get(key)
			if norm_factor is not None:
				return norm_factor
		if self.assume_padded:
			attenuation_map = pad_object(self.attenuation_map)
		else:
			attenuation_map = self.attenuation_map.clone()
		# CPU copy of the angle: converting a 0-d device tensor to a Python float inside the rotation would synchronise with the device
		norm_factor = get_prob_of_detection_matrix(rotate_detector_z(attenuation_map, self._host_angles()[int(ang_idx)]), self.object_meta.dx)
		if self.cache_probabilities:
			self._prob_cache[key] = norm_factor
		return norm_factor

	def _host_angles(self) -> torch.Tensor:
		"""CPU copy of ``proj_meta.angles`` (made once per ``angles`` tensor, same dtype so the rotation angle ``270 - angle`` is computed exactly as before)."""
		angles = self.proj_meta.angles
		if getattr(self, '_host_angles_src', None) is not angles:
			self._host_angles_cache = angles.cpu()
			self._host_angles_src = angles
		return self._host_angles_cache

	@torch.no_grad()
	def backward(
		self,
		object_i: torch.Tensor,
		ang_idx: torch.Tensor, 
	) -> torch.Tensor:
		r"""Applies the transpose :math:`A^T:\mathbb{U} \to \mathbb{U}`, which is :meth:`forward` (the operator is diagonal).

		Args:
			object_i (torch.Tensor): A back projected view, [Lx, Ly, Lz] (padded), in the view's rotated frame.
			ang_idx (int): Index of the view in ``proj_meta.angles``.

		Returns:
			torch.Tensor: The object times the probability of detection, [Lx, Ly, Lz].
		"""
		return object_i*self._prob_of_detection(ang_idx)

	@torch.no_grad()
	def compute_average_prob_matrix(self):
		"""The probability of detection averaged over every view, in the object's own frame.

		Returns:
			torch.Tensor: The average probability, [Lx, Ly, Lz], on ``pytomography.device``.
		"""
		attenuation_map = pad_object(self.attenuation_map)
		average_norm_factor = torch.zeros(attenuation_map.shape).to(pytomography.device)
		for angle in self.proj_meta.angles:
			average_norm_factor += rotate_detector_z(get_prob_of_detection_matrix(rotate_detector_z(attenuation_map, angle), self.object_meta.dx), angle, negative=True)
		average_norm_factor /= len(self.proj_meta.angles)
		return unpad_object(average_norm_factor)


