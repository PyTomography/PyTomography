from __future__ import annotations
from typing import Sequence, Callable
import torch
import pytomography
from pytomography.utils import rotate_detector_z, rev_cumsum, pad_object, unpad_object
from pytomography.transforms import Transform
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta

def get_prob_of_detection_matrix(attenuation_map: torch.Tensor, dx: float) -> torch.tensor: 
	r"""Converts an attenuation map (:math:`\text{mm}^{-1}`) to a probability of photon detection matrix (scanner at +x). Note that this requires the attenuation map to be at the energy of photons being emitted.

	Args:
		attenuation_map (torch.tensor): Tensor of size [Lx, Ly, Lz] corresponding to the attenuation coefficient in :math:`{\text{mm}^{-1}}`
		dx (float): Axial plane pixel spacing (mm).

	Returns:
		torch.tensor: Tensor of size [Lx, Ly, Lz] corresponding to probability of photon being detected at detector at +x axis.
	"""
	return torch.exp(-rev_cumsum(attenuation_map * dx))

class SPECTAttenuationTransform(Transform):
	r"""obj2obj transform used to model the effects of attenuation in SPECT.

	Args:
		attenuation_map (torch.tensor): Tensor of size [Lx, Ly, Lz] on the object's grid: the attenuation coefficient in :math:`{\text{mm}^{-1}}` at the photon energy of the energy window reconstructed (``proj_meta.energy_windows[i].photon_keV``), e.g. from :func:`pytomography.io.attenuation_map`.
		assume_padded (bool): Assumes objects and projections fed into forward and backward methods are padded, as they will be in reconstruction algorithms
		cache_probabilities (bool): If True, the rotated probability-of-detection map of each projection angle is computed once and kept on the device (memory: one padded object per angle, e.g. 0.2 GB for a 64x64x64 object and 96 angles, 1.6 GB for 128x128x128), instead of being recomputed from ``attenuation_map`` at every forward/backward call. The cache is discarded whenever a new tensor is assigned to ``attenuation_map`` or to ``proj_meta.angles`` (in-place modification of either tensor is not detected). Defaults to False.
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
		r"""Forward projection :math:`A:\mathbb{U} \to \mathbb{U}` of attenuation correction.

		Args:
			object_i (torch.tensor): Tensor of size [Lx, Ly, Lz] being projected along ``axis=0``.
			ang_idx (torch.Tensor): The projection indices: used to find the corresponding angle in projection space corresponding to each projection angle in ``object_i``.

		Returns:
			torch.tensor: Tensor of size [Lx, Ly, Lz] such that projection of this tensor along the first axis corresponds to an attenuation corrected projection.
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
		r"""Back projection :math:`A^T:\mathbb{U} \to \mathbb{U}` of attenuation correction. Since the matrix is diagonal, the implementation is the same as forward projection. The only difference is the optional normalization parameter.

		Args:
			object_i (torch.tensor): Tensor of size [Lx, Ly, Lz] being projected along ``axis=0``.
			ang_idx (torch.Tensor): The projection indices: used to find the corresponding angle in projection space corresponding to each projection angle in ``object_i``.
			norm_constant (torch.tensor, optional): A tensor used to normalize the output during back projection. Defaults to None.

		Returns:
			torch.tensor: Tensor of size [Lx, Ly, Lz] such that projection of this tensor along the first axis corresponds to an attenuation corrected projection.
		"""
		return object_i*self._prob_of_detection(ang_idx)

	@torch.no_grad()
	def compute_average_prob_matrix(self):
		attenuation_map = pad_object(self.attenuation_map)
		average_norm_factor = torch.zeros(attenuation_map.shape).to(pytomography.device)
		for angle in self.proj_meta.angles:
			average_norm_factor += rotate_detector_z(get_prob_of_detection_matrix(rotate_detector_z(attenuation_map, angle), self.object_meta.dx), angle, negative=True)
		average_norm_factor /= len(self.proj_meta.angles)
		return unpad_object(average_norm_factor)


