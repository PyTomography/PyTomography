from __future__ import annotations
from typing import Sequence, Callable
import torch
import pytomography
from pytomography.utils import rotate_detector_z, rev_cumsum, pad_object, unpad_object
from pytomography.transforms import Transform
from pytomography.io.shared import open_multifile
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.io.SPECT import get_attenuation_map_from_CT_slices

def get_prob_of_detection_matrix(attenuation_map: torch.Tensor, dx: float) -> torch.tensor: 
	r"""Converts an attenuation map of :math:`\text{cm}^{-1}` to a probability of photon detection matrix (scanner at +x). Note that this requires the attenuation map to be at the energy of photons being emitted.

	Args:
		attenuation_map (torch.tensor): Tensor of size [Lx, Ly, Lz] corresponding to the attenuation coefficient in :math:`{\text{cm}^{-1}}
		dx (float): Axial plane pixel spacing.

	Returns:
		torch.tensor: Tensor of size [Lx, Ly, Lz] corresponding to probability of photon being detected at detector at +x axis.
	"""
	return torch.exp(-rev_cumsum(attenuation_map * dx))

class SPECTAttenuationTransform(Transform):
	r"""obj2obj transform used to model the effects of attenuation in SPECT. This transform accepts either an ``attenuation_map`` (which must be aligned with the SPECT projection data) or a ``filepath`` consisting of folder containing CT DICOM files all pertaining to the same scan

	Args:
		attenuation_map (torch.tensor): Tensor of size [Lx, Ly, Lz] corresponding to the attenuation coefficient in :math:`{\text{cm}^{-1}}` at the photon energy corresponding to the particular scan
		filepath (Sequence[str]): Folder location of CT scan; all .dcm files must correspond to different slices of the same scan.
		mode (str): Mode used for extrapolation of CT beyond edges when aligning DICOM SPECT/CT data. Defaults to `'constant'`, which means the image is padded with zeros.
		assume_padded (bool): Assumes objects and projections fed into forward and backward methods are padded, as they will be in reconstruction algorithms
		HU2mu_technique (str): Technique to convert HU to attenuation coefficients. The default, 'from_table', uses a table of coefficients for bilinear curves obtained for a variety of common radionuclides. The technique 'from_cortical_bone_fit' looks for a cortical bone peak in the scan and uses that to obtain the bilinear coefficients. For phantom scans where the attenuation coefficient is always significantly less than bone, the corticol bone technique will still work, since the first part of the bilinear curve (in the air to water range) does not depend on the cortical bone fit. Alternatively, one can provide an arbitrary function here which takes in a 3D scan with units of HU and converts to mu.
	"""
	def __init__(
		self,
		attenuation_map: torch.Tensor | None = None,
		filepath: Sequence[str] | None = None,
		mode: str = 'constant',
		assume_padded: bool = True,
		HU2mu_technique: str | Callable = 'from_table'
		)-> None:
		super(SPECTAttenuationTransform, self).__init__()
		self.filepath = filepath
		self.mode = mode
		if attenuation_map is None and filepath is None:
			raise Exception("Please supply only one of `attenuation_map` or `filepath` as arguments")
		elif filepath is None:
			# Assumes CT is aligned with SPECT projections
			self.attenuation_map = attenuation_map.to(self.device)
		else:
			# TODO: offer support for all input types
			self.CT_unaligned_numpy = open_multifile(filepath)
			# Will then get aligned with projections when configured
		self.assume_padded = assume_padded
		self.HU2mu_technique = HU2mu_technique
	 
	def configure(
		self,
		object_meta: SPECTObjectMeta,
		proj_meta: SPECTProjMeta
	) -> None:
		"""Function used to initalize the transform using corresponding object and projection metadata

		Args:
			object_meta (SPECTObjectMeta): Object metadata.
			proj_meta (SPECTProjMeta): Projection metadata.
		"""
		super(SPECTAttenuationTransform, self).configure(object_meta, proj_meta)
		# Align CT with SPECT and rescale units TODO: If CT extends beyond boundaries
		if self.filepath is not None:
			self.attenuation_map = get_attenuation_map_from_CT_slices(self.filepath, proj_meta.filepath, proj_meta.index_peak, mode=self.mode, HU2mu_technique=self.HU2mu_technique)
				
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
		if self.assume_padded:
			attenuation_map = pad_object(self.attenuation_map)
		else:
			attenuation_map = self.attenuation_map.clone()
		norm_factor = get_prob_of_detection_matrix(rotate_detector_z(attenuation_map, self.proj_meta.angles[ang_idx]), self.object_meta.dx)
		return object_i*norm_factor

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
		if self.assume_padded:
			attenuation_map = pad_object(self.attenuation_map)
		else:
			attenuation_map = self.attenuation_map.clone()
		norm_factor = get_prob_of_detection_matrix(rotate_detector_z(attenuation_map, self.proj_meta.angles[ang_idx]), self.object_meta.dx)
		return object_i*norm_factor

	@torch.no_grad()
	def compute_average_prob_matrix(self):
		attenuation_map = pad_object(self.attenuation_map)
		average_norm_factor = torch.zeros(attenuation_map.shape).to(pytomography.device)
		for angle in self.proj_meta.angles:
			average_norm_factor += rotate_detector_z(get_prob_of_detection_matrix(rotate_detector_z(attenuation_map, angle), self.object_meta.dx), angle, negative=True)
		average_norm_factor /= len(self.proj_meta.angles)
		return unpad_object(average_norm_factor)


