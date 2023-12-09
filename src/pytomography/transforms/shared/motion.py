from __future__ import annotations
import torch
import pytomography
from pytomography.transforms import Transform
from scipy.ndimage import map_coordinates

class DVFMotionTransform(Transform):
	def __init__(
		self,
		dvf_forward: torch.Tensor | None = None,
		dvf_backward: torch.Tensor | None = None,
		)-> None:
		self.dvf_forward = dvf_forward
		self.dvf_backward = dvf_backward
		super(DVFMotionTransform, self).__init__()  ## go to the _init_ in Class Transform
  
	def _get_coordinates(self):  
		dim_x, dim_y, dim_z = self.object_meta.shape
		coordinates=torch.stack(torch.meshgrid(torch.arange(dim_x),torch.arange(dim_y), torch.arange(dim_z), indexing='ij')).permute((1,2,3,0)).reshape(-1, 3)
		return coordinates

	def _generate_new_coordinates(self, coordinates, DVF, reverse: bool = False):
		sign = -1 if reverse else 1 
		dim_x, dim_y, dim_z = self.object_meta.shape
		coordinates = coordinates.view(dim_x, dim_y, dim_z, 3)
		return (coordinates + sign*DVF).flatten(0,2)
		
	def _apply_dvf(self, DVF, object_i):
		coordinates = self._get_coordinates().cpu()
		xyz_array = self._generate_new_coordinates(coordinates, DVF)
		return torch.tensor(map_coordinates(object_i[0].cpu(), xyz_array.T, order=1, mode="nearest")).reshape(self.object_meta.shape).unsqueeze(0).to(pytomography.device)

	def forward( 
		self,
		object_i: torch.Tensor,
	)-> torch.Tensor:
		return self._apply_dvf(self.dvf_forward, object_i)
	
	def backward( 
		self,
		object_i: torch.Tensor,
	)-> torch.Tensor:
		return self._apply_dvf(self.dvf_backward, object_i)