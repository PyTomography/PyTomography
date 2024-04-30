from __future__ import annotations
import torch
import pytomography
from pytomography.transforms import Transform
from scipy.ndimage import map_coordinates
from torch.nn.functional import grid_sample

class DVFMotionTransform(Transform):
    def __init__(
        self,
        dvf_forward: torch.Tensor | None = None,
        dvf_backward: torch.Tensor | None = None,
        )-> None:
        """Object to object transform that uses a deformation vector field to deform an object. 

        Args:
            dvf_forward (torch.Tensor[Lx,Ly,Lz,3] | None, optional): Vector field correspond to forward transformation. If None, then no transformation is used. Defaults to None.
            dvf_backward (torch.Tensor[Lx,Ly,Lz,3] | None, optional): Vector field correspond to backward transformation. If None, then no transformation is used. Defaults to None. Defaults to None.
        """
        self.dvf_forward = dvf_forward.to(pytomography.device).to(pytomography.dtype)
        self.dvf_backward = dvf_backward.to(pytomography.device).to(pytomography.dtype)
        #self.dvf_forward_vol_ratio = self._get_vol_ratio(self.dvf_forward)
        #self.dvf_backward_vol_ratio = self._get_vol_ratio(self.dvf_backward)
        self.dvf_forward_vol_ratio = 1
        self.dvf_backward_vol_ratio = 1
        super(DVFMotionTransform, self).__init__()  ## go to the _init_ in Class Transform
  
    def _get_vol_ratio(self, DVF):
        xhat = torch.zeros((3,1,1,1)).to(pytomography.device)
        xhat[0] = 1
        yhat = torch.zeros((3,1,1,1)).to(pytomography.device)
        yhat[1] = 1
        zhat = torch.zeros((3,1,1,1)).to(pytomography.device)
        zhat[2] = 1
        v = DVF.permute((3,0,1,2))
        delv = torch.stack(torch.gradient(v, axis=(1,2,3)), axis=0) 
        vol_ratio = torch.abs((torch.cross(delv[0]+xhat,delv[1]+yhat)*(delv[2]+zhat)).sum(axis=0)).unsqueeze(0)
        return vol_ratio
  
    def _get_old_coordinates(self):
        """Obtain meshgrid of coordinates corresponding to the object

        Returns:
            torch.Tensor: Tensor of coordinates corresponding to input object
        """
        dim_x, dim_y, dim_z = self.object_meta.shape
        coordinates=torch.stack(torch.meshgrid(torch.arange(dim_x),torch.arange(dim_y), torch.arange(dim_z), indexing='ij')).permute((1,2,3,0)).to(pytomography.device).to(pytomography.dtype)
        return coordinates

    def _get_new_coordinates(self, old_coordinates: torch.Tensor, DVF: torch.Tensor):
        """Obtain the new coordinates of each voxel based on the DVF.

        Args:
            old_coordinates (torch.Tensor): Old coordinates of each voxel
            DVF (torch.Tensor): Deformation vector field.

        Returns:
            _type_: _description_
        """
        dimensions = torch.tensor(self.object_meta.shape).to(pytomography.device)
        new_coordinates = old_coordinates + DVF
        new_coordinates = 2/(dimensions-1)*new_coordinates - 1 
        return new_coordinates
        
    def _apply_dvf(self, DVF: torch.Tensor, vol_ratio, object_i: torch.Tensor):
        """Applies the deformation vector field to the object

        Args:
            DVF (torch.Tensor): Deformation vector field
            object_i (torch.Tensor): Old object.

        Returns:
            torch.Tensor: Deformed object.
        """
        old_coordinates = self._get_old_coordinates()
        new_coordinates = self._get_new_coordinates(old_coordinates, DVF)
        # Adjust for strecthcing of object
        return torch.nn.functional.grid_sample(object_i.unsqueeze(0).unsqueeze(0), new_coordinates.unsqueeze(0).flip(dims=[-1]), align_corners=True).squeeze() * vol_ratio

    def forward( 
        self,
        object_i: torch.Tensor,
    )-> torch.Tensor:
        """Forward transform of deformation vector field

        Args:
            object_i (torch.Tensor): Original object.

        Returns:
            torch.Tensor: Deformed object corresponding to forward transform.
        """
        if self.dvf_forward is None:
            return object_i
        else:
            return self._apply_dvf(self.dvf_forward, self.dvf_forward_vol_ratio, object_i)
    
    def backward( 
        self,
        object_i: torch.Tensor,
    )-> torch.Tensor:
        """Backward transform of deformation vector field

        Args:
            object_i (torch.Tensor): Original object.

        Returns:
            torch.Tensor: Deformed object corresponding to backward transform.
        """
        if self.dvf_backward is None:
            return object_i
        else:
            return self._apply_dvf(self.dvf_backward, self.dvf_backward_vol_ratio, object_i)