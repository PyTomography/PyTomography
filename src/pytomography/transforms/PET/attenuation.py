from __future__ import annotations
import torch
import pytomography
from pytomography.utils import rotate_detector_z, pad_proj
from pytomography.transforms import Transform
from pytomography.metadata import ObjectMeta, ProjMeta

def get_prob_of_detection_matrix(CT: torch.Tensor, dx: float) -> torch.tensor: 
    r"""Converts an attenuation map of :math:`\text{cm}^{-1}` to a probability of photon detection projection (detector pair oriented along x axis). Note that this requires the attenuation map to be at the energy of photons being emitted (511keV).

    Args:
        CT (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] corresponding to the attenuation coefficient in :math:`{\text{cm}^{-1}}`
        dx (float): Axial plane pixel spacing.

    Returns:
        torch.tensor: Tensor of size [batch_size, 1, Ly, Lz] corresponding to probability of photon being detected at a detector pairs oriented along the x axis.
    """
    return torch.exp(-torch.sum(CT * dx, axis=1)).unsqueeze(dim=1)

class PETAttenuationTransform(Transform):
    r"""proj2proj mapping used to model the effects of attenuation in PET.

    Args:
        CT (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] corresponding to the attenuation coefficient in :math:`{\text{cm}^{-1}}` at a photon energy of 511keV.
        device (str, optional): Pytorch device used for computation. If None, uses the default device `pytomography.device` Defaults to None.
	"""
    def __init__(self, CT: torch.Tensor) -> None:
        super(PETAttenuationTransform, self).__init__()
        self.CT = CT.to(self.device)
        
    def configure(self, object_meta: ObjectMeta, proj_meta: ProjMeta) -> None:
        """Function used to initalize the transform using corresponding object and projection metadata

        Args:
            object_meta (ObjectMeta): Object metadata.
            proj_meta (ProjMeta): Projection metadata.
        """
        super(PETAttenuationTransform, self).configure(object_meta, proj_meta)
        self.norm_proj = torch.zeros(self.proj_meta.padded_shape).to(self.device)
        CT = pad_proj(self.CT)
        for i, angle in enumerate(self.proj_meta.angles):
            self.norm_proj[i] = get_prob_of_detection_matrix(rotate_detector_z(CT, angle), self.object_meta.dx)
    
    @torch.no_grad()
    def forward(
		self,
		proj: torch.Tensor,
	) -> torch.Tensor:
        r"""Applies forward projection of attenuation modeling :math:`B:\mathbb{V} \to \mathbb{V}` to 2D PET projections.

        Args:
            proj (torch.Tensor): Tensor of size [batch_size, Ltheta, Lr, Lz] which transform is appplied to

        Returns:
            torch.Tensor: Tensor of size [batch_size, Ltheta, Lr, Lz]  corresponding to attenuation-corrected projections.
        """
        return proj*self.norm_proj.unsqueeze(dim=0)
    
    @torch.no_grad()
    def backward(
		self,
		proj: torch.Tensor,
		norm_constant: torch.Tensor | None = None,
	) -> torch.tensor:
        r"""Applies back projection of attenuation modeling :math:`B^T:\mathbb{V} \to \mathbb{V}` to 2D PET projections. Since the matrix is diagonal, its the ``backward`` implementation is identical to the ``forward`` implementation; the only difference is the optional ``norm_constant`` which is needed if one wants to normalize the back projection.

        Args:
            proj (torch.Tensor): Tensor of size [batch_size, Ltheta, Lr, Lz] which transform is appplied to
            norm_constant (torch.Tensor | None, optional): A tensor used to normalize the output during back projection. Defaults to None.

        Returns:
            torch.tensor: Tensor of size [batch_size, Ltheta, Lr, Lz]  corresponding to attenuation-corrected projections.
        """
        proj = proj*self.norm_proj.unsqueeze(dim=0)
        if norm_constant is not None:
            norm_constant = norm_constant*self.norm_proj.unsqueeze(dim=0)
            return proj, norm_constant
        else:
            return proj
    