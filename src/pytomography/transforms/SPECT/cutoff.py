from __future__ import annotations
import torch
from pytomography.transforms import Transform
import pydicom
from pytomography.utils import get_blank_below_above

class CutOffTransform(Transform):
    """proj2proj transformation used to set pixel values equal to zero at the first and last few z slices. This is often required when reconstructing DICOM data due to the finite field of view of the projection data, where additional axial slices are included on the top and bottom, with zero measured detection events. This transform is included in the system matrix, to model the sharp cutoff at the finite FOV.

        Args:
            proj (torch.tensor): Measured projection data.
    """
    def __init__(self, proj: torch.tensor | None = None, file_NM: str | None = None) -> None:
        super(CutOffTransform, self).__init__()
        if file_NM is not None:
            ds = pydicom.read_file(file_NM)
            dZ = (ds.DetectorInformationSequence[0].FieldOfViewDimensions[1]) / 2 / ds.PixelSpacing[1]
            central = (ds.Rows-1)/2
            lower = central - dZ
            upper = central + dZ
            self.blank_above = round(upper)+1
            self.blank_below = round(lower)-1
        else:
            self.blank_below, self.blank_above = get_blank_below_above(proj)
    @torch.no_grad()
    def forward(
		self,
		proj: torch.Tensor,
	) -> torch.tensor:
        r"""Forward projection :math:`B:\mathbb{V} \to \mathbb{V}` of the cutoff transform.

        Args:
            proj (torch.Tensor): Tensor of size [batch_size, Ltheta, Lr, Lz] which transform is appplied to

        Returns:
            torch.tensor: Original projections, but with certain z-slices equal to zero.
        """
        # Diagonal matrix so FP and BP is the same
        proj[:,:,:,:self.blank_below+1] = 0
        proj[:,:,:,self.blank_above:] = 0
        return proj
        
    @torch.no_grad()
    def backward(
		self,
		proj: torch.Tensor,
		norm_constant: torch.Tensor | None = None,
	) -> torch.tensor:
        r"""Back projection :math:`B^T:\mathbb{V} \to \mathbb{V}` of the cutoff transform. Since this is a diagonal matrix, the implementation is the same as forward projection, but with the optional `norm_constant` argument.

        Args:
            proj (torch.Tensor): Tensor of size [batch_size, Ltheta, Lr, Lz] which transform is appplied to
            norm_constant (torch.Tensor | None, optional): A tensor used to normalize the output during back projection. Defaults to None.

        Returns:
            torch.tensor: Original projections, but with certain z-slices equal to zero.
        """
        proj[:,:,:,:self.blank_below+1] = 0
        proj[:,:,:,self.blank_above:] = 0
        if norm_constant is not None:
            norm_constant[:,:,:,:self.blank_below+1] = 0
            norm_constant[:,:,:,self.blank_above:] = 0
            return proj, norm_constant
        else:
            return proj