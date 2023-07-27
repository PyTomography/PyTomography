from __future__ import annotations
import torch
from pytomography.transforms import Transform
from pytomography.utils import get_blank_below_above

class CutOffTransform(Transform):
    """im2im transformation used to set pixel values equal to zero at the first and last few z slices. This is often required when reconstructing DICOM data due to the finite field of view of the projection data, where additional axial slices are included on the top and bottom, with zero measured detection events. This transform is included in the system matrix, to model the sharp cutoff at the finite FOV.

        Args:
            image (torch.tensor): Measured image data.
    """
    def __init__(self, image: torch.tensor) -> None:
        super(CutOffTransform, self).__init__()
        self.blank_below, self.blank_above = get_blank_below_above(image)
    @torch.no_grad()
    def forward(
		self,
		image: torch.Tensor,
	) -> torch.tensor:
        r"""Forward projection :math:`B:\mathbb{V} \to \mathbb{V}` of the cutoff transform.

        Args:
            image (torch.Tensor): Tensor of size [batch_size, Ltheta, Lr, Lz] which transform is appplied to

        Returns:
            torch.tensor: Original image, but with certain z-slices equal to zero.
        """
        # Diagonal matrix so FP and BP is the same
        image[:,:,:,:self.blank_below] = 0
        image[:,:,:,self.blank_above:] = 0
        return image
        
    @torch.no_grad()
    def backward(
		self,
		image: torch.Tensor,
		norm_constant: torch.Tensor | None = None,
	) -> torch.tensor:
        r"""Back projection :math:`B^T:\mathbb{V} \to \mathbb{V}` of the cutoff transform. Since this is a diagonal matrix, the implementation is the same as forward projection, but with the optional `norm_constant` argument.

        Args:
            image (torch.Tensor): Tensor of size [batch_size, Ltheta, Lr, Lz] which transform is appplied to
            norm_constant (torch.Tensor | None, optional): A tensor used to normalize the output during back projection. Defaults to None.

        Returns:
            torch.tensor: Original image, but with certain z-slices equal to zero.
        """
        image[:,:,:,:self.blank_below] = 0
        image[:,:,:,self.blank_above:] = 0
        if norm_constant is not None:
            norm_constant[:,:,:,:self.blank_below] = 0
            norm_constant[:,:,:,self.blank_above:] = 0
            return image, norm_constant
        else:
            return image