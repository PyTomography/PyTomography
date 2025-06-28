import torch
from pytomography.transforms import Transform
from pytomography.utils.spatial import pad_proj

class CutOffTransform(Transform):
    def __init__(self, mask):
        """Transform that cuts off the projection data outside of a certain region. This is used to remove the background from the projection data.
        Args:
            mask (torch.Tensor): Mask to cut off the projection data
        """
        super(CutOffTransform, self).__init__()
        self.padded_mask = pad_proj(mask)
        self.mask = mask
    @torch.no_grad()
    def forward(
		self,
		proj: torch.Tensor,
        padded: bool = True,
	) -> torch.tensor:
        """Cuts off the projection data outside of a certain region.
        Args:
            proj (torch.Tensor): Projection data
            padded (bool, optional): Whether or not the projection data is padded. Defaults to True.
        Returns:
            torch.Tensor: Projection data with cutoff applied
        """
        if padded:
            return proj * self.padded_mask
        else:
            return proj * self.mask
    @torch.no_grad()
    def backward(
		self,
		proj: torch.Tensor,
        padded: bool = True,
	) -> torch.tensor:
        """Returns the projection data without the cutoff.
        Args:
            proj (torch.Tensor): Projection data
            padded (bool, optional): Whether or not the projection data is padded. Defaults to True.
        Returns:
            torch.Tensor: Projection data without cutoff"""
        if padded:
            return proj * self.padded_mask
        else:
            return proj * self.mask