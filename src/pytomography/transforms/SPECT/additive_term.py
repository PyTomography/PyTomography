import torch
from pytomography.transforms import Transform

class AdditiveTermTransform(Transform):
    def __init__(self, additive_term: float):
        """Transform that adds an additive term to the projection data. This is used to add the scatter and total projections together.
        Args:
            additive_term (float): Additive term to add to the projection data
        """
        super(AdditiveTermTransform, self).__init__()
        self.additive_term = additive_term
    @torch.no_grad()
    def forward(
		self,
		proj: torch.Tensor,
        padded: bool = True,
	) -> torch.tensor:
        """Adds an additive term to the projection data.
        Args:
            proj (torch.Tensor): Projection data
            padded (bool, optional): Whether or not the projection data is padded. Defaults to True.
        Returns:
            torch.Tensor: Projection data with additive term added
        """
        return proj + self.additive_term
    @torch.no_grad()
    def backward(
		self,
		proj: torch.Tensor,
        padded: bool = True,
	) -> torch.tensor:
        """Returns the projection data without the additive term.
        Args:
            proj (torch.Tensor): Projection data
            padded (bool, optional): Whether or not the projection data is padded. Defaults to True.
        Returns:
            torch.Tensor: Projection data without additive term
        """
        return proj # DON'T ADD HERE