"""A projection-to-projection transform that adds a term to the projections."""
import torch
from pytomography.transforms import Transform

class AdditiveTermTransform(Transform):
    """Projection-to-projection transform that adds a term to the projections in forward projection.

    It is for forward models that include their additive term, such as
    :class:`~pytomography.projectors.SPECT.MonteCarloHybridSPECTSystemMatrix` (whose simulated projections it is
    applied to unpadded); :meth:`backward` returns the projections unchanged. The likelihoods take an additive term of
    their own (``additive_term``), which is how reconstructions usually model scatter.

    Args:
        additive_term (float | torch.Tensor): The term, a number or a tensor of the projections' shape.
    """
    def __init__(self, additive_term: float):
        super(AdditiveTermTransform, self).__init__()
        self.additive_term = additive_term
    @torch.no_grad()
    def forward(
		self,
		proj: torch.Tensor,
        padded: bool = True,
	) -> torch.tensor:
        """Adds the term to the projections.

        Args:
            proj (torch.Tensor): The projections.
            padded (bool, optional): Accepted for the interface of the projection transforms and not used: the term
                must have the shape of ``proj``. Defaults to True.

        Returns:
            torch.Tensor: ``proj`` plus the term.
        """
        return proj + self.additive_term
    @torch.no_grad()
    def backward(
		self,
		proj: torch.Tensor,
        padded: bool = True,
	) -> torch.tensor:
        """Returns the projections unchanged: back projection takes no additive term.

        Args:
            proj (torch.Tensor): The projections.
            padded (bool, optional): Not used. Defaults to True.

        Returns:
            torch.Tensor: ``proj``.
        """
        return proj # DON'T ADD HERE