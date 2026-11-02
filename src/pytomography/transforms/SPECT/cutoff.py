"""A projection-to-projection transform that sets the projections to zero outside a mask."""
import torch
from pytomography.transforms import Transform
from pytomography.utils.spatial import pad_proj

class CutOffTransform(Transform):
    """Projection-to-projection transform that multiplies the projections by a mask, e.g. the detector's field of view.

    Masking is its own transpose, so :meth:`backward` is :meth:`forward`.

    Args:
        mask (torch.Tensor): The mask, [r, z] (e.g. :func:`pytomography.io.SPECT.fov_mask`), or [θ, r, z] for
            reconstructions without subsets; 1 inside, 0 outside.
    """
    def __init__(self, mask):
        super(CutOffTransform, self).__init__()
        self.padded_mask = pad_proj(mask)
        self.mask = mask
    @torch.no_grad()
    def forward(
		self,
		proj: torch.Tensor,
        padded: bool = True,
	) -> torch.tensor:
        """Multiplies the projections by the mask.

        Args:
            proj (torch.Tensor): The projections, [θ, r, z].
            padded (bool, optional): Whether the projections are padded along r, as they are inside
                :class:`~pytomography.projectors.SPECT.SPECTSystemMatrix`. Defaults to True.

        Returns:
            torch.Tensor: The masked projections.
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
        """The transpose of :meth:`forward`: the mask applied again (masking is diagonal, so it is its own transpose).

        Args:
            proj (torch.Tensor): The projections, [θ, r, z].
            padded (bool, optional): Whether the projections are padded along r. Defaults to True.

        Returns:
            torch.Tensor: The masked projections.
        """
        if padded:
            return proj * self.padded_mask
        else:
            return proj * self.mask