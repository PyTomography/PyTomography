import torch
from pytomography.transforms import Transform
from pytomography.utils.spatial import pad_proj

class CutOffTransform(Transform):
    def __init__(self, mask):
        super(CutOffTransform, self).__init__()
        self.mask = pad_proj(mask)
    @torch.no_grad()
    def forward(
		self,
		proj: torch.Tensor,
    ang_idx: torch.Tensor,
	) -> torch.tensor:
        return proj * self.mask
    @torch.no_grad()
    def backward(
		self,
		proj: torch.Tensor,
    ang_idx: torch.Tensor,
	) -> torch.tensor:
        return proj * self.mask