"""Least squares objectives, for CT and other data without Poisson statistics."""
from __future__ import annotations
import pytomography
import torch
from .likelihood import Likelihood
from pytomography.projectors import SystemMatrix

class NegativeMSELikelihood(Likelihood):
    r"""The negative squared error :math:`L(g|f) = -\frac{1}{2} \alpha \sum_i \left(g_i-(Hf+s)_i\right)^2`.

    :math:`g` is the measured data, :math:`H` the system matrix, :math:`s` the additive term, :math:`f` the object
    and :math:`\alpha` the scaling constant. It is negative so that gradient ascent algorithms maximise it.

    Args:
        system_matrix (SystemMatrix): The system matrix :math:`H`.
        projections (torch.Tensor | None, optional): The measured data :math:`g`. Defaults to None.
        additive_term (torch.Tensor, optional): The additive term :math:`s`, added after forward projection. Defaults
            to None.
        scaling_constant (float, optional): The constant :math:`\alpha`. Defaults to 1.
    """
    def __init__(
        self,
        system_matrix: SystemMatrix,
        projections: torch.Tensor | None = None,
        additive_term: torch.Tensor = None,
        scaling_constant: float = 1.0
        ) -> None:
        super().__init__(system_matrix, projections, additive_term)
        self.scaling_constant = scaling_constant
    def compute_gradient(
        self,
        object: torch.Tensor,
        subset_idx: int | None = None,
        norm_BP_subset_method: str = 'subset_specific'
        ) -> torch.Tensor:
        r"""The gradient :math:`\nabla_f L(g|f) = \alpha H_m^T \left(g_m - (H_m f + s_m)\right)` over subset :math:`m`.

        Args:
            object (torch.Tensor): The object :math:`f`, [Lx, Ly, Lz].
            subset_idx (int | None, optional): Subset index :math:`m`. Defaults to None: all the projections.
            norm_BP_subset_method (str, optional): Not used here; accepted for the interface of the likelihoods.
                Defaults to ``'subset_specific'``.

        Returns:
            torch.Tensor: The gradient, [Lx, Ly, Lz].
        """
        proj_subset = self._get_projection_subset(self.projections, subset_idx)
        self._forward_with_additive_term(object, subset_idx)
        return self.system_matrix.backward(proj_subset - self.projections_predicted , subset_idx) * self.scaling_constant
    
class SARTWeightedNegativeMSELikelihood(Likelihood):
    r"""The squared error weighted by the line lengths :math:`H 1`, whose gradient is SART's update direction.

    :class:`~pytomography.algorithms.SART` uses it: its gradient is
    :math:`H_m^T \left((g_m - (H_m f + s_m)) / H_m 1\right)`.

    Args:
        system_matrix (SystemMatrix): The system matrix :math:`H`.
        projections (torch.Tensor): The measured data :math:`g`.
        additive_term (torch.Tensor, optional): The additive term :math:`s`, added after forward projection. Defaults
            to None.
    """
    def __init__(
        self,
        system_matrix: SystemMatrix,
        projections: torch.Tensor,
        additive_term: torch.Tensor = None,
        ) -> None:
        super().__init__(system_matrix, projections, additive_term)
        
    def compute_gradient(
        self,
        object: torch.Tensor,
        subset_idx: int | None = None,
        norm_BP_subset_method: str = 'subset_specific'
        ) -> torch.Tensor:
        r"""The weighted gradient :math:`H_m^T \left((g_m - (H_m f + s_m)) / H_m 1\right)` over subset :math:`m`.

        Args:
            object (torch.Tensor): The object :math:`f`, [Lx, Ly, Lz].
            subset_idx (int | None, optional): Subset index :math:`m`. Defaults to None: all the projections.
            norm_BP_subset_method (str, optional): Not used here; accepted for the interface of the likelihoods.
                Defaults to ``'subset_specific'``.

        Returns:
            torch.Tensor: The gradient, [Lx, Ly, Lz].
        """
        proj_subset = self._get_projection_subset(self.projections, subset_idx)
        self._forward_with_additive_term(object, subset_idx)
        norm_FP = self.system_matrix.forward(object*0+1, subset_idx) # TODO: Slow implementation
        norm_BP = self._get_normBP(subset_idx)
        return self.system_matrix.backward((proj_subset - self.projections_predicted)/(norm_FP+pytomography.delta) , subset_idx)