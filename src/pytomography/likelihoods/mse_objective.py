from __future__ import annotations
import pytomography
import torch
from .likelihood import Likelihood
from pytomography.projectors import SystemMatrix

class NegativeMSELikelihood(Likelihood):
    r"""Negative mean squared error likelihood function :math:`L(g|f) = -\frac{1}{2} \alpha \sum_i \left(g_i-(Hf)_i\right)^2` where :math:`g` is the acquired data, :math:`H` is the system matrix, :math:`f` is the object being reconstructed, and :math:`\alpha` is the scaling constant. The negative is taken so that the it works in gradient ascent (as opposed to descent) algorithms

    Args:
        system_matrix (SystemMatrix): The system matrix modeling the particular system whereby the projections were obtained
        projections (torch.Tensor): Acquired data
        additive_term (torch.Tensor, optional): Additional term added after forward projection by the system matrix. This term might include things like scatter and randoms. Defaults to None.
        additive_term_variance_estimate (torch.tensor, optional): Variance estimate of the additive term. If none, then uncertainty estimation does not include contribution from the additive term. Defaults to None.
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
        r"""Computes the gradient for the mean squared error objective function given by :math:`\nabla_f L(g|f) =  H^T \left(g-Hf\right)`. 

        Args:
            object (torch.Tensor): Object :math:`f` on which the likelihood is computed
            subset_idx (int | None, optional): Specifies the subset for forward/back projection. If none, then forward/back projection is done over all subsets, and the entire projections :math:`g` are used. Defaults to None.
            norm_BP_subset_method (str, optional): Specifies how :math:`H^T 1` is calculated when subsets are used. If 'subset_specific', then uses :math:`H_m^T 1`. If `average_of_subsets`, then uses the average of all :math:`H_m^T 1`s for any given subset (scaled to the relative size of the subset if subsets are not equal size). Defaults to 'subset_specific'.

        Returns:
            torch.Tensor: The gradient of the Poisson likelihood.
        """
        proj_subset = self._get_projection_subset(self.projections, subset_idx)
        additive_term_subset = self._get_projection_subset(self.additive_term, subset_idx)
        self.projections_predicted = self.system_matrix.forward(object, subset_idx) + additive_term_subset
        return self.system_matrix.backward(proj_subset - self.projections_predicted , subset_idx) * self.scaling_constant
    
class SARTWeightedNegativeMSELikelihood(Likelihood):
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
        r"""Computes the gradient for the mean squared error objective function given by :math:`\nabla_f L(g|f) =  H^T \left(g-Hf\right)`. 

        Args:
            object (torch.Tensor): Object :math:`f` on which the likelihood is computed
            subset_idx (int | None, optional): Specifies the subset for forward/back projection. If none, then forward/back projection is done over all subsets, and the entire projections :math:`g` are used. Defaults to None.
            norm_BP_subset_method (str, optional): Specifies how :math:`H^T 1` is calculated when subsets are used. If 'subset_specific', then uses :math:`H_m^T 1`. If `average_of_subsets`, then uses the average of all :math:`H_m^T 1`s for any given subset (scaled to the relative size of the subset if subsets are not equal size). Defaults to 'subset_specific'.

        Returns:
            torch.Tensor: The gradient of the Poisson likelihood.
        """
        proj_subset = self._get_projection_subset(self.projections, subset_idx)
        additive_term_subset = self._get_projection_subset(self.additive_term, subset_idx)
        self.projections_predicted = self.system_matrix.forward(object, subset_idx) + additive_term_subset
        norm_FP = self.system_matrix.forward(object*0+1, subset_idx) # TODO: Slow implementation
        norm_BP = self._get_normBP(subset_idx)
        return self.system_matrix.backward((proj_subset - self.projections_predicted)/(norm_FP+pytomography.delta) , subset_idx)