from __future__ import annotations
from typing import Callable
import pytomography
import torch
from .likelihood import Likelihood

class PoissonLogLikelihood(Likelihood):
    r"""The log-likelihood function for Poisson random variables. The likelihood is given by :math:`L(g|f) = \sum_i g_i [\ln(Hf)]_i - [Hf]_i - ...`. The :math:`...` contains terms that are not dependent on :math:`f`.

    Args:
        system_matrix (SystemMatrix): The system matrix :math:`H` modeling the particular system whereby the projections were obtained
        projections (torch.Tensor): Acquired data (assumed to have Poisson statistics).
        additive_term (torch.Tensor, optional): Additional term added after forward projection by the system matrix. This term might include things like scatter and randoms. Defaults to None.
    """
                
    def compute_gradient(
        self,
        object: torch.Tensor,
        subset_idx: int | None = None,
        norm_BP_subset_method: str = 'subset_specific'
        ) -> torch.Tensor:
        r"""Computes the gradient for the Poisson log likelihood given by :math:`\nabla_f L(g|f) =  H^T (g / Hf) - H^T 1`. 

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
        norm_BP = self._get_normBP(subset_idx)
        return self.system_matrix.backward(proj_subset / (self.projections_predicted + pytomography.delta), subset_idx) - norm_BP
    
    def compute_gradient_ff(
        self,
        object: torch.Tensor,
        precomputed_forward_projection: torch.Tensor | None = None,
        subset_idx: int = None,
        ) -> Callable:
        r"""Computes the second order derivative :math:`\nabla_{ff} L(g|f) = -H^T (g/(Hf+s)^2) H`. 

        Args:
            object (torch.Tensor): Object :math:`f` used in computation.
            precomputed_forward_projection (torch.Tensor | None, optional): The quantity :math:`Hf`. If this value is None, then the forward projection is recomputed. Defaults to None.
            subset_idx (int, optional): Specifies the subset for all computations. Defaults to None.

        Returns:
            Callable: The operator given by the second order derivative.
        """
        if precomputed_forward_projection is None:
            FP = self.system_matrix.forward(object, subset_idx)
        else:
            FP = precomputed_forward_projection
        proj_subset = self._get_projection_subset(self.projections, subset_idx)
        def operator(input):
            input = self.system_matrix.forward(input, subset_idx)
            input = input * proj_subset / (FP**2 + pytomography.delta)
            return -self.system_matrix.backward(input, subset_idx)
        return operator
    
    def compute_gradient_gf(
        self,
        object,
        precomputed_forward_projection = None,
        subset_idx=None,
        ):
        r"""Computes the second order derivative :math:`\nabla_{gf} L(g|f) = 1/(Hf+s) H`. 

        Args:
            object (torch.Tensor): Object :math:`f` used in computation.
            precomputed_forward_projection (torch.Tensor | None, optional): The quantity :math:`Hf`. If this value is None, then the forward projection is recomputed. Defaults to None.
            subset_idx (int, optional): Specifies the subset for all computations. Defaults to None.

        Returns:
            Callable: The operator given by the second order derivative.
        """
        if precomputed_forward_projection is None:
            FP = self.system_matrix.forward(object, subset_idx)
        else:
            FP = precomputed_forward_projection
        def operator(input):
            input = self.system_matrix.forward(input, subset_idx)
            return input / (FP + pytomography.delta)
        return operator
    
    def compute_gradient_sf(
        self,
        object,
        precomputed_forward_projection = None,
        subset_idx=None,
        ):
        r"""Computes the second order derivative :math:`\nabla_{sf} L(g|f,s) = -g/(Hf+s)^2 H` where :math:`s` is an additive term representative of scatter. 

        Args:
            object (torch.Tensor): Object :math:`f` used in computation.
            precomputed_forward_projection (torch.Tensor | None, optional): The quantity :math:`Hf`. If this value is None, then the forward projection is recomputed. Defaults to None.
            subset_idx (int, optional): Specifies the subset for all computations. Defaults to None.

        Returns:
            Callable: The operator given by the second order derivative.
        """
        proj_subset = self._get_projection_subset(self.projections, subset_idx)
        if precomputed_forward_projection is None:
            FP = self.system_matrix.forward(object, subset_idx)
        else:
            FP = precomputed_forward_projection
        def operator(input):
            input = self.system_matrix.forward(input, subset_idx)
            return -input * proj_subset / (FP + pytomography.delta)**2
        return operator
    
            
        
        