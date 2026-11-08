"""The Poisson log-likelihood of emission data (SPECT, PET)."""
from __future__ import annotations
from typing import Callable
import pytomography
import torch
from .likelihood import Likelihood, _rows_per_block
from pytomography.transforms.SPECT import AdditiveTermTransform

class PoissonLogLikelihood(Likelihood):
    r"""The Poisson log-likelihood :math:`L(g|f) = \sum_i g_i \ln(Hf+s)_i - (Hf+s)_i` (up to terms without :math:`f`).

    :math:`g` is the measured data, :math:`H` the system matrix, :math:`s` the additive term (scatter, randoms) and
    :math:`f` the object. The likelihood of OSEM, OSMAPOSL, BSREM and the other emission algorithms.

    Args:
        system_matrix (SystemMatrix): The system matrix :math:`H`.
        projections (torch.Tensor | None, optional): The measured data :math:`g`, with Poisson statistics (left out for
            list mode). A :class:`~pytomography.io.PET.LazySinogram` is accepted too. Defaults to None.
        additive_term (torch.Tensor, optional): The additive term :math:`s`, added after forward projection. Defaults
            to None.
        additive_term_variance_estimate (Callable, optional): A function that applies the variance of the additive
            term, for uncertainty estimates. Defaults to None.

    Example:
        .. code-block:: python

            likelihood = PoissonLogLikelihood(system_matrix, photopeak, additive_term=scatter)
            recon = OSEM(likelihood)(n_iters=4, n_subsets=8)
    """
                
    def compute_gradient(
        self,
        object: torch.Tensor,
        subset_idx: int | None = None,
        norm_BP_subset_method: str = 'subset_specific'
        ) -> torch.Tensor:
        r"""The gradient :math:`\nabla_f L(g|f) = H_m^T \left(g_m / (H_m f + s_m)\right) - H_m^T 1` over subset :math:`m`.

        Args:
            object (torch.Tensor): The object :math:`f`, [Lx, Ly, Lz].
            subset_idx (int | None, optional): Subset index :math:`m`. Defaults to None: all the projections.
            norm_BP_subset_method (str, optional): Not used: :math:`H_m^T 1` is always the subset's own. Accepted for
                the interface of the likelihoods. Defaults to ``'subset_specific'``.

        Returns:
            torch.Tensor: The gradient, [Lx, Ly, Lz].
        """
        proj_subset = self._get_projection_subset(self.projections, subset_idx)
        FP = self._forward_with_additive_term(object, subset_idx)
        norm_BP = self._get_normBP(subset_idx)
        if self._is_own_copy(proj_subset) and proj_subset.shape == FP.shape and proj_subset.dtype == FP.dtype and proj_subset.device == FP.device:
            # g / (Hf + s + delta) in the memory of this subset's copy of the data, a block of rows at a time, so that a
            # subset holds two arrays of its size (the data and the expected projections) rather than three
            rows = _rows_per_block(FP)
            for start in range(0, FP.shape[0], rows):
                proj_subset[start:start + rows].div_(FP[start:start + rows] + pytomography.delta)
            ratio = proj_subset
        else:
            # computed in the memory of the denominator (a new tensor) when the shapes allow
            ratio = FP + pytomography.delta
            if ratio.shape == proj_subset.shape and ratio.dtype == proj_subset.dtype and ratio.device == proj_subset.device:
                ratio = torch.div(proj_subset, ratio, out=ratio)
            else:
                ratio = proj_subset / ratio
        del proj_subset
        return self.system_matrix.backward(ratio, subset_idx) - norm_BP
    
    def compute_gradient_ff(
        self,
        object: torch.Tensor,
        precomputed_forward_projection: torch.Tensor | None = None,
        subset_idx: int = None,
        ) -> Callable:
        r"""The second derivative :math:`\nabla_{ff} L(g|f) = -H^T \left(g/(Hf+s)^2\right) H`, as an operator.

        Args:
            object (torch.Tensor): The object :math:`f`, [Lx, Ly, Lz].
            precomputed_forward_projection (torch.Tensor | None, optional): The expected projections :math:`Hf + s`.
                Defaults to None: :math:`Hf` is computed (without the additive term).
            subset_idx (int, optional): Subset index :math:`m`. Defaults to None: all the projections.

        Returns:
            Callable: A function that applies the operator to an object.
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
        r"""The second derivative :math:`\nabla_{gf} L(g|f) = H / (Hf+s)`, as an operator.

        Args:
            object (torch.Tensor): The object :math:`f`, [Lx, Ly, Lz].
            precomputed_forward_projection (torch.Tensor | None, optional): The expected projections :math:`Hf + s`.
                Defaults to None: :math:`Hf` is computed (without the additive term).
            subset_idx (int, optional): Subset index :math:`m`. Defaults to None: all the projections.

        Returns:
            Callable: A function that applies the operator to an object (giving projections).
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
        r"""The second derivative :math:`\nabla_{sf} L(g|f,s) = -g H / (Hf+s)^2`, as an operator (:math:`s` the scatter).

        Args:
            object (torch.Tensor): The object :math:`f`, [Lx, Ly, Lz].
            precomputed_forward_projection (torch.Tensor | None, optional): The expected projections :math:`Hf + s`.
                Defaults to None: :math:`Hf` is computed (without the additive term).
            subset_idx (int, optional): Subset index :math:`m`. Defaults to None: all the projections.

        Returns:
            Callable: A function that applies the operator to an object (giving projections).
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
    
class MonteCarloHybridSPECTPoissonLogLikelihood(PoissonLogLikelihood):
    """The Poisson log-likelihood for :class:`~pytomography.projectors.SPECT.MonteCarloHybridSPECTSystemMatrix`.

    Its gradient guards against the noise of the Monte Carlo forward projection: where the simulation predicts no
    counts but some were measured, the ratio of measured to expected counts is left out, and the ratio is clipped at
    1000. If the system matrix has an :class:`~pytomography.transforms.SPECT.AdditiveTermTransform`, its additive term
    is scaled by the mean ratio at every gradient (and the new value printed). The arguments are those of
    :class:`PoissonLogLikelihood`.
    """
    def compute_gradient(
        self,
        object: torch.Tensor,
        subset_idx: int | None = None,
        norm_BP_subset_method: str = 'subset_specific'
        ) -> torch.Tensor:
        """The gradient of the log-likelihood with respect to the object, with the guards above.

        Args:
            object (torch.Tensor): The object :math:`f`, [Lx, Ly, Lz] (the source of the simulation).
            subset_idx (int | None, optional): Subset index :math:`m`. Defaults to None: all the projections.
            norm_BP_subset_method (str, optional): Not used. Defaults to ``'subset_specific'``.

        Returns:
            torch.Tensor: The gradient, [Lx, Ly, Lz].
        """
        proj_subset = self._get_projection_subset(self.projections, subset_idx)
        self._forward_with_additive_term(object, subset_idx)
        mask_bad = (self.projections_predicted < pytomography.delta)*(proj_subset > pytomography.delta)
        ratio = (~mask_bad * proj_subset + pytomography.delta) / (self.projections_predicted + pytomography.delta)
        ratio[ratio>1000] = 1000 # clip to prevent MC noise causing instability
        norm_BP = self._get_normBP(subset_idx)
        # Look for AdditiveTermTransform in System Matrix proj2proj transforms and update self.additive_term within that transform if estimate_background is true
        for transform in self.system_matrix.proj2proj_transforms:
            if isinstance(transform, AdditiveTermTransform):
                transform.additive_term = transform.additive_term * ratio.sum() / (ratio*0+1).sum() 
                print(f'Updated additive term to {transform.additive_term}')
        return self.system_matrix.backward(ratio, subset_idx) - norm_BP
    
            
        
        