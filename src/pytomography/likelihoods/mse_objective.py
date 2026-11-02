"""Least squares objectives, for CT and other data without Poisson statistics."""
from __future__ import annotations
from collections.abc import Sequence

import numpy as np
import pytomography
import torch
from .likelihood import Likelihood
from pytomography.projectors import SystemMatrix
from pytomography.utils.misc import get_1d_gaussian_kernel


def poisson_weights(
    projections: torch.Tensor,
    smoothing: float | Sequence[float] = 2.0,
    min_counts: float = 0.5,
    ) -> torch.Tensor:
    r"""Weights for weighted least squares on measured counts: one over each bin's variance, estimated from the data.

    A bin of counts has a variance equal to its mean, which the measured counts estimate. They are smoothed first
    (a Gaussian, cut off at 3 standard deviations, with the edge bins repeated beyond the edges), since a bin's own
    noise would otherwise set its weight: bins that happen to count low would weigh more, and the image would be
    biased low. The smoothed counts are kept above ``min_counts``, so that bins with no
    counts (outside the patient, at the edge of the field of view) get a finite weight, at most ``1 / min_counts``.
    The weights are :math:`w_i = 1 / \max(G_\sigma * g, c_{\min})_i`.

    Args:
        projections (torch.Tensor): The measured counts :math:`g`, e.g. SPECT projections [views, r, z].
        smoothing (float | Sequence[float], optional): The Gaussian's standard deviation :math:`\sigma`, in bins. A
            number applies along every axis but the first (for SPECT projections, within each view, not between
            views); a sequence gives one per axis. 0 uses the counts as they are. Defaults to 2.
        min_counts (float, optional): The least variance a bin is given, :math:`c_{\min} > 0`, in counts. Defaults
            to 0.5.

    Returns:
        torch.Tensor: The weights, with the projections' shape, device and dtype.
    """
    if min_counts <= 0:
        raise ValueError("min_counts must be positive")
    sigma = [0.0] + [float(smoothing)] * (projections.dim() - 1) if np.isscalar(smoothing) else [float(x) for x in smoothing]
    if len(sigma) != projections.dim() or min(sigma) < 0:
        raise ValueError(f"smoothing needs one non-negative value per axis of the projections ({projections.dim()})")
    counts = projections.detach().to(pytomography.dtype)
    with torch.no_grad():
        for axis, sig in enumerate(sigma):
            if sig > 0:
                blur = get_1d_gaussian_kernel(sig, int(2 * np.ceil(3 * sig) + 1), padding_mode='replicate').to(counts)
                lines = counts.movedim(axis, -1)
                counts = blur(lines.reshape(-1, 1, lines.shape[-1])).reshape(lines.shape).movedim(-1, axis)
    return (1 / counts.clamp(min=min_counts)).to(projections.dtype)


class NegativeMSELikelihood(Likelihood):
    r"""The negative squared error :math:`L(g|f) = -\frac{1}{2} \alpha \sum_i w_i \left(g_i-(Hf+s)_i\right)^2`.

    :math:`g` is the measured data, :math:`H` the system matrix, :math:`s` the additive term, :math:`f` the object,
    :math:`\alpha` the scaling constant and :math:`w` the weights (one for every bin unless given). It is negative so
    that gradient ascent algorithms maximise it. Weights of one over each bin's variance make it weighted least squares:
    for counts, :func:`poisson_weights` estimates them from the data.

    Args:
        system_matrix (SystemMatrix): The system matrix :math:`H`.
        projections (torch.Tensor | None, optional): The measured data :math:`g`. Defaults to None.
        additive_term (torch.Tensor, optional): The additive term :math:`s`, added after forward projection. Defaults
            to None.
        scaling_constant (float, optional): The constant :math:`\alpha`. Defaults to 1.
        weights (torch.Tensor | None, optional): The weight :math:`w_i \geq 0` of every bin, with the projections'
            shape. Defaults to None: every bin weighs one.
    """
    def __init__(
        self,
        system_matrix: SystemMatrix,
        projections: torch.Tensor | None = None,
        additive_term: torch.Tensor = None,
        scaling_constant: float = 1.0,
        weights: torch.Tensor | None = None
        ) -> None:
        super().__init__(system_matrix, projections, additive_term)
        self.scaling_constant = scaling_constant
        self.weights = None
        if weights is not None:
            if tuple(weights.shape) != tuple(self.projections.shape):
                raise ValueError(f"weights have shape {tuple(weights.shape)}, the projections {tuple(self.projections.shape)}")
            if not torch.all(torch.isfinite(weights) & (weights >= 0)):
                raise ValueError("weights must be finite and non-negative")
            self.weights = weights.to(self.projections.device).to(pytomography.dtype)
    def compute_gradient(
        self,
        object: torch.Tensor,
        subset_idx: int | None = None,
        norm_BP_subset_method: str = 'subset_specific'
        ) -> torch.Tensor:
        r"""The gradient :math:`\nabla_f L(g|f) = \alpha H_m^T w_m \left(g_m - (H_m f + s_m)\right)` over subset :math:`m`.

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
        residual = proj_subset - self.projections_predicted
        if self.weights is not None:
            residual = residual * self._get_projection_subset(self.weights, subset_idx)
        return self.system_matrix.backward(residual, subset_idx) * self.scaling_constant
    
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