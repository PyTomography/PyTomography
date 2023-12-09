from __future__ import annotations
from .callback import Callback
import torch
from pytomography.projectors import SystemMatrix
from pytomography.priors import Prior

class LogLikelihoodCallback(Callback):
    r"""Computes the log-liklihood :math:`\sum \left(g\log(Hf) - Hf - \beta V(f) \right)` after a given iteration.

        Args:
            projections (torch.tensor): Data corresponding to measured projections
            system_matrix (SystemMatrix): System matrix of imaging system.
            prior (Prior, optional): Prior used in Bayesian algorithm. Defaults to None.
    """
    def __init__(
        self,
        projections: torch.tensor,
        system_matrix: SystemMatrix,
        prior: Prior | None = None
    ) -> None:
        self.projections = projections
        self.system_matrix = system_matrix
        self.liklihoods = []
        self.liklihoods_prior_component = []
        self.prior = prior
    def run(self, object: torch.tensor, n_iter: int):
        """Method used to compute the log liklihood

        Args:
            object (torch.tensor): Object on which the liklihood is computed
            n_iter (int): Iteration number
        """
        projection_estimate = self.system_matrix.forward(object)
        liklihood = (self.projections*torch.log(projection_estimate)) - projection_estimate
        liklihood[self.projections<=0] = -projection_estimate[self.projections<=0]
        # Sum components of each voxel
        liklihood = liklihood.sum().item()
        # Add prior component
        if self.prior is not None:
            self.prior.set_object(object)
            prior_component = -self.prior.compute_prior()
            self.liklihoods_prior_component.append(prior_component)
            liklihood -= prior_component
        self.liklihoods.append(liklihood)