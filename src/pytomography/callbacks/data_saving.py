from __future__ import annotations
import torch
from .callback import Callback
from pytomography.likelihoods import Likelihood
import torch

class DataStorageCallback(Callback):
    """Callback that stores the object and forward projection at each iteration

    Args:
        likelihood (Likelihood): Likelihood function used in reconstruction
        object_initial (torch.Tensor[Lx, Ly, Lz]): Initial object in the reconstruction algorithm
    """
    def __init__(
        self,
        likelihood: Likelihood,
        object_initial: torch.Tensor
        ) -> None:
        
        self.object_previous = torch.clone(object_initial)
        self.objects = []
        self.projections_predicted = []
        self.likelihood = likelihood

    def run(self, object: torch.Tensor, n_iter: int, n_subset: int) -> torch.Tensor:
        """Applies the callback

        Args:
            object (torch.Tensor[Lx, Ly, Lz]): Object at current iteration
            n_iter (int): Current iteration number
            n_subset (int): Current subset index

        Returns:
            torch.Tensor: Original object passed (object is not modifed)
        """
        # Append from previous iteration
        self.objects.append(self.object_previous.cpu())
        # FP contains scatter
        self.projections_predicted.append(self.likelihood.projections_predicted.cpu())
        self.object_previous = torch.clone(object)
        return object
        
    def finalize(self, object: torch.Tensor):
        """Finalizes the callback after all iterations are called

        Args:
            object (torch.Tensor[Lx, Ly, Lz]): Reconstructed object (all iterations/subsets completed)
        """
        self.objects.append(object.cpu())
        self.projections_predicted.append(None)