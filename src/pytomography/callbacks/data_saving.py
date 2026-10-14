"""A callback that keeps the object and the expected projections of every subiteration, for uncertainty estimates."""
from __future__ import annotations
import torch
from .callback import Callback
from pytomography.likelihoods import Likelihood
import torch

class DataStorageCallback(Callback):
    """Keeps the object and the expected projections of every subiteration (on the CPU), for uncertainty estimates.

    ``objects[n]`` is the object before subiteration ``n`` (the initial object first, the reconstruction last), and
    ``projections_predicted[n]`` its expected projections :math:`Hf + s` (None for the last). Pass it to the
    algorithm's ``compute_uncertainty`` after the reconstruction. It holds two arrays per subiteration: one image
    and one set of projections.

    Args:
        likelihood (Likelihood): The likelihood of the reconstruction.
        object_initial (torch.Tensor): The initial object of the reconstruction, [Lx, Ly, Lz]
            (``algorithm.object_prediction`` before it runs).

    Attributes:
        objects (list[torch.Tensor]): The object before each subiteration, and the reconstruction last.
        projections_predicted (list[torch.Tensor | None]): The expected projections of each object.

    Example:
        .. code-block:: python

            algorithm = OSEM(likelihood)
            storage = DataStorageCallback(likelihood, algorithm.object_prediction)
            recon = algorithm(n_iters=4, n_subsets=8, callback=storage)
            uncertainty = algorithm.compute_uncertainty(mask, storage)
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
        """Keeps the object before this subiteration and its expected projections.

        Args:
            object (torch.Tensor): The object after the subiteration, [Lx, Ly, Lz].
            n_iter (int): The iteration number.
            n_subset (int): The subset index.

        Returns:
            torch.Tensor: ``object``, unchanged.
        """
        # Append from previous iteration
        self.objects.append(self.object_previous.cpu())
        # FP contains scatter
        self.projections_predicted.append(self.likelihood.projections_predicted.cpu())
        self.object_previous = torch.clone(object)
        return object
        
    def finalize(self, object: torch.Tensor):
        """Keeps the reconstructed object as the last of ``objects``.

        Args:
            object (torch.Tensor): The reconstructed object, [Lx, Ly, Lz].
        """
        self.objects.append(object.cpu())
        self.projections_predicted.append(None)