from __future__ import annotations
import torch
import torch.nn as nn
from pytomography.likelihoods import Likelihood
from .preconditioned_gradient_ascent import OSEM

def _positive_root(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""The non-negative root of :math:`x^2 - a x - b = 0` (with :math:`b \geq 0`), :math:`\frac{1}{2}(a + \sqrt{a^2 + 4b})`.
    Where :math:`a < 0` the two terms nearly cancel in float32 (the EM update outweighs the network there), so it is
    computed as :math:`2b / (\sqrt{a^2 + 4b} - a)`, the same value; 0 where :math:`a \leq 0` and :math:`b = 0`."""
    root = torch.sqrt(a * a + 4 * b)
    denominator = root - a
    small = 2 * b / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return torch.where(a > 0, 0.5 * (a + root), torch.where(denominator > 0, small, torch.zeros_like(a)))

class DIPRecon:
    r"""Implementation of the Deep Image Prior reconstruction technique (see https://ieeexplore.ieee.org/document/8581448). This reconstruction technique requires an instance of a user-defined ``prior_network`` that implements two functions: (i) a ``fit`` method that takes in an ``object`` (:math:`x`) which the network ``f(z;\theta)`` is subsequently fit to, and (ii) a ``predict`` function that returns the current network prediction :math:`f(z;\theta)`. For more details, see the Deep Image Prior tutorial.

        Args:
            likelihood (Likelihood): Initialized likelihood function for the imaging system considered
            prior_network (nn.Module): User defined prior network that implements the neural network :math:`f(z;\theta)` that predicts an object given a prior image :math:`z`. This network also implements a ``fit`` method that takes in an object and fits the network to the object (for a specified number of iterations: SubIt2 in the paper).
            rho (float, optional): Value of :math:`\rho` used in the optimization procedure. Larger values of :math:`\rho` give larger weight to the neural network, while smaller values of :math:`\rho` give larger weight to the EM updates: in voxel :math:`j`, an update moves about :math:`s_j / (\rho x_j)` of an EM step towards the data, where :math:`s` is the sensitivity image. :math:`\rho` therefore depends on the units of the data and the image; a value near the median sensitivity divided by the mean activity in the object weighs the two about equally. Defaults to 3e-3 (the value in the paper).
        """
    def __init__(
        self,
        likelihood: Likelihood,
        prior_network: nn.Module,
        rho: float = 3e-3,
    ) -> None:
        self.EM_algorithm = OSEM(
            likelihood,
            object_initial = nn.ReLU()(prior_network.predict().clone())
            )
        self.likelihood = likelihood
        self.prior_network = prior_network
        self.rho = rho
        
    def _compute_callback(self, n_iter: int, n_subset: int):
        """Method for computing callbacks after each reconstruction iteration

        Args:
            n_iter (int): Number of iterations
            n_subset (int): Number of subsets
        """
        self.object_prediction = self.callback.run(self.object_prediction, n_iter, n_subset)
        
    def __call__(
        self,
        n_iters,
        subit1,
        n_subsets_osem=1,
        callback=None,
    ):  
        r"""Implementation of Algorithm 1 in https://ieeexplore.ieee.org/document/8581448. This implementation gives the additional option to use ordered subsets. The quantity SubIt2 specified in the paper is controlled by the user-defined ``prior_network`` class.

        Args:
            n_iters (int): Number of iterations (MaxIt in paper)
            subit1 (int): Number of OSEM iterations before retraining neural network (SubIt1 in paper)
            n_subsets_osem (int, optional): Number of subsets to use in OSEM reconstruction. Defaults to 1.

        Returns:
            torch.Tensor: Reconstructed image
        """
        self.callback = callback
        # Initialize quantities
        mu = 0 
        norm_BP = self.likelihood.system_matrix.compute_normalization_factor()
        x = self.prior_network.predict()
        x_network = x.clone()
        for _ in range(n_iters):
            for j in range(subit1):
                for k in range(n_subsets_osem):
                    self.EM_algorithm.object_prediction = nn.ReLU()(x.clone())
                    x_EM = self.EM_algorithm(n_iters = 1, n_subsets = n_subsets_osem, n_subset_specific=k)
                    # the maximum of the EM surrogate plus the penalty: the positive root of x^2 - a x - b = 0
                    x = _positive_root(x_network - mu - norm_BP / self.rho, x_EM * norm_BP / self.rho)
            self.prior_network.fit(x + mu)
            x_network = self.prior_network.predict()
            mu += x - x_network
            self.object_prediction = nn.ReLU()(x_network)
            # evaluate callback
            if self.callback is not None:
                self._compute_callback(n_iter = _, n_subset=None)
        return self.object_prediction