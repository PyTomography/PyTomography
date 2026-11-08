"""Deep image prior reconstruction (:class:`DIPRecon`): EM updates alternated with fitting a neural network."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import pytomography
from pytomography.likelihoods import Likelihood
from pytomography.projectors.system_matrix import held_parts, on_device
from pytomography.utils.memory import MemoryEstimate, MemoryPart
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
    r"""Deep image prior reconstruction, as in https://ieeexplore.ieee.org/document/8581448.

    The reconstruction alternates EM updates of the image :math:`x` with fitting a neural network
    :math:`f(z;\theta)` to it, joined by a penalty :math:`\rho` and a dual variable (Algorithm 1 of the paper). The
    ``prior_network`` is defined by the user and must have two methods: ``fit(object)``, which fits the network to an
    object (for SubIt2 iterations, in the paper's terms), and ``predict()``, which returns the network's current
    prediction :math:`f(z;\theta)`, [Lx, Ly, Lz]. See
    the Deep Image Prior tutorial.

    Args:
        likelihood (Likelihood): The likelihood, with the data and the system matrix.
        prior_network (nn.Module): The network, with ``fit`` and ``predict`` methods.
        rho (float, optional): The ADMM penalty :math:`\rho`. Larger values give more weight to the network, smaller
            ones to the EM updates: in voxel :math:`j`, an update moves about :math:`s_j / (\rho x_j)` of an EM step
            towards the data, where :math:`s` is the sensitivity image. :math:`\rho` therefore depends on the units of
            the data and the image; a value near the median sensitivity divided by the mean activity in the object
            weighs the two about equally. Defaults to 3e-3 (the value in the paper).
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
        
    def estimate_memory(self, n_subsets_osem: int = 1, N_splits: int | None = None,
                        network_gb: float | None = None) -> MemoryEstimate:
        r"""The predicted peak RAM and GPU memory of this reconstruction, without running it.

        It is the estimate of its EM updates (:meth:`OSEM.estimate_memory
        <pytomography.algorithms.OSEM.estimate_memory>`: the system matrix, with the likelihood's projections and
        additive term) and DIP's own images: four for the whole run (the image :math:`x`, the network's prediction, the
        dual variable :math:`\mu` and the sensitivity image) and eight in each update (the EM image and the
        temporaries of the update). What the network takes while it is fitted (its parameters, their gradients, the
        optimiser's state and the activations) depends on the network and its input, so it is counted only when
        ``network_gb`` gives it: measure one ``prior_network.fit`` with :class:`~pytomography.utils.memory.PeakMemory`.
        The fit and the EM updates take turns, so adding them overestimates a little. ``print`` it, or ``to_dict()``
        for JSON.

        Args:
            n_subsets_osem (int, optional): Number of subsets of the EM updates (the call's ``n_subsets_osem``).
                Defaults to 1.
            N_splits (int | None, optional): Projector splits, for system matrices that take them. Defaults to None:
                the system matrix's own.
            network_gb (float | None, optional): Memory the network takes while it is fitted, in GB, on PyTomography's
                device. Defaults to None: not counted, and the estimate's title says so.

        Returns:
            MemoryEstimate: The estimate.

        Example:
            >>> print(DIPRecon(likelihood, prior_network, rho=240).estimate_memory(network_gb=7))
        """
        system_matrix = self.likelihood.system_matrix
        settings = {name: present for name, (present, _) in system_matrix._memory_settings().items()}
        settings['n_subsets'] = n_subsets_osem
        if N_splits is not None and 'N_splits' in settings:
            settings['N_splits'] = N_splits
        image = 4 * math.prod(system_matrix.object_meta.shape)
        device = pytomography.device
        dip = [MemoryPart("DIP: the image, the network's prediction, the dual variable and the sensitivity image",
                          **on_device(device, 4 * image)),
               MemoryPart('DIP: the EM image and the temporaries of an update', **on_device(device, 8 * image),
                          scope='subset')]
        if network_gb is not None:
            dip.append(MemoryPart("the network's fit", **on_device(device, network_gb * 1e9), scope='subset'))
        title = f"DIPRecon ({type(system_matrix).__name__})" + ("" if network_gb is not None else
                                                                 ", without the network's fit")
        return system_matrix._memory_estimate(settings, held_parts(self.EM_algorithm._memory_held()),
                                              extra=lambda n: self.EM_algorithm._memory_extra(n) + dip, title=title,
                                              title_settings={'N_splits': N_splits})

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
        r"""Reconstructs (Algorithm 1 of the paper), with the option of ordered subsets in the EM updates.

        SubIt2 of the paper (the network's training iterations) is up to the ``prior_network``'s ``fit``.

        Args:
            n_iters (int): Number of outer iterations (MaxIt in the paper).
            subit1 (int): Number of EM iterations before each fit of the network (SubIt1 in the paper).
            n_subsets_osem (int, optional): Number of subsets of the EM updates. Defaults to 1.
            callback (Callback, optional): Run on the network's prediction after each outer iteration. Defaults to
                None.

        Returns:
            torch.Tensor: The reconstructed image, the network's last prediction (negative values set to 0),
            [Lx, Ly, Lz].
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