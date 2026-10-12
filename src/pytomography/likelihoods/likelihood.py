from __future__ import annotations
import pytomography
from pytomography.projectors import SystemMatrix
from collections.abc import Callable
import math
import torch
from pytomography.utils.memory import subsets_for_budget

def _rows_per_block(x: torch.Tensor, block_bytes: float = 2.5e8) -> int:
    """Rows of ``x`` (along its first dimension) in a block of about ``block_bytes``."""
    return max(1, int(block_bytes // max(1, x[0].numel() * x.element_size()))) if x.ndim > 0 and x.shape[0] > 0 else 1

class Likelihood:
    """Generic likelihood class in PyTomography. Subclasses may implement specific likelihoods with methods to compute the likelihood itself as well as particular gradients of the likelihood 

    Args:
        system_matrix (SystemMatrix): The system matrix modeling the particular system whereby the projections were obtained
        projections (torch.Tensor | None): Acquired data. If listmode, then this argument need not be provided, and it is set to a tensor of ones. A sinogram that is computed one subset at a time (:class:`~pytomography.io.PET.shared.LazySinogram`) is also accepted, for reconstructions with subsets. Defaults to None.
        additive_term (torch.Tensor, optional): Additional term added after forward projection by the system matrix. This term might include things like scatter and randoms. Like ``projections``, it may be computed one subset at a time. Defaults to None (no additive term).
        additive_term_variance_estimate (Callable, optional): Operator for variance estimate of additive term. If none, then uncertainty estimation does not include contribution from the additive term. Defaults to None.
    """
    def __init__(
        self,
        system_matrix: SystemMatrix,
        projections: torch.Tensor | None = None,
        additive_term: torch.Tensor = None,
        additive_term_variance_estimate: Callable | None = None
        ) -> None:
        self.system_matrix = system_matrix
        if projections is None: # listmode reconstruction
            self.projections = torch.tensor([1.]).to(pytomography.device)
        else:
            self.projections = projections
        self.FP = None # stores current state of forward projection
        if isinstance(additive_term, torch.Tensor):
            self.additive_term = additive_term.to(self.projections.device).to(pytomography.dtype)
            self.exists_additive_term = True
        elif additive_term is not None: # computed one subset at a time, e.g. a LazySinogram
            self.additive_term = additive_term
            self.exists_additive_term = True
        else:
            # Nothing is added. This used to be a tensor of zeros the size of the projections, which for the TOF sinogram
            # of a clinical scanner is tens of GB.
            self.additive_term = None
            self.exists_additive_term = False
        self.n_subsets_previous = -1
        self.additive_term_variance_estimate = additive_term_variance_estimate
    
    def _set_n_subsets(
        self,
        n_subsets: int
        )-> None:
        """Sets the number of subsets to be used when computing the likelihood

        Args:
            n_subsets (int): Number of subsets
        """
        self.n_subsets = n_subsets
        self._check_memory_budget(n_subsets)
        if n_subsets < 2:
            self.norm_BP = self.system_matrix.compute_normalization_factor()
        else:
            self.system_matrix.set_n_subsets(n_subsets)
            if self.n_subsets_previous!=self.n_subsets:
                self.norm_BPs = []
                for k in range(self.n_subsets):
                    self.norm_BPs.append(self.system_matrix.compute_normalization_factor(k))
        self.n_subsets_previous = n_subsets
        
    def _check_memory_budget(self, n_subsets: int) -> None:
        """Stops a reconstruction whose subsets would not fit in the memory budget (:func:`pytomography.set_memory_budget`) before it starts, with the number of subsets that would fit. A subset needs about three arrays of its size at once; half the budget is left for everything else.

        Args:
            n_subsets (int): Number of subsets
        """
        if pytomography.memory_budget is None:
            return
        projection_bytes = 4 * math.prod(self.projections.shape)   # all the projections, even if they are computed a subset at a time
        needed = subsets_for_budget(projection_bytes)
        if n_subsets < needed:
            raise ValueError(f"one subset of {n_subsets} needs about {3 * projection_bytes / n_subsets / 1e9:.1f} GB at once, more than the "
                             f"{pytomography.memory_budget / 2e9:.1f} GB the memory budget of {pytomography.memory_budget / 1e9:.1f} GB leaves for it: "
                             f"use at least {needed} subsets, or raise the budget with pytomography.set_memory_budget")

    def _projection_device(self) -> torch.device | None:
        """The device the system matrix puts its projections on (its ``output_device``), if it has one."""
        device = getattr(self.system_matrix, 'output_device', None)
        return None if device is None else torch.device(device)

    def _get_projection_subset(self, projections: torch.Tensor, subset_idx: int | None = None) -> torch.Tensor:
        """Method for getting projection subset corresponding to given subset index. The subset is on the device the
        system matrix puts its projections on: a subset of projections computed one subset at a time (a LazySinogram)
        is computed there (on a GPU, it never passes through host memory), and a tensor's subset is copied there.

        Args:
            projections (torch.Tensor): Projection data
            subset_idx (int): Subset index

        Returns:
            torch.Tensor: Subset projection data
        """
        device = self._projection_device()
        if subset_idx is None:
            # projections computed one subset at a time (a LazySinogram) are computed whole
            whole = projections if isinstance(projections, torch.Tensor) else projections.to_dense()
            return whole if device is None else whole.to(device)
        angles = self._sinogram_subset_angles(subset_idx)
        if not isinstance(projections, torch.Tensor) and hasattr(projections, 'compute_at') and angles is not None:
            return projections.compute_at(angles, device if device is not None else 'cpu')
        subset = self.system_matrix.get_projection_subset(projections, subset_idx)
        return subset if device is None or not isinstance(subset, torch.Tensor) else subset.to(device)

    def _sinogram_subset_angles(self, subset_idx: int | None) -> torch.Tensor | None:
        """The angles (on the CPU) of subset ``subset_idx``, if the system matrix splits its projections by angle (as the PET sinogram system matrix does); otherwise None."""
        if subset_idx is None or not hasattr(self.system_matrix, 'proj_meta') or not hasattr(self.system_matrix.proj_meta, 'N_angles'):
            return None
        return self.system_matrix.subset_indices_array[subset_idx].cpu()

    def _forward_with_additive_term(self, object: torch.Tensor, subset_idx: int | None = None) -> torch.Tensor:
        r"""Computes the expected projections :math:`H_m f + s_m` of a subset, and keeps them in ``projections_predicted``.

        The additive term is added into the forward projection, which is a new tensor, instead of making a third tensor of the same size: with a time of flight sinogram, each of these is the size of a subset of the data. The previous subset's prediction is released first, for the same reason.

        Args:
            object (torch.Tensor): Object :math:`f`
            subset_idx (int | None, optional): Subset index :math:`m`. Defaults to None.

        Returns:
            torch.Tensor: Expected projections of the subset.
        """
        self.projections_predicted = None
        FP = self.system_matrix.forward(object, subset_idx)
        if self.additive_term is not None:
            angles = self._subset_angles(subset_idx, FP)
            if angles is not None and not isinstance(self.additive_term, torch.Tensor):
                # computed a subset at a time (a LazySinogram): added a block of the subset's angles at a time, so the
                # subset's additive term is never held whole
                rows = _rows_per_block(FP)
                for start in range(0, len(angles), rows):
                    FP[start:start + rows] += self._get_projection_rows(self.additive_term, angles[start:start + rows], FP.device)
            else:
                additive_term_subset = self._get_projection_subset(self.additive_term, subset_idx)
                if torch.broadcast_shapes(FP.shape, additive_term_subset.shape) == FP.shape and FP.dtype == additive_term_subset.dtype:
                    FP += additive_term_subset
                else:
                    FP = FP + additive_term_subset
                del additive_term_subset
        self.projections_predicted = FP
        return FP

    def _subset_angles(self, subset_idx: int | None, FP: torch.Tensor) -> torch.Tensor | None:
        """The angles of a sinogram subset, if the system matrix splits its projections by their first dimension (as the PET sinogram system matrix does) and ``FP`` has one row per angle; otherwise None."""
        if subset_idx is None or not hasattr(self.system_matrix, 'proj_meta') or not hasattr(self.system_matrix.proj_meta, 'N_angles'):
            return None
        angles = self.system_matrix.subset_indices_array[subset_idx]
        return angles if FP.shape[0] == len(angles) else None

    def _is_own_copy(self, proj_subset: torch.Tensor) -> bool:
        """Whether a subset of the projections is a tensor of its own (a copy made by indexing, or computed a subset at a time), which may be overwritten, rather than the projections themselves or a view of them."""
        if not isinstance(proj_subset, torch.Tensor) or proj_subset._base is not None:
            return False
        if not isinstance(self.projections, torch.Tensor):
            return True
        return proj_subset.untyped_storage().data_ptr() != self.projections.untyped_storage().data_ptr()

    @staticmethod
    def _get_projection_rows(projections, angles: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """Rows ``angles`` of projections indexed by angle (a tensor or a LazySinogram), on ``device`` (a LazySinogram computes them there)."""
        if not isinstance(projections, torch.Tensor) and hasattr(projections, 'compute_at') and device is not None:
            return projections.compute_at(angles, device)
        rows = projections[angles.to(projections.device) if isinstance(projections, torch.Tensor) else angles]
        return rows if device is None else rows.to(device)
        
        
    def _get_normBP(self, subset_idx: int, return_sum: bool = False):
        """Gets normalization factor (back projection of ones)

        Args:
            subset_idx (int): Subset index
            return_sum (bool, optional): Sum normalization factor from all subsets. Defaults to False.

        Returns:
            torch.Tensor: Normalization factor
        """
        if subset_idx is None:
            return self.norm_BP
        else:
            if return_sum:
                return torch.stack(self.norm_BPs).sum(axis=0)
            else:
                # Put on PyTomography device in case stored on CPU
                return self.norm_BPs[subset_idx].to(pytomography.device)
        
    def compute_gradient(self, *args, **kwargs):
        r"""Function used to compute the gradient of the likelihood :math:`\nabla_{f} L(g|f)`

        Raises:
            NotImplementedError: Must be implemented by sub classes
        """
        raise NotImplementedError("Compute gradient not implemented for this likelihood function")
    
    def compute_gradient_ff(self, *args, **kwargs):
        r"""Function used to compute the second order gradient (with respect to the object twice) of the likelihood :math:`\nabla_{ff} L(g|f)`

        Raises:
            NotImplementedError: Must be implemented by sub classes
        """
        raise NotImplementedError("gradient_ff not implemented for this likelihood function")
    
    def compute_gradient_gf(self, *args, **kwargs):
        r"""Function used to compute the second order gradient (with respect to the object then image) of the likelihood :math:`\nabla_{gf} L(g|f)`

        Raises:
            NotImplementedError: Must be implemented by sub classes
        """
        raise NotImplementedError("gradient_gf not implemented for this likelihood function")
    
    def compute_gradient_sf(self, *args, **kwargs):
        r"""Function used to compute the second order gradient (with respect to the object then additive term) of the likelihood :math:`\nabla_{sf} L(g|f,s)`

        Raises:
            NotImplementedError: Must be implemented by sub classes
        """
        raise NotImplementedError("gradient_sf not implemented for this likelihood function")