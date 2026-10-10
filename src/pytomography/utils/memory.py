"""Memory budgets. The host (RAM) budget set with :func:`pytomography.set_memory_budget` sizes the blocks and subsets of
PyTomography's memory-heavy steps (without a budget, every step keeps its fixed default); the GPU budget of chunked
computations keeps them from taking over a shared GPU."""
from __future__ import annotations
import math
import torch
import pytomography

#: Default budget (bytes) of the chunked computations that take one, such as filtered back projection.
DEFAULT_BUDGET = 1.5e9


def gpu_budget(budget: float | None = None, device: str | torch.device | None = None, fraction: float = 0.25) -> float:
    """Bytes a chunked computation may hold on ``device``: ``budget`` (default :data:`DEFAULT_BUDGET`), and on a CUDA
    device never more than ``fraction`` of the memory free when it is called, since the GPU may be shared with other
    jobs. On other devices the budget is returned as given; it then bounds host memory instead.

    Args:
        budget (float, optional): Requested budget in bytes. Defaults to :data:`DEFAULT_BUDGET`.
        device (str | torch.device, optional): Device the computation runs on. Defaults to ``pytomography.device``.
        fraction (float, optional): Largest fraction of the free CUDA memory to use. Defaults to 0.25.

    Returns:
        float: Budget in bytes.
    """
    device = torch.device(pytomography.device if device is None else device)
    budget = DEFAULT_BUDGET if budget is None else float(budget)
    if device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        budget = min(budget, fraction * free)
    return budget


class PeakMemory:
    """Context manager that measures the peak memory allocated by PyTorch on a CUDA device above the level at entry,
    in bytes (``peak``). On other devices ``peak`` stays 0.

    Example:
        >>> with PeakMemory() as m:
        ...     image = FilteredBackProjection(projections, system_matrix)()
        >>> m.peak / 1e9
    """
    def __init__(self, device: str | torch.device | None = None):
        self.device = torch.device(pytomography.device if device is None else device)
        self.peak = 0

    def __enter__(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self._base = torch.cuda.memory_allocated(self.device)
        return self

    def __exit__(self, *exc):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            self.peak = torch.cuda.max_memory_allocated(self.device) - self._base
        return False

#: Largest block of temporaries (bytes) a blocked step uses, whatever the budget. Larger blocks are no faster for these
#: steps, and on Windows a process keeps part of what its larger blocks took after freeing them (3.5 GB after smoothing
#: the mMR randoms in 3 GB blocks, 1.5 GB with 0.25 GB blocks), which then counts against the budget.
BLOCK_CAP_BYTES = 5e8


def block_size(bytes_per_item: float, default: int, fraction: float = 1 / 8) -> int:
    """Number of items (crystal pairs, events, rows) a step processes at once: as many as fit in ``fraction`` of the memory budget, but at most :data:`BLOCK_CAP_BYTES` of temporaries; ``default`` without a budget.

    Args:
        bytes_per_item (float): Memory the step needs per item, temporaries included.
        default (int): Number of items without a budget.
        fraction (float, optional): Share of the budget the step may take. Defaults to 1/8.

    Returns:
        int: Number of items per block (at least 1).
    """
    if pytomography.memory_budget is None:
        return default
    return max(1, int(min(pytomography.memory_budget * fraction, BLOCK_CAP_BYTES) // bytes_per_item))


#: Without a memory budget, an array larger than this (bytes) is still computed one part at a time rather than held
#: whole: a TOF sinogram of a clinical scanner (34.6 GB for the Siemens Biograph mMR with 21 TOF bins) never is.
LAZY_WITHOUT_BUDGET_BYTES = 8e9


def prefer_lazy(nbytes: float) -> bool:
    """Whether an array of ``nbytes`` bytes should be computed one part at a time instead of held whole: when it would take more than a quarter of the memory budget, or, without a budget, more than :data:`LAZY_WITHOUT_BUDGET_BYTES` (8 GB).

    Args:
        nbytes (float): Size of the whole array.

    Returns:
        bool: True if the array should be computed a part at a time.
    """
    if pytomography.memory_budget is None:
        return nbytes > LAZY_WITHOUT_BUDGET_BYTES
    return nbytes > pytomography.memory_budget / 4


def subsets_for_budget(projection_bytes: float, arrays: int = 3, held_bytes: float | None = None, minimum: int = 1) -> int:
    """Fewest subsets for which a reconstruction fits in the memory budget.

    A subset needs ``arrays`` arrays the size of one subset of the projections at once (a Poisson likelihood holds the
    measured subset, its expected value and their ratio), next to ``held_bytes`` held for the whole reconstruction (the
    data, the additive term, the images; half the budget if not given).

    Args:
        projection_bytes (float): Size of all the projections (e.g. a whole time of flight sinogram, even if it is never held whole).
        arrays (int, optional): Number of subset-sized arrays held at once. Defaults to 3.
        held_bytes (float | None, optional): Memory held besides them. Defaults to None (half the budget).
        minimum (int, optional): Fewest subsets to return. Defaults to 1.

    Returns:
        int: Number of subsets (``minimum`` without a budget).
    """
    budget = pytomography.memory_budget
    if budget is None:
        return minimum
    available = budget - (budget / 2 if held_bytes is None else held_bytes)
    if available <= 0:
        raise ValueError(f"the memory budget ({budget / 1e9:.1f} GB) leaves nothing for the subsets after the {held_bytes / 1e9:.1f} GB held: raise it with pytomography.set_memory_budget")
    return max(minimum, math.ceil(arrays * projection_bytes / available))
