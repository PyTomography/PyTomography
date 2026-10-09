"""Memory budgets for computations that are split into chunks, so that they do not take over a shared GPU."""
from __future__ import annotations
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
