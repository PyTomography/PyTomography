"""Memory budgets and estimates. The host (RAM) budget set with :func:`pytomography.set_memory_budget` sizes the blocks
and subsets of PyTomography's memory-heavy steps (without a budget, every step keeps its fixed default); the GPU budget
of chunked computations keeps them from taking over a shared GPU. :class:`MemoryEstimate` is what the ``estimate_memory``
methods (e.g. :meth:`pytomography.projectors.SystemMatrix.estimate_memory`) return: the predicted peak RAM and GPU
memory of a computation, what it is made of, and what other settings would need."""
from __future__ import annotations
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence
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


@contextmanager
def memory_budget_set(gb: float | None):
    """Context manager that sets the memory budget (:func:`pytomography.set_memory_budget`) to ``gb`` inside its block and
    restores the previous budget afterwards, also if the block raises. Estimates use it to see what a lower budget
    would change.

    Example:
        >>> with memory_budget_set(16):
        ...     estimate = system_matrix.estimate_memory(n_subsets=14)
    """
    previous = pytomography.memory_budget
    pytomography.set_memory_budget(gb)
    try:
        yield
    finally:
        pytomography.memory_budget = previous


#: Python, PyTorch and the CUDA context: the committed memory of a process after ``import torch`` and its first CUDA
#: call, measured on Windows with an RTX 5090 (1.9 GB).
FIXED_OVERHEAD_BYTES = 1.9e9

#: What each scope of a :class:`MemoryPart` means, in the order estimates print them.
MEMORY_SCOPES = {
    'held': 'held for the whole run',
    'subset': 'one subset at a time',
    'chunk': 'projector chunks',
    'overhead': 'Python, CUDA, allocator',
}


@dataclass
class MemoryPart:
    """One array, or one group of arrays, in a memory estimate.

    Args:
        name (str): What it is, as printed (e.g. ``'expected counts'``).
        ram_bytes (float, optional): Host memory it takes, in bytes. Defaults to 0.
        gpu_bytes (float, optional): GPU memory it takes, in bytes. Defaults to 0.
        scope (str, optional): ``'held'`` for the whole computation, ``'subset'`` for one subset at a time,
            ``'chunk'`` for projector or kernel temporaries, or ``'overhead'`` (Python, CUDA, allocator). Defaults to
            ``'held'``.
    """
    name: str
    ram_bytes: float = 0.0
    gpu_bytes: float = 0.0
    scope: str = 'held'

    def __post_init__(self):
        if self.scope not in MEMORY_SCOPES:
            raise ValueError(f"the scope of a memory part is one of {list(MEMORY_SCOPES)}, not {self.scope!r}")


def counts_gpu_as_ram() -> bool:
    """Whether GPU memory a process holds also counts as its RAM: on Windows, where the GPU memory PyTorch holds (in
    use or cached) is part of the process's committed memory, the measure the 25 GB tutorial cap uses."""
    return sys.platform == 'win32'


def fixed_overhead() -> MemoryPart:
    """Python, PyTorch and the CUDA context (:data:`FIXED_OVERHEAD_BYTES`), as the ``'overhead'`` part every estimate adds."""
    return MemoryPart('Python, PyTorch and CUDA', ram_bytes=FIXED_OVERHEAD_BYTES, scope='overhead')


#: Memory PyTorch's CPU allocator keeps after tensors are freed, on Windows, as a fraction of the computation's own
#: arrays (held, subset and chunk parts). PyTorch's allocator there (mimalloc) seldom gives freed memory back during a
#: long computation; calibrated against the PET tutorials' measured peaks.
WINDOWS_ALLOCATOR_FRACTION = 0.35


def memory_estimate(title: str, parts: Sequence[MemoryPart], alternatives: Sequence[tuple[str, Sequence[MemoryPart]]] = ()) -> MemoryEstimate:
    """A :class:`MemoryEstimate` of a computation's own arrays (``parts``), with the overheads every estimate adds: Python,
    PyTorch and CUDA (:func:`fixed_overhead`) and, on Windows, the memory PyTorch's allocator keeps after freeing
    (:data:`WINDOWS_ALLOCATOR_FRACTION` of the arrays). The alternatives get the same overheads, so their totals compare
    with the main one. The ``estimate_memory`` methods build their estimates with it.

    Args:
        title (str): What is estimated, with its settings.
        parts (Sequence[MemoryPart]): The computation's arrays (no overhead parts).
        alternatives (Sequence[tuple[str, Sequence[MemoryPart]]], optional): Other settings and their arrays, for the
            "To use less" line. Defaults to none.

    Returns:
        MemoryEstimate: The estimate.
    """
    def with_overheads(parts):
        parts = list(parts)
        overheads = [fixed_overhead()]
        if counts_gpu_as_ram():
            arrays = sum(p.ram_bytes for p in parts if p.scope != 'overhead')
            if arrays > 0:
                overheads.append(MemoryPart('memory the allocator keeps (Windows)', ram_bytes=WINDOWS_ALLOCATOR_FRACTION * arrays, scope='overhead'))
        return parts + overheads
    return MemoryEstimate(title, with_overheads(parts), [(label, MemoryEstimate(label, with_overheads(p))) for label, p in alternatives])


def nbytes(x) -> tuple[float, float]:
    """Memory an array keeps, as (RAM bytes, GPU bytes): a tensor's storage on its device, or what a
    :class:`~pytomography.io.PET.shared.LazySinogram` keeps (``memory_bytes``, on the host); 0 for anything else."""
    if isinstance(x, torch.Tensor):
        size = x.untyped_storage().nbytes()
        return (0.0, float(size)) if x.device.type == 'cuda' else (float(size), 0.0)
    return float(getattr(x, 'memory_bytes', 0)), 0.0


def _gb(nbytes: float) -> str:
    return f"{nbytes / 1e9:.1f}" if nbytes >= 0.05e9 else f"{nbytes / 1e9:.2f}"


class MemoryEstimate:
    """The predicted peak memory of a computation: the arrays it holds (:class:`MemoryPart`), their total RAM and GPU
    memory, and what other settings would need. ``print`` it to see the estimate; ``ram_gb`` and ``gpu_gb`` give the
    numbers to code.

    On Windows the GPU memory also counts as RAM (:func:`counts_gpu_as_ram`), so ``ram_gb`` includes it there.

    Args:
        title (str): What is estimated, with its settings (e.g. ``'PET sinogram reconstruction, 14 subsets'``).
        parts (Sequence[MemoryPart]): The arrays, including the overhead.
        alternatives (Sequence[tuple[str, MemoryEstimate]], optional): Other settings and their estimates, for the
            "To use less" line (e.g. ``('28 subsets', estimate_28)``). Defaults to none.
    """
    def __init__(self, title: str, parts: Sequence[MemoryPart], alternatives: Sequence[tuple[str, MemoryEstimate]] = ()) -> None:
        self.title = title
        self.parts = list(parts)
        self.alternatives = list(alternatives)

    @property
    def gpu_gb(self) -> float:
        """Peak GPU memory, in GB."""
        return sum(p.gpu_bytes for p in self.parts) / 1e9

    @property
    def ram_gb(self) -> float:
        """Peak RAM, in GB (on Windows, with the GPU memory)."""
        ram = sum(p.ram_bytes for p in self.parts)
        if counts_gpu_as_ram():
            ram += sum(p.gpu_bytes for p in self.parts)
        return ram / 1e9

    def __str__(self) -> str:
        lines = [self.title, f"Peak ~ {self.ram_gb:.1f} GB RAM, {self.gpu_gb:.1f} GB GPU"
                 + (" (on Windows, GPU memory counts as RAM too)" if counts_gpu_as_ram() and self.gpu_gb >= 0.05 else "")]
        for scope, label in MEMORY_SCOPES.items():
            parts = [p for p in self.parts if p.scope == scope]
            if not parts:
                continue
            ram, gpu = sum(p.ram_bytes for p in parts), sum(p.gpu_bytes for p in parts)
            items = " · ".join(p.name + " " + " + ".join(s for s in (f"{_gb(p.ram_bytes)} RAM" if p.ram_bytes else "",
                                                                     f"{_gb(p.gpu_bytes)} GPU" if p.gpu_bytes else "") if s)
                               for p in parts)
            lines.append(f"  {label}: {_gb(ram)} GB RAM, {_gb(gpu)} GB GPU ({items})")
        if self.alternatives:
            lines.append("To use less: " + " · ".join(f"{label} ~ {estimate.ram_gb:.1f} GB RAM, {estimate.gpu_gb:.1f} GB GPU"
                                                      for label, estimate in self.alternatives))
        lines.append("(plus whatever else your script keeps)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()
