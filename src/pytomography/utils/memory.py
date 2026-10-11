"""Memory budgets, estimates and plans. The host (RAM) budget set with :func:`pytomography.set_memory_budget` sizes the
blocks and subsets of PyTomography's memory-heavy steps (without a budget, every step keeps its fixed default); the GPU
budget of chunked computations keeps them from taking over a shared GPU.

Two questions can be asked of a computation before running it:

* settings in, memory out: the ``estimate_memory`` methods (e.g.
  :meth:`pytomography.projectors.SystemMatrix.estimate_memory`) return a :class:`MemoryEstimate`, the predicted peak RAM
  and GPU memory of the computation with given settings, what it is made of, and what other settings would need;
* memory in, settings out: the ``plan_memory`` methods (e.g. :meth:`pytomography.projectors.SystemMatrix.plan_memory`)
  return a :class:`MemoryPlan`, the fastest settings whose estimate fits a RAM and a GPU budget (by default what is free
  on this machine now, :func:`available_memory`), keeping the settings given.

Both print as short tables and convert to JSON (``to_dict()``), for people and for programs planning a reconstruction
for the hardware they have."""
from __future__ import annotations
import itertools
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
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


#: Python, PyTorch and CUDA when PyTomography runs on a CUDA device: the committed memory of a reconstruction's process
#: besides its arrays and PyTorch's caches. Measured on Windows with an RTX 5090 (10 Oct 2026): 1.85 GB once it has
#: imported pytomography and made its first CUDA call (``import torch`` alone is 0.92 GB), 2.72 GB with parallelproj
#: loaded, 3.03 GB with cuBLAS, cuFFT, cuDNN and CuPy used. CUDA then loads the code of each kind of kernel the first
#: time it runs (0.42 GB for the first division, 0.17 GB for the first masked assignment), so a whole reconstruction's
#: process holds more: 3.1 to 3.6 GB in the reconstructions of ``tests/calibrate_memory.py`` (most for SPECT and
#: StarGuide, which use the most kinds of kernels), above their arrays and PyTorch's cache. Rounded up.
FIXED_OVERHEAD_BYTES = 3.7e9

#: Python and PyTorch when PyTomography runs on the CPU: 1.97 GB measured as for :data:`FIXED_OVERHEAD_BYTES` with no
#: CUDA device (importing parallelproj still loads the CUDA runtime), rounded up.
FIXED_OVERHEAD_CPU_BYTES = 2.2e9

#: PyTorch's CUDA cache: what it holds (``torch.cuda.memory_reserved``) above the arrays in use. It rounds each block
#: up (an array of 1 to 10 MB takes a 20 MB segment) and keeps freed blocks for reuse; in the reconstructions of
#: ``tests/calibrate_memory.py`` the memory reserved ran 0.01 to 0.3 GB above the peak allocated, at most 30% above it
#: (13% for more than 1 GB). Every estimate with GPU arrays adds this much and :data:`CUDA_CACHE_FRACTION` of them.
CUDA_CACHE_BYTES = 0.05e9

#: See :data:`CUDA_CACHE_BYTES`.
CUDA_CACHE_FRACTION = 0.2

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


def fixed_overhead_bytes(device: str | torch.device | None = None) -> float:
    """Committed memory of Python, PyTorch and CUDA in a process running PyTomography on ``device`` (default
    ``pytomography.device``): :data:`FIXED_OVERHEAD_BYTES` on a CUDA device, :data:`FIXED_OVERHEAD_CPU_BYTES` otherwise."""
    device = torch.device(pytomography.device if device is None else device)
    return FIXED_OVERHEAD_BYTES if device.type == 'cuda' else FIXED_OVERHEAD_CPU_BYTES


def fixed_overhead() -> MemoryPart:
    """Python, PyTorch and the CUDA context (:func:`fixed_overhead_bytes`), as the ``'overhead'`` part every estimate adds."""
    return MemoryPart('Python, PyTorch and CUDA', ram_bytes=fixed_overhead_bytes(), scope='overhead')


def cuda_cache(gpu_bytes: float) -> MemoryPart | None:
    """PyTorch's CUDA cache above ``gpu_bytes`` of arrays (:data:`CUDA_CACHE_BYTES` and :data:`CUDA_CACHE_FRACTION` of
    them), as an ``'overhead'`` part; None without GPU arrays."""
    if gpu_bytes <= 0:
        return None
    return MemoryPart("PyTorch's GPU cache (rounded and kept blocks)", gpu_bytes=CUDA_CACHE_BYTES + CUDA_CACHE_FRACTION * gpu_bytes, scope='overhead')


#: Memory PyTorch's CPU allocator keeps after tensors are freed, on Windows, as a fraction of the computation's own
#: arrays (held, subset and chunk parts). PyTorch's allocator there (mimalloc) seldom gives freed memory back during a
#: long computation; calibrated against the PET tutorials' measured peaks.
WINDOWS_ALLOCATOR_FRACTION = 0.35


def memory_estimate(title: str, parts: Sequence[MemoryPart], alternatives: Sequence[tuple] = (), allocator_fraction: float | None = None,
                    settings: Mapping | None = None) -> MemoryEstimate:
    """A :class:`MemoryEstimate` of a computation's own arrays (``parts``), with the overheads every estimate adds: Python,
    PyTorch and CUDA (:func:`fixed_overhead`), PyTorch's CUDA cache above the GPU arrays (:func:`cuda_cache`) and, on
    Windows, the memory PyTorch's allocator keeps after freeing (:data:`WINDOWS_ALLOCATOR_FRACTION` of the arrays). The
    alternatives get the same overheads, so their totals compare with the main one. The ``estimate_memory`` methods
    build their estimates with it.

    Args:
        title (str): What is estimated, with its settings.
        parts (Sequence[MemoryPart]): The computation's arrays (no overhead parts).
        alternatives (Sequence[tuple], optional): Other settings and their arrays, for the "To use less" line, as
            ``(label, parts)`` or ``(label, parts, settings)``. Defaults to none.
        allocator_fraction (float | None, optional): The Windows allowance as a fraction of the arrays (RAM and GPU),
            for computations measured to need another. Defaults to None: :data:`WINDOWS_ALLOCATOR_FRACTION`.
        settings (Mapping | None, optional): The settings the estimate is for, by name (e.g. ``{'n_subsets': 14}``),
            kept in :attr:`MemoryEstimate.settings`. Defaults to none.

    Returns:
        MemoryEstimate: The estimate.
    """
    def with_overheads(parts):
        parts = list(parts)
        overheads = [fixed_overhead()]
        cache = cuda_cache(sum(p.gpu_bytes for p in parts if p.scope != 'overhead'))
        if cache is not None:
            overheads.append(cache)
        if counts_gpu_as_ram():
            # what PyTorch's allocators keep after freeing (its CPU allocator, and its GPU cache, which Windows also counts)
            arrays = sum(p.ram_bytes + p.gpu_bytes for p in parts if p.scope != 'overhead')
            fraction = WINDOWS_ALLOCATOR_FRACTION if allocator_fraction is None else allocator_fraction
            if arrays > 0:
                overheads.append(MemoryPart('memory the allocators keep (Windows)', ram_bytes=fraction * arrays, scope='overhead'))
        return parts + overheads
    others = []
    for alternative in alternatives:
        label, alternative_parts = alternative[0], alternative[1]
        alternative_settings = alternative[2] if len(alternative) > 2 else None
        others.append((label, MemoryEstimate(label, with_overheads(alternative_parts), settings=alternative_settings)))
    return MemoryEstimate(title, with_overheads(parts), others, settings=settings)


def tensor_nbytes(x: torch.Tensor) -> float:
    """Bytes of a tensor's own elements: its number of elements times their size. A view counts at its own size, not
    its base tensor's: a larger base (e.g. the projections of every energy window that ``photopeak = projections[0]``
    is taken from) is kept by the caller, not by the computation. An expanded view (stride 0) counts no more than the
    storage it repeats.

    Args:
        x (torch.Tensor): The tensor.

    Returns:
        float: Bytes.
    """
    return float(min(x.numel() * x.element_size(), x.untyped_storage().nbytes()))


def nbytes(x) -> tuple[float, float]:
    """Memory an array keeps, as (RAM bytes, GPU bytes): a tensor's elements (:func:`tensor_nbytes`) on its device, or
    what a :class:`~pytomography.io.PET.shared.LazySinogram` keeps (``memory_bytes``, on the host); 0 for anything
    else."""
    if isinstance(x, torch.Tensor):
        size = tensor_nbytes(x)
        return (0.0, size) if x.device.type == 'cuda' else (size, 0.0)
    return float(getattr(x, 'memory_bytes', 0)), 0.0


def tensor_bytes(obj, exclude: Sequence[str] = ()) -> tuple[float, float]:
    """Memory the tensors an object keeps as attributes take, as (RAM bytes, GPU bytes): its tensor attributes and the
    tensors in its list, tuple and dict attributes (one level deep), each at its own size (:func:`tensor_nbytes`) and
    each counted once. Estimates use it for the arrays of transforms they do not know in detail.

    Args:
        obj: Any object (None gives 0).
        exclude (Sequence[str], optional): Attribute names to leave out. Defaults to none.

    Returns:
        tuple[float, float]: RAM and GPU bytes.
    """
    ram = gpu = 0.0
    seen = set()
    def add(x):
        nonlocal ram, gpu
        if isinstance(x, torch.Tensor):
            key = (str(x.device), x.data_ptr(), tensor_nbytes(x))
            if key not in seen:
                seen.add(key)
                r, g = nbytes(x)
                ram, gpu = ram + r, gpu + g
    for name, value in vars(obj).items() if obj is not None and hasattr(obj, '__dict__') else ():
        if name in exclude:
            continue
        if isinstance(value, dict):
            for item in value.values():
                add(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)
        else:
            add(value)
    return ram, gpu


def _gb(nbytes: float) -> str:
    return f"{nbytes / 1e9:.1f}" if nbytes >= 0.05e9 else f"{nbytes / 1e9:.2f}"


def _round_gb(gb: float) -> float | None:
    """GB rounded to the kilobyte, for JSON (None for an infinite or undefined number)."""
    return round(float(gb), 6) if math.isfinite(gb) else None


def _jsonable(value):
    """``value`` as plain Python values that ``json.dumps`` takes: numbers, strings, booleans, None, lists and dicts
    (tensors and NumPy scalars become numbers or lists, infinite numbers None, anything else its ``str``)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, torch.Tensor):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, range)):
        return [_jsonable(v) for v in value]
    if hasattr(value, 'item') and callable(value.item):     # NumPy scalars
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _settings_text(settings: Mapping) -> str:
    """Settings as ``name=value`` pairs (``'n_subsets=14, N_splits=1'``), or ``'its present settings'``."""
    def value(v):
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    return ", ".join(f"{k}={value(v)}" for k, v in settings.items()) if settings else "its present settings"


class MemoryEstimate:
    """The predicted peak memory of a computation: the arrays it holds (:class:`MemoryPart`), their total RAM and GPU
    memory, and what other settings would need. ``print`` it to see the estimate; ``ram_gb`` and ``gpu_gb`` give the
    numbers to code, and :meth:`to_dict` all of it as JSON.

    On Windows the GPU memory also counts as RAM (:func:`counts_gpu_as_ram`), so ``ram_gb`` includes it there.

    Args:
        title (str): What is estimated, with its settings (e.g. ``'PET sinogram reconstruction, 14 subsets'``).
        parts (Sequence[MemoryPart]): The arrays, including the overhead.
        alternatives (Sequence[tuple[str, MemoryEstimate]], optional): Other settings and their estimates, for the
            "To use less" line (e.g. ``('28 subsets', estimate_28)``). Defaults to none.
        settings (Mapping | None, optional): The settings this estimate is for, by name (e.g.
            ``{'n_subsets': 14, 'N_splits': 1}``). Defaults to none.
    """
    def __init__(self, title: str, parts: Sequence[MemoryPart], alternatives: Sequence[tuple[str, MemoryEstimate]] = (),
                 settings: Mapping | None = None) -> None:
        self.title = title
        self.parts = list(parts)
        self.alternatives = list(alternatives)
        self.settings = dict(settings or {})

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
            items = ", ".join(p.name + " " + " + ".join(s for s in (f"{_gb(p.ram_bytes)} RAM" if p.ram_bytes else "",
                                                                     f"{_gb(p.gpu_bytes)} GPU" if p.gpu_bytes else "") if s)
                               for p in parts)
            lines.append(f"  {label}: {_gb(ram)} GB RAM, {_gb(gpu)} GB GPU ({items})")
        if self.alternatives:
            lines.append("To use less: " + ", ".join(f"{label} ~ {estimate.ram_gb:.1f} GB RAM, {estimate.gpu_gb:.1f} GB GPU"
                                                      for label, estimate in self.alternatives))
        lines.append("(plus whatever else your script keeps)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict:
        """The estimate as plain Python values, which ``json.dumps`` takes (and ``json.loads`` gives back): ``title``;
        ``settings``, the settings it is for; ``ram_gb`` and ``gpu_gb``, the peak in GB; ``gpu_counts_as_ram``, whether
        ``ram_gb`` includes the GPU memory (on Windows); ``parts``, each as ``{name, ram_gb, gpu_gb, scope}``; and
        ``alternatives``, each as ``{label, ram_gb, gpu_gb, settings}``. GB are rounded to the kilobyte.

        Returns:
            dict: The estimate.

        Example:
            >>> json.dumps(system_matrix.estimate_memory(n_subsets=14).to_dict())
        """
        return {'title': self.title,
                'settings': _jsonable(self.settings),
                'ram_gb': _round_gb(self.ram_gb),
                'gpu_gb': _round_gb(self.gpu_gb),
                'gpu_counts_as_ram': counts_gpu_as_ram(),
                'parts': [{'name': p.name, 'ram_gb': _round_gb(p.ram_bytes / 1e9), 'gpu_gb': _round_gb(p.gpu_bytes / 1e9), 'scope': p.scope}
                          for p in self.parts],
                'alternatives': [{'label': label, 'ram_gb': _round_gb(e.ram_gb), 'gpu_gb': _round_gb(e.gpu_gb), 'settings': _jsonable(e.settings)}
                                 for label, e in self.alternatives]}


# ---------------------------------------------------------------------------------------------- memory in, settings out

#: RAM :func:`available_memory` keeps free for everything else on the machine (the operating system, other programs and
#: jobs): 16 GB, or :data:`RAM_MARGIN_FRACTION` of the machine's RAM if that is less (machines with less than 64 GB).
RAM_MARGIN_BYTES = 16e9

#: Fraction of the machine's RAM :func:`available_memory` keeps free on machines with less than 64 GB (4 GB of 16 GB).
RAM_MARGIN_FRACTION = 0.25

#: Fraction of a GPU's memory :func:`available_memory` keeps free: the workspaces CUDA's libraries allocate as they go
#: (cuFFT plans, cuBLAS) and the fragmentation of PyTorch's cache, which estimates do not count.
GPU_MARGIN_FRACTION = 0.05

#: Most subsets :meth:`~pytomography.projectors.SystemMatrix.plan_memory` tries (fewer if there are fewer views).
PLAN_MAX_SUBSETS = 256


def split_candidates(present: int | None = None, most: int = 256) -> list[int]:
    """The projector splits ``N_splits`` a plan tries, fewest (fastest) first: 1, 2, 4, ... ``most``, and ``present``.

    Args:
        present (int | None, optional): The present value, tried too. Defaults to None.
        most (int, optional): Most splits. Defaults to 256.

    Returns:
        list[int]: The candidates, ascending.
    """
    values = {2 ** k for k in range(int(math.log2(most)) + 1)}
    if present is not None:
        values.add(int(present))
    return sorted(values)


def _windows_memory() -> tuple[float, float, float] | None:
    """(total physical, available physical, available commit) bytes on Windows, from GlobalMemoryStatusEx; None if it
    cannot be read."""
    import ctypes
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
    except Exception:
        return None
    return float(status.ullTotalPhys), float(status.ullAvailPhys), float(status.ullAvailPageFile)


def _windows_process_bytes() -> float:
    """Memory this process has committed (its private bytes) on Windows, from GetProcessMemoryInfo; 0 if unknown."""
    import ctypes
    from ctypes import wintypes
    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
                    ('PrivateUsage', ctypes.c_size_t)]
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
        if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return 0.0
    except Exception:
        return 0.0
    return float(counters.PrivateUsage)


def _host_memory() -> tuple[float, float, float]:
    """Host memory in bytes, as (total, available, this process). On Windows "available" is the smaller of the free
    physical memory and the free commit (memory can be committed only up to the commit limit, and GPU memory PyTorch
    holds is committed too), and "this process" its private bytes; on Linux, ``MemAvailable`` and the resident size;
    elsewhere what psutil reports, if it is installed."""
    total = available = process = None
    if sys.platform == 'win32':
        status = _windows_memory()
        if status is not None:
            total, physical, commit = status
            available = min(physical, commit)
        process = _windows_process_bytes()
    elif os.path.exists('/proc/meminfo'):
        fields = {}
        with open('/proc/meminfo') as f:
            for line in f:
                key, _, rest = line.partition(':')
                fields[key] = float(rest.split()[0]) * 1024
        total, available = fields.get('MemTotal'), fields.get('MemAvailable', fields.get('MemFree'))
        try:
            with open('/proc/self/statm') as f:
                process = float(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, IndexError):
            process = None
    if total is None or available is None or process is None:
        try:
            import psutil
            memory = psutil.virtual_memory()
            total = float(memory.total) if total is None else total
            available = float(memory.available) if available is None else available
            if process is None:
                info = psutil.Process().memory_info()
                process = float(getattr(info, 'private', 0) or info.rss)
        except Exception:
            pass
    if total is None:
        try:
            total = float(os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
        except (AttributeError, ValueError, OSError):
            total = 0.0
    if available is None:
        try:
            available = float(os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
        except (AttributeError, ValueError, OSError):
            available = total / 2                     # unknown: assume half the machine is in use
    if not process:
        try:                                          # macOS without psutil: the peak resident size (bytes there)
            import resource
            peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            process = peak if sys.platform == 'darwin' else peak * 1024
        except (ImportError, AttributeError, ValueError, OSError):
            process = 0.0                             # unknown: nothing is added back to the budget
    return float(total), float(available), float(process or 0.0)


def _gpu_memory(device) -> tuple[float, float]:
    """(free, total) bytes of a CUDA device for this process: the device's free memory plus the memory PyTorch's cache
    holds unused; (0, 0) without a CUDA device."""
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info(device)
    cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return float(free + max(0, cached)), float(total)


def available_memory(device: str | torch.device | None = None) -> tuple[float, float]:
    """RAM and GPU memory, in GB, a computation in this process can plan for now: what is free on the machine, less a
    margin kept free for everything else (:data:`RAM_MARGIN_BYTES`, 16 GB, or :data:`RAM_MARGIN_FRACTION` of the RAM on
    machines with less than 64 GB; :data:`GPU_MARGIN_FRACTION` of the GPU). The RAM also includes the Python, PyTorch
    and CUDA overhead this process already holds (at most :func:`fixed_overhead_bytes`), since every estimate counts it;
    arrays the process already holds are not added back, since an estimate may not count them. The ``plan_memory``
    methods use it when they are not given budgets.

    On Windows GPU memory counts as RAM (:func:`counts_gpu_as_ram`): the RAM returned is what is left for both, and a plan
    checks an estimate's ``ram_gb``, which includes its GPU memory there, against it.

    Args:
        device (str | torch.device | None, optional): The GPU. Defaults to ``pytomography.device``.

    Returns:
        tuple[float, float]: (RAM in GB, GPU memory in GB); the GPU memory is 0 without a CUDA device.

    Example:
        >>> ram_gb, gpu_gb = available_memory()
    """
    total, available, process = _host_memory()
    margin = min(RAM_MARGIN_BYTES, RAM_MARGIN_FRACTION * total)
    ram = max(0.0, available - margin + min(process, fixed_overhead_bytes()))
    free, gpu_total = _gpu_memory(pytomography.device if device is None else device)
    gpu = max(0.0, free - GPU_MARGIN_FRACTION * gpu_total)
    return ram / 1e9, gpu / 1e9


class MemoryPlan:
    """Settings of a computation chosen to fit a memory budget, which the ``plan_memory`` methods return (e.g.
    :meth:`pytomography.projectors.SystemMatrix.plan_memory`): the fastest settings whose estimate fits both the RAM and
    the GPU budget, keeping the settings that were given, with their estimate and every setting tried. If nothing fits,
    ``fits`` is False and the settings are those that come closest. ``print`` it to see the settings chosen and those
    tried; :meth:`to_dict` gives it as JSON.

    Args:
        title (str): What is planned (e.g. ``'OSEM (PETSinogramSystemMatrix)'``).
        settings (Mapping): The settings chosen, by name (e.g. ``{'N_splits': 1, 'n_subsets': 8}``).
        estimate (MemoryEstimate): The estimate of the settings chosen.
        fits (bool): Whether that estimate fits both budgets.
        options (Sequence[Mapping]): Every setting tried, in the order tried, each as ``{'settings', 'ram_gb', 'gpu_gb',
            'fits'}``.
        ram_gb (float): The RAM budget, in GB (on Windows it covers GPU memory too).
        gpu_gb (float): The GPU budget, in GB.
        budget_from (Mapping | None, optional): Where each budget came from, ``'given'`` or ``'available'``
            (:func:`available_memory`), by ``'ram_gb'`` and ``'gpu_gb'``. Defaults to both given.
        fixed (Sequence[str], optional): Names of the settings that were given rather than chosen. Defaults to none.

    Attributes:
        settings (dict): The settings chosen.
        estimate (MemoryEstimate): Their estimate.
        fits (bool): Whether they fit both budgets.
        options (list[dict]): Every setting tried.
        ram_gb (float): The RAM budget used, in GB.
        gpu_gb (float): The GPU budget used, in GB.
    """
    #: Rows of the options table ``str`` shows at most (the first, those around the choice, and the last).
    MAX_ROWS = 12

    def __init__(self, title: str, settings: Mapping, estimate: MemoryEstimate, fits: bool, options: Sequence[Mapping],
                 ram_gb: float, gpu_gb: float, budget_from: Mapping | None = None, fixed: Sequence[str] = ()) -> None:
        self.title = title
        self.settings = dict(settings)
        self.estimate = estimate
        self.fits = bool(fits)
        self.options = [dict(o) for o in options]
        self.ram_gb = float(ram_gb)
        self.gpu_gb = float(gpu_gb)
        self.budget_from = dict(budget_from or {'ram_gb': 'given', 'gpu_gb': 'given'})
        self.fixed = list(fixed)

    def _chosen_index(self) -> int | None:
        for i, option in enumerate(self.options):
            if option['settings'] == self.settings:
                return i
        return None

    def __str__(self) -> str:
        def budget(gb):
            return "no limit" if not math.isfinite(gb) else f"{gb:.1f} GB"
        sources = set(self.budget_from.values())
        notes = (["available now" if sources == {'available'} else "RAM available now" if self.budget_from.get('ram_gb') == 'available'
                  else "GPU available now"] if 'available' in sources else []) + (["on Windows, GPU memory counts as RAM too"] if counts_gpu_as_ram() else [])
        lines = [f"Memory plan for {self.title}, in {budget(self.ram_gb)} RAM and {budget(self.gpu_gb)} GPU"
                 + (f" ({'; '.join(notes)})" if notes else "")]
        chosen = f"{_settings_text(self.settings)} ~ {self.estimate.ram_gb:.1f} GB RAM, {self.estimate.gpu_gb:.1f} GB GPU"
        lines.append(("Fits: " if self.fits else "Nothing fits; closest: ") + chosen
                     + (f" (kept: {', '.join(self.fixed)})" if self.fixed else ""))
        names = list(self.options[0]['settings']) if self.options else []
        if names and len(self.options) > 1:
            def cell(value):
                value = _jsonable(value)
                return f"{value:.4g}" if isinstance(value, float) else str(value)
            widths = [max(len(n), *(len(cell(o['settings'][n])) for o in self.options)) for n in names]
            lines.append(f"Tried ({len(self.options)}):")
            lines.append("  " + "  ".join(n.rjust(w) for n, w in zip(names, widths)) + "  RAM GB  GPU GB  fits")
            c = self._chosen_index()
            n = len(self.options)
            if n <= self.MAX_ROWS:
                shown = list(range(n))
            else:
                around = {c - 1, c, c + 1} if c is not None else set()
                shown = sorted(i for i in set(range(4)) | around | {n - 2, n - 1} if 0 <= i < n)
            previous = -1
            for i in shown:
                if i > previous + 1:
                    lines.append("  ...")
                option = self.options[i]
                cells = "  ".join(cell(option['settings'][k]).rjust(w) for k, w in zip(names, widths))
                lines.append(f"  {cells}  {option['ram_gb']:6.1f}  {option['gpu_gb']:6.1f}  {'yes' if option['fits'] else 'no':4s}"
                             + (" <- chosen" if i == c else "").rstrip())
                previous = i
        return "\n".join(line.rstrip() for line in lines)

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict:
        """The plan as plain Python values, which ``json.dumps`` takes (and ``json.loads`` gives back): ``title``;
        ``fits``; ``settings``, those chosen; ``fixed``, the names of those that were given; ``budget``, the RAM and
        GPU budgets used in GB (None for no limit) and where they came from (``'given'`` or ``'available'``);
        ``gpu_counts_as_ram``; ``estimate``, the estimate of the settings chosen (:meth:`MemoryEstimate.to_dict`); and
        ``options``, every setting tried, as ``{settings, ram_gb, gpu_gb, fits}``.

        Returns:
            dict: The plan.
        """
        return {'title': self.title,
                'fits': self.fits,
                'settings': _jsonable(self.settings),
                'fixed': list(self.fixed),
                'budget': {'ram_gb': _round_gb(self.ram_gb), 'gpu_gb': _round_gb(self.gpu_gb), 'from': dict(self.budget_from)},
                'gpu_counts_as_ram': counts_gpu_as_ram(),
                'estimate': self.estimate.to_dict(),
                'options': [{'settings': _jsonable(o['settings']), 'ram_gb': _round_gb(o['ram_gb']), 'gpu_gb': _round_gb(o['gpu_gb']),
                             'fits': bool(o['fits'])} for o in self.options]}


def plan_settings(title: str, settings: Mapping[str, tuple], estimate: Callable[[dict, bool], MemoryEstimate],
                  fixed: Mapping | None = None, ram_gb: float | None = None, gpu_gb: float | None = None,
                  device: str | torch.device | None = None) -> MemoryPlan:
    """The fastest settings of a computation whose estimate fits in ``ram_gb`` of RAM and ``gpu_gb`` of GPU memory, as a
    :class:`MemoryPlan`. The ``plan_memory`` methods build their plans with it.

    The settings are tried in order: every combination of the candidates of each setting, the first setting's being the
    outer loop, each list fastest first, so that the first combination that fits is the plan (e.g. the fewest projector
    splits, then the fewest subsets that fit with them). Settings in ``fixed`` keep their value. If no combination fits,
    the plan has ``fits`` False and the combination that exceeds the budgets least; it does not raise. On Windows an
    estimate's ``ram_gb`` includes its GPU memory, so the RAM budget is checked against the two together.

    Args:
        title (str): What is planned.
        settings (Mapping[str, tuple]): The settings to choose, as ``{name: (present value, candidates)}``, in the order
            they are searched, each list of candidates fastest first.
        estimate (Callable[[dict, bool], MemoryEstimate]): The estimate of a dict of every setting; the second argument
            says whether its "To use less" alternatives are needed (only for the plan's choice).
        fixed (Mapping | None, optional): Settings to keep, by name; a value of None is not kept. Defaults to none.
        ram_gb (float | None, optional): RAM budget in GB. Defaults to None: what is available now
            (:func:`available_memory`).
        gpu_gb (float | None, optional): GPU budget in GB. Defaults to None: what is available now.
        device (str | torch.device | None, optional): The GPU whose free memory is the default GPU budget. Defaults to
            ``pytomography.device``.

    Returns:
        MemoryPlan: The plan.

    Raises:
        ValueError: ``fixed`` names a setting that is not in ``settings``.
    """
    fixed = {k: v for k, v in (fixed or {}).items() if v is not None}
    unknown = sorted(set(fixed) - set(settings))
    if unknown:
        raise ValueError(f"{title} has no setting {', '.join(unknown)}: its settings are {', '.join(settings) or 'none'}")
    budget_from = {'ram_gb': 'given' if ram_gb is not None else 'available', 'gpu_gb': 'given' if gpu_gb is not None else 'available'}
    if ram_gb is None or gpu_gb is None:
        available_ram, available_gpu = available_memory(device)
        ram_gb = available_ram if ram_gb is None else ram_gb
        gpu_gb = available_gpu if gpu_gb is None else gpu_gb
    names = list(settings)
    candidates = []
    for name in names:
        present, values = settings[name]
        values = [fixed[name]] if name in fixed else list(values)
        candidates.append(values if values else [present])
    options, chosen, closest, least = [], None, None, math.inf
    tolerance = 1e-9
    for values in itertools.product(*candidates):
        trial = dict(zip(names, values))
        e = estimate(trial, False)
        excess = max(0.0, e.ram_gb - ram_gb) + max(0.0, e.gpu_gb - gpu_gb)
        fits = excess <= tolerance
        options.append({'settings': trial, 'ram_gb': e.ram_gb, 'gpu_gb': e.gpu_gb, 'fits': fits})
        if fits:
            chosen = trial
            break
        if excess < least:
            closest, least = trial, excess
    settings_chosen = chosen if chosen is not None else closest
    return MemoryPlan(title, settings_chosen, estimate(settings_chosen, True), chosen is not None, options, ram_gb, gpu_gb,
                      budget_from, [n for n in names if n in fixed])
