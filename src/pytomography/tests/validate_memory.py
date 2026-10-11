"""Validates PyTomography's memory estimates and plans on synthetic data, before a release. Not a test (pytest does not
collect it): it needs a GPU, takes minutes, and measures whole processes::

    python src/pytomography/tests/validate_memory.py --quick              # one case per model, a few plans (minutes)
    python src/pytomography/tests/validate_memory.py --full               # every model at 2-3 sizes, sweeps, plans
    python src/pytomography/tests/validate_memory.py --full --only pet-lm --out D:/v4runs/memval
    python src/pytomography/tests/validate_memory.py --list --full
    python src/pytomography/tests/validate_memory.py --report-only --out D:/v4runs/memval

Run it as a script (not with ``-m``), so that each case's process caps its memory before it imports PyTorch.

What a case does, in a process of its own (Windows' peak commit is per process):

1. caps its own committed memory (a Windows job object, as ``docs/tools/run_tutorials.py`` does; a case that goes
   over fails) and PyTorch's GPU memory (``torch.cuda.set_per_process_memory_fraction``);
2. makes random data of the right shape where the case keeps it (memory depends on shapes, not values): Poisson
   projections, list mode events between random valid detector pairs with random TOF bins, random images and maps;
3. builds the system matrix, likelihood and algorithm, asks for ``estimate_memory`` (or, for a plan case,
   ``plan_memory(ram_gb, gpu_gb)`` with ``pytomography.set_memory_budget(ram_gb)``, and applies the plan's settings);
4. sets up the subsets (the H_m^T 1 of every subset; for list mode, every subset's projector order), then runs ONE
   subiteration (the largest subset, the first), a second if the first took under 2 s (for a steady time). Filtered
   back projection runs once; the single scatter simulation runs once, whole (its scaling step comes last);
5. reports the estimate and what was measured: the process's peak committed memory (Windows: peak pagefile usage,
   which includes the GPU memory PyTorch holds; elsewhere peak resident size), PyTorch's peak reserved and allocated
   GPU memory, and the wall time of the setup and of a subiteration.

The driver runs the GPU cases one at a time (and CPU-only cases, with CUDA hidden, beside them), samples the GPU's
memory with nvidia-smi while each runs (per-process GPU memory is not reported on Windows, so the device's use above
the level before the case started), and writes ``results.jsonl``, ``results.json`` and ``report.md`` to ``--out``.

An estimate passes when it is at least what was measured: ``process peak / ram_gb`` and ``GPU reserved / gpu_gb`` at
most 1. A plan passes when its reconstruction runs under caps of exactly its budgets.

Safety: every case is capped at 25 GB of committed memory (less if the machine has less than 16 GB free beyond it) and
10 GB of GPU memory, and is skipped before its reconstruction if its estimate is above either. With ``--lock`` the
driver holds the shared GPU lock directory while it runs GPU cases.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

RAM_LIMIT_GB = 25.0          # most any one case may commit
GPU_LIMIT_GB = 10.0          # most any one case may hold on the GPU
KEEP_FREE_GB = 16.0          # RAM kept free on the machine
PLAN_RAM_GB = (6, 8, 12, 16)
PLAN_GPU_GB = (2, 4, 8)


# =============================================================================================== Windows process memory
def _win_counters():
    """(private bytes, peak pagefile usage) of this process on Windows; (0, 0) elsewhere."""
    if sys.platform != 'win32':
        return 0.0, 0.0
    from ctypes import wintypes

    class COUNTERS(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t), ('PrivateUsage', ctypes.c_size_t)]
    c = COUNTERS()
    c.cb = ctypes.sizeof(COUNTERS)
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(COUNTERS), wintypes.DWORD]
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return float(c.PrivateUsage), float(c.PeakPagefileUsage)


def _peak_commit_bytes() -> float:
    """This process's peak committed memory (Windows), or its peak resident size elsewhere."""
    if sys.platform == 'win32':
        return _win_counters()[1]
    import resource
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))


def _commit_bytes() -> float:
    if sys.platform == 'win32':
        return _win_counters()[0]
    try:
        with open('/proc/self/statm') as f:
            return float(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except OSError:
        return 0.0


_JOB = None


def _cap_this_process(limit_gb: float) -> bool:
    """Puts this process in a Windows job object whose committed memory is capped at ``limit_gb``: an allocation past
    it fails (``MemoryError``, or PyTorch's or CUDA's out of memory error), so a case that needs more fails instead of
    taking the machine's memory. As ``docs/tools/run_tutorials.py`` caps the tutorials. False where it cannot."""
    global _JOB
    if sys.platform != 'win32':
        return False
    from ctypes import wintypes
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    class BASIC(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong),
                    ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                    ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                    ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', BASIC), ('IoInfo', ctypes.c_ulonglong * 6),
                    ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                    ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x200          # JOB_OBJECT_LIMIT_JOB_MEMORY
    info.JobMemoryLimit = int(limit_gb * 1e9)
    if not job or not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        return False
    if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        return False
    _JOB = job                                             # kept open for the life of the process
    return True


def _machine_free_gb() -> float:
    """The smaller of free RAM and free commit, in GB."""
    if sys.platform == 'win32':
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ('ullTotalPhys', 'ullAvailPhys', 'ullTotalPageFile', 'ullAvailPageFile',
                                                  'ullTotalVirtual', 'ullAvailVirtual', 'ullAvailExtendedVirtual')]
        s = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return min(s.ullAvailPhys, s.ullAvailPageFile) / 1e9
    import psutil
    return psutil.virtual_memory().available / 1e9


# =============================================================================================== synthetic data
# Everything below runs in a case's process, after its caps are set. Values are random: memory depends on shapes.

def _blockwise(shape, device, fill, rows: int | None = None):
    """A float32 tensor of ``shape`` on ``device`` filled ``rows`` rows (first dimension) at a time by ``fill(n_rows)``,
    so that making it never holds more than a block of temporaries beside it."""
    import torch
    out = torch.empty(tuple(shape), dtype=torch.float32, device=device)
    per_row = max(1, math.prod(shape[1:]))
    rows = rows or max(1, int(2 ** 24 // per_row))
    for start in range(0, shape[0], rows):
        n = min(rows, shape[0] - start)
        out[start:start + n] = fill((n, *shape[1:]))
    return out


def _poisson(shape, device, mean=5.0):
    import torch
    return _blockwise(shape, device, lambda s: torch.poisson(mean * torch.rand(s, device=device)))


def _uniform(shape, device, low=0.0, high=1.0):
    import torch
    return _blockwise(shape, device, lambda s: low + (high - low) * torch.rand(s, device=device))


def _pet_info(crystals: int, rings: int) -> dict:
    """A polygonal scanner of ``crystals`` per ring (blocks of 8 x 8 crystals of 4 mm) and ``rings`` rings."""
    sectors = crystals // 8
    apothem = 32.0 / (2 * math.tan(math.pi / sectors))
    return dict(min_rsector_difference=0, crystal_length=20.0, radius=round(1.03 * apothem, 1), firstCrystalAxis=0,
                rsectorTransNr=sectors, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=rings // 8, moduleTransSpacing=0.0,
                moduleAxialSpacing=33.0, submoduleTransNr=1, submoduleAxialNr=1, submoduleTransSpacing=0.0,
                submoduleAxialSpacing=0.0, crystalTransNr=8, crystalAxialNr=8, crystalTransSpacing=4.0,
                crystalAxialSpacing=4.0, NrCrystalsPerRing=crystals, NrRings=rings)


#: PET scanners: (crystals per ring, rings, image (Nxy, Nz, dz mm)) - small, medium, and a Biograph mMR-sized one
SCANNERS = {'S': (256, 16, (128, 32, 2.0)), 'M': (384, 32, (192, 64, 2.0)), 'L': (448, 64, (256, 128, 2.0))}


def _random_events(info: dict, n: int, tof_bins: int, device='cpu'):
    """``n`` list mode events between random valid detector pairs (the second crystal in the opposite half of its ring,
    any two rings) with random TOF bins, as int32 detector IDs (and TOF bin)."""
    import torch
    C, R = info['NrCrystalsPerRing'], info['NrRings']
    ids = torch.empty((n, 3 if tof_bins else 2), dtype=torch.int32, device=device)
    for s in range(0, n, 2 ** 24):
        m = min(2 ** 24, n - s)
        c0 = torch.randint(0, C, (m,), device=device)
        c1 = (c0 + C // 2 + torch.randint(-(C // 4), C // 4 + 1, (m,), device=device)) % C
        ids[s:s + m, 0] = (torch.randint(0, R, (m,), device=device) * C + c0).to(torch.int32)
        ids[s:s + m, 1] = (torch.randint(0, R, (m,), device=device) * C + c1).to(torch.int32)
        if tof_bins:
            ids[s:s + m, 2] = torch.randint(0, tof_bins, (m,), device=device, dtype=torch.int32)
    return ids


def _tof_meta(bins: int, info: dict):
    from pytomography.metadata.PET import PETTOFMeta
    return PETTOFMeta(bins, 2.0 * info['radius'], 60.0, n_sigmas=3) if bins else None


def _pet_object(scanner: str, p: dict):
    from pytomography.metadata import ObjectMeta
    _, _, (n, nz, dz) = SCANNERS[scanner]
    n, nz, dz = p.get('n', n), p.get('nz', nz), p.get('dz', dz)
    return ObjectMeta(dr=(2.0, 2.0, dz), shape=(n, n, nz))


def _algorithm(likelihood, name: str = 'OSEM', prior: str | None = None):
    from pytomography import algorithms
    from pytomography.priors import RelativeDifferencePrior
    kwargs = {}
    if prior == 'RDP' or (prior is None and name in ('BSREM', 'OSMAPOSL')):
        kwargs['prior'] = RelativeDifferencePrior(beta=1.0)
    return getattr(algorithms, name)(likelihood, **kwargs)


class Case:
    """What a case's process runs: ``estimate(settings)``, ``plan(ram_gb, gpu_gb)``, ``apply(settings)``, ``setup()`` (the
    subsets and their H_m^T 1), ``step()`` (one subiteration, or the whole computation for FBP and SSS)."""
    def __init__(self, algorithm=None, n_subsets: int = 1, system_matrix=None):
        self.algorithm, self.n_subsets, self.sm = algorithm, n_subsets, system_matrix
        self.settings = {}
        self.iterations = 0         # > 0: step() runs this many whole iterations in one call, as a reconstruction does

    def estimate(self):
        kwargs = {'n_subsets': self.n_subsets}
        if self.settings.get('N_splits') is not None:
            kwargs['N_splits'] = self.settings['N_splits']
        return self.algorithm.estimate_memory(**kwargs)

    def plan(self, ram_gb, gpu_gb):
        return self.algorithm.plan_memory(ram_gb=ram_gb, gpu_gb=gpu_gb)

    def apply(self, settings: dict):
        if 'n_subsets' in settings:
            self.n_subsets = int(settings['n_subsets'])
        if settings.get('N_splits') is not None and hasattr(self.sm, 'N_splits'):
            self.sm.N_splits = int(settings['N_splits'])
            self.settings['N_splits'] = int(settings['N_splits'])

    def setup(self):
        self.algorithm._set_n_subsets(self.n_subsets)
        sm = self.algorithm.likelihood.system_matrix
        if hasattr(sm, '_event_order') and self.n_subsets > 1:
            for k in range(self.n_subsets):          # every subset's projector order, kept as a full iteration keeps them
                sm._event_order(k)

    def step(self):
        if self.iterations:
            self.algorithm(self.iterations, self.n_subsets)
        elif type(self.algorithm).__name__ == 'MLEM':   # one subset: an iteration is its subiteration
            self.algorithm(1)
        else:
            self.algorithm(1, self.n_subsets, n_subset_specific=0)

    @property
    def subiterations(self) -> int:
        return self.n_subsets


# ------------------------------------------------------------------------------------------------------- PET sinogram
def build_pet_sino(p: dict) -> Case:
    import torch
    import pytomography
    from pytomography.metadata.PET import PETSinogramPolygonProjMeta
    from pytomography.projectors.PET import PETSinogramSystemMatrix
    from pytomography.likelihoods import PoissonLogLikelihood
    from pytomography.io.PET.shared import listmode_to_sinogram, LazySinogram
    dev = pytomography.device
    crystals, rings, _ = SCANNERS[p['scanner']]
    info = _pet_info(crystals, rings)
    T = p.get('tof', 0)
    tof_meta = _tof_meta(T, info)
    proj_meta = PETSinogramPolygonProjMeta(info, tof_meta)
    object_meta = _pet_object(p['scanner'], p)
    out = p.get('out', str(dev))
    shape = tuple(proj_meta.shape) + ((T,) if T else ())
    sm = PETSinogramSystemMatrix(object_meta, proj_meta, attenuation_map=_uniform(object_meta.shape, dev, 0, 0.0096),
                                 sinogram_sensitivity=_uniform(proj_meta.shape, out, 0.5, 1.0),
                                 N_splits=p.get('N_splits', 1), device=out)
    if p.get('lazy'):
        ids = _random_events(info, int(p['events']), T, device='cpu')
        data = listmode_to_sinogram(ids, info, tof_meta=tof_meta, lazy=True)
        del ids
        rows = tuple(shape[1:])
        const = lambda angles, device='cpu': torch.full((len(angles), *rows), 0.1, device=device)
        additive = LazySinogram(const, shape, description='constant background', compute_on=lambda a, d: const(a, d))
    else:
        data = _poisson(shape, out)
        additive = _uniform(shape, out, 0.0, 0.2)
    algorithm = _algorithm(PoissonLogLikelihood(sm, data, additive_term=additive), p.get('algorithm', 'OSEM'), p.get('prior'))
    case = Case(algorithm, p.get('n_subsets', 8), sm)
    case.settings = {'N_splits': p.get('N_splits', 1)}
    return case


# ------------------------------------------------------------------------------------------------------- PET list mode
def build_pet_lm(p: dict) -> Case:
    import pytomography
    from pytomography.metadata.PET import PETLMProjMeta
    from pytomography.projectors.PET import PETLMSystemMatrix
    from pytomography.likelihoods import PoissonLogLikelihood
    dev = pytomography.device
    crystals, rings, _ = SCANNERS[p['scanner']]
    info = _pet_info(crystals, rings)
    T = p.get('tof', 0)
    tof_meta = _tof_meta(T, info)
    object_meta = _pet_object(p['scanner'], p)
    lor_device = p.get('lor_device', str(dev))
    ids = _random_events(info, int(p['events']), T, device=lor_device)
    extra = {}
    if p.get('weights_sensitivity'):
        n = crystals * rings
        extra['weights_sensitivity'] = _uniform((n * (n - 1) // 2,), 'cpu', 0.5, 1.0)
    proj_meta = PETLMProjMeta(ids, info, tof_meta=tof_meta, **extra)
    del ids
    sm = PETLMSystemMatrix(object_meta, proj_meta, attenuation_map=_uniform(object_meta.shape, dev, 0, 0.0096),
                           N_splits=p.get('N_splits', 1), device=dev, lor_device=lor_device,
                           sort_events=p.get('sort_events', True))
    additive = _uniform((int(p['events']),), dev, 0.0, 0.1)
    algorithm = _algorithm(PoissonLogLikelihood(sm, additive_term=additive), p.get('algorithm', 'OSEM'), p.get('prior'))
    case = Case(algorithm, p.get('n_subsets', 8), sm)
    case.settings = {'N_splits': p.get('N_splits', 1)}
    return case


# ------------------------------------------------------------------------------------------------------- SSS
def build_sss(p: dict) -> Case:
    import torch
    import pytomography
    from pytomography.metadata.PET import PETSinogramPolygonProjMeta, PETLMProjMeta
    from pytomography.projectors.PET import PETSinogramSystemMatrix, PETLMSystemMatrix
    from pytomography.io.PET.shared import listmode_to_sinogram
    from pytomography.utils import sss
    if p.get('fused') is False:
        sss._sss_cuda.available = lambda *args, **kwargs: False
    crystals, rings, _ = SCANNERS[p['scanner']]
    info = _pet_info(crystals, rings)
    T = p.get('tof', 9)
    tof_meta = _tof_meta(T, info)
    object_meta = _pet_object(p['scanner'], p)
    x = (torch.arange(object_meta.shape[0]) - object_meta.shape[0] / 2) * object_meta.dr[0]
    radius = 0.35 * object_meta.shape[0] * object_meta.dr[0]
    body = ((x[:, None] ** 2 + x[None, :] ** 2) < radius ** 2).float()[:, :, None].expand(object_meta.shape)
    attenuation = (0.0096 * body).contiguous()
    image = (body * (1 + torch.rand(object_meta.shape))).contiguous()
    if p.get('listmode'):
        ids = _random_events(info, int(p['events']), T, device='cpu')
        proj_meta = PETLMProjMeta(ids, info, tof_meta=tof_meta)
        del ids
        sm = PETLMSystemMatrix(object_meta, proj_meta, N_splits=p.get('sm_splits', 4), device=pytomography.device,
                               lor_device=pytomography.device)
        data = None
    else:
        proj_meta = PETSinogramPolygonProjMeta(info, tof_meta)
        sm = PETSinogramSystemMatrix(object_meta, proj_meta, N_splits=1, device='cpu')
        if p.get('lazy_data'):
            ids = _random_events(info, int(p['events']), T, device='cpu')
            data = listmode_to_sinogram(ids, info, tof_meta=tof_meta, lazy=True)
            del ids
        else:
            data = _poisson(tuple(proj_meta.shape) + (T,), 'cpu', mean=1.0)
    arguments = dict(proj_data=data, tof_meta=tof_meta, sinogram_interring_stepsize=p.get('step', 4),
                     sinogram_intraring_stepsize=p.get('step', 4), N_splits=p.get('N_splits', 1), lazy=p.get('lazy', True))

    class SSSCase(Case):
        def estimate(self):
            return sss.estimate_sss_memory(object_meta, proj_meta, sm, **arguments)

        def plan(self, ram_gb, gpu_gb):
            a = {k: v for k, v in arguments.items() if k not in ('N_splits', 'lazy')}
            return sss.plan_sss_memory(object_meta, proj_meta, sm, **a, ram_gb=ram_gb, gpu_gb=gpu_gb)

        def apply(self, settings):
            arguments.update({k: v for k, v in settings.items() if k in ('N_splits', 'lazy')})

        def setup(self):
            pass

        def step(self):
            self.result = sss.get_sss_scatter_estimate(object_meta, proj_meta, image, attenuation, sm, **arguments)

        @property
        def subiterations(self):
            return 1
    case = SSSCase()
    case.settings = {'N_splits': arguments['N_splits'], 'lazy': arguments['lazy']}
    return case


# ------------------------------------------------------------------------------------------------------- CT
def _helix_meta(views_per_rotation: int, n_cols: int, n_rows: int, z_extent_mm: float):
    """A clinical helical scan: focal spot 595 mm from the axis, 1085.6 mm to a cylindrical detector of ``n_cols`` x
    ``n_rows`` (0.6 mm rows at the axis, pitch 1), a 500 mm field of view, enough rotations to cover ``z_extent_mm``."""
    import numpy as np
    import torch
    from pytomography.metadata.CT import CTGen3ProjMeta
    feed = n_rows * 0.6
    rotations = int(math.ceil((z_extent_mm + feed) / feed)) + 1
    n = views_per_rotation * rotations
    phis = torch.arange(n, dtype=torch.float64) * (2 * np.pi / views_per_rotation)
    zs = feed / (2 * np.pi) * phis
    zero = torch.zeros(n)
    return CTGen3ProjMeta(phis.float(), torch.full((n,), 595.0), (zs - zs.mean()).float(), zero, zero, zero,
                          torch.full((n,), (n_cols + 1) / 2), torch.full((n,), (n_rows + 1) / 2),
                          float(2 * np.arcsin(250.0 / 595.0) / (n_cols - 1)), 0.6 * 1085.6 / 595.0, 1085.6,
                          shape=(n_cols, n_rows))


def _ct_system_matrix(p: dict):
    import numpy as np
    import torch
    import pytomography
    from pytomography.metadata import ObjectMeta
    from pytomography.metadata.CT import CTConeBeamFlatPanelProjMeta
    from pytomography.projectors.CT import CTGen3SystemMatrix, CTConeBeamFlatPanelSystemMatrix
    n, nz = p['n'], p['nz']
    out = p.get('out', str(pytomography.device))
    if p['geometry'] == 'helical':
        dz = p.get('dz', 1.0)
        meta = _helix_meta(p['views'], p['cols'], p['rows'], nz * dz)
        sm = CTGen3SystemMatrix(ObjectMeta(dr=(500.0 / n, 500.0 / n, dz), shape=(n, n, nz)), meta,
                                N_splits=p.get('N_splits', 1), device=out)
        shape = (meta.N_angles, *meta.shape)
    else:
        views = p['views']
        angles = torch.linspace(0, 2 * np.pi, views + 1)[:-1]
        du = p.get('du', 0.4)
        meta = CTConeBeamFlatPanelProjMeta(angles, torch.zeros(views), detector_radius=500.0, beam_radius=500.0,
                                           shape=(p['cols'], p['rows']), dr=(du, du))
        voxel = du / 2 * p['cols'] / n            # the detector's width at the axis, over the image
        sm = CTConeBeamFlatPanelSystemMatrix(ObjectMeta(dr=(voxel, voxel, voxel), shape=(n, n, nz)), meta,
                                             N_splits=p.get('N_splits', 1), device=out)
        shape = (meta.N_angles, *meta.shape)
    return sm, shape, out


def build_ct(p: dict) -> Case:
    from pytomography.algorithms import SART
    sm, shape, out = _ct_system_matrix(p)
    projections = _uniform(shape, out, 0.0, 4.0)
    case = Case(SART(sm, projections), p.get('n_subsets', 8), sm)
    case.settings = {'N_splits': p.get('N_splits', 1)}
    return case


class FBPCase(Case):
    def __init__(self, projections, sm, options):
        self.projections, self.sm, self.options = projections, sm, dict(options)
        self.settings = {k: v for k, v in options.items() if k == 'gpu_budget'}

    def _fbp(self):
        from pytomography.algorithms import FilteredBackProjection
        return FilteredBackProjection(self.projections, self.sm, **self.options)

    def estimate(self):
        return self._fbp().estimate_memory()

    def plan(self, ram_gb, gpu_gb):
        return self._fbp().plan_memory(ram_gb=ram_gb, gpu_gb=gpu_gb)

    def apply(self, settings):
        import pytomography
        if settings.get('gpu_budget') is not None:
            self.options['gpu_budget'] = settings['gpu_budget']
        if 'memory_budget_gb' in settings:
            pytomography.set_memory_budget(settings['memory_budget_gb'])

    def setup(self):
        pass

    def step(self):
        self.result = self._fbp()()

    @property
    def subiterations(self):
        return 1


def build_fbp(p: dict) -> Case:
    if p['geometry'] == 'spect':
        import pytomography
        from pytomography.projectors.SPECT import SPECTSystemMatrix
        object_meta, proj_meta = _spect_metas(p)
        sm = SPECTSystemMatrix([], [], object_meta, proj_meta)
        return FBPCase(_uniform(proj_meta.shape, pytomography.device, 0, 10), sm, {})
    sm, shape, out = _ct_system_matrix(p)
    options = {'gpu_budget': p['gpu_budget']} if p.get('gpu_budget') else {}
    return FBPCase(_uniform(shape, out, 0.0, 4.0), sm, options)


# ------------------------------------------------------------------------------------------------------- SPECT
def _spect_metas(p: dict):
    import numpy as np
    from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
    n, views = p['n'], p.get('views', 120)
    dx = p.get('dx', 0.48 * 128 / n)
    object_meta = SPECTObjectMeta([dx] * 3, (n, n, n))
    radii = 35.0 + 5 * np.sin(np.linspace(0, 2 * np.pi, views, endpoint=False))            # a non-circular orbit, cm
    proj_meta = SPECTProjMeta((n, n), [dx, dx], list(np.linspace(0, 360, views, endpoint=False)), radii=radii)
    return object_meta, proj_meta


class _GaussianOperator:
    """A PSF operator that exposes its kernel (as SPECTPSFToolbox operators do): a Gaussian widening with distance."""
    def _get_kernel(self, xv, yv, distances):
        import torch
        sigma = (0.03235 * distances + 0.1324).reshape(-1, 1, 1)
        k = torch.exp(-(xv ** 2 + yv ** 2) / (2 * sigma ** 2))
        return k / k.sum(dim=(1, 2), keepdim=True)

    def __call__(self, input, xv, yv, distances, normalize=False):
        import torch
        kernel = self._get_kernel(xv, yv, distances).unsqueeze(1)
        return torch.nn.functional.conv2d(input.unsqueeze(0), kernel, padding='same', groups=input.shape[0]).squeeze(0)


def _spect_system_matrix(p: dict):
    import pytomography
    from pytomography.metadata.SPECT import SPECTPSFMeta
    from pytomography.projectors.SPECT import SPECTSystemMatrix
    from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
    object_meta, proj_meta = _spect_metas(p)
    transforms = []
    if p.get('attenuation', True):
        transforms.append(SPECTAttenuationTransform(_uniform(object_meta.shape, pytomography.device, 0, 0.15),
                                                    cache_probabilities=p.get('cache_probabilities', False)))
    psf = p.get('psf', 'gaussian')
    if psf == 'gaussian':
        transforms.append(SPECTPSFTransform(SPECTPSFMeta((0.03235, 0.1324))))
    elif psf == 'operator':
        transforms.append(SPECTPSFTransform(psf_operator=_GaussianOperator(), cache_kernel=True))
    return SPECTSystemMatrix(transforms, [], object_meta, proj_meta)


def _spect_algorithm(sm, p: dict):
    import pytomography
    from pytomography.likelihoods import PoissonLogLikelihood
    shape = tuple(sm.proj_meta.shape)
    likelihood = PoissonLogLikelihood(sm, _poisson(shape, pytomography.device), additive_term=_uniform(shape, pytomography.device, 0, 1))
    return _algorithm(likelihood, p.get('algorithm', 'OSEM'), p.get('prior'))


def build_spect(p: dict) -> Case:
    sm = _spect_system_matrix(p)
    return Case(_spect_algorithm(sm, p), 1 if p.get('algorithm') == 'MLEM' else p.get('n_subsets', 8), sm)


def build_starguide(p: dict) -> Case:
    import numpy as np
    import torch
    import pytomography
    from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTPSFMeta, StarGuideProjMeta
    from pytomography.projectors.SPECT import StarGuideSystemMatrix
    from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
    dev = pytomography.device
    n, lz = p['n'], p['lz']
    object_meta = SPECTObjectMeta((0.25, 0.25, 0.25), (n, n, lz))
    n_angles, heads = p.get('angles', 30), 12
    angles = torch.tensor(np.repeat(np.linspace(0, 360, n_angles, endpoint=False), heads), dtype=torch.float32)
    views = n_angles * heads
    offsets = (torch.rand(views) * 2 - 1) * 8
    proj_meta = StarGuideProjMeta((views, 16, lz), angles.to(dev), (0.5 + torch.rand(views)).to(dev), offsets.to(dev),
                                  15.0 + 10 * torch.rand(views).numpy())
    transforms = [SPECTAttenuationTransform(_uniform((n, n, lz), dev, 0, 0.15), assume_padded=False)]
    if p.get('psf', True):
        transforms.append(SPECTPSFTransform(SPECTPSFMeta((0.02, 0.05)), assume_padded=False))
    sm = StarGuideSystemMatrix(object_meta, proj_meta, obj2obj_transforms=transforms)
    return Case(_spect_algorithm(sm, p), p.get('n_subsets', 5), sm)


def build_kem(p: dict) -> Case:
    import pytomography
    from pytomography.algorithms import KEM
    from pytomography.likelihoods import PoissonLogLikelihood
    from pytomography.projectors.shared import KEMSystemMatrix
    from pytomography.transforms.shared import KEMTransform
    sm_spect = _spect_system_matrix(p)
    sm = KEMSystemMatrix(sm_spect, KEMTransform([_uniform(sm_spect.object_meta.shape, 'cpu')], size=p.get('size', 3)))
    shape = tuple(sm_spect.proj_meta.shape)
    likelihood = PoissonLogLikelihood(sm, _poisson(shape, pytomography.device), additive_term=_uniform(shape, pytomography.device, 0, 1))
    return Case(KEM(likelihood), p.get('n_subsets', 8), sm)


def build_motion(p: dict) -> Case:
    import torch
    import pytomography
    from pytomography.projectors.SPECT import SPECTSystemMatrix
    from pytomography.projectors.shared import MotionSystemMatrix
    from pytomography.transforms.shared import DVFMotionTransform
    from pytomography.likelihoods import PoissonLogLikelihood
    from pytomography.algorithms import OSEM
    object_meta, proj_meta = _spect_metas(p)
    gates = p.get('gates', 4)
    dvf = lambda: _uniform((*object_meta.shape, 3), 'cpu', -1, 1)
    sm = MotionSystemMatrix([SPECTSystemMatrix([], [], object_meta, proj_meta) for _ in range(gates)],
                            [DVFMotionTransform(dvf(), dvf()) for _ in range(gates)])
    projections = _poisson((gates, *proj_meta.shape), pytomography.device)
    return Case(OSEM(PoissonLogLikelihood(sm, projections)), p.get('n_subsets', 8), sm)


def build_multibed(p: dict) -> Case:
    """Multi-bed SPECT: ``beds`` beds of SPECT, each with its own OSEM, stitched as ``dicom.stitch_multibed`` stitches
    them. The DICOM headers it reads (each bed's position, rows and pixel spacing) are made up."""
    import types
    import torch
    from pytomography.io.SPECT import dicom
    from pytomography.algorithms import PGAAMultiBedSPECT
    beds = p.get('beds', 3)
    n = p['n']
    object_meta, _ = _spect_metas(p)
    dz = object_meta.dr[2] * 10                                    # mm
    overlap = n // 4
    headers = {f'bed{k}': types.SimpleNamespace(
        DetectorInformationSequence=[types.SimpleNamespace(ImagePositionPatient=[0.0, 0.0, k * (n - overlap) * dz])],
        Rows=n, PixelSpacing=[dz, dz]) for k in range(beds)}
    dicom.pydicom.dcmread = lambda name, *a, **k: headers[name]
    algorithms = [_spect_algorithm(_spect_system_matrix(p), p) for _ in range(beds)]
    multibed = PGAAMultiBedSPECT(list(headers), algorithms)

    class MultiBedCase(Case):
        def estimate(self):
            return multibed.estimate_memory(n_subsets=self.n_subsets)

        def plan(self, ram_gb, gpu_gb):
            return multibed.plan_memory(ram_gb=ram_gb, gpu_gb=gpu_gb)

        def setup(self):
            for a in algorithms:
                a._set_n_subsets(self.n_subsets)

        def step(self):                     # PGAAMultiBedSPECT.__call__ for its first subset (j = 0) of one iteration
            if self.iterations:
                self.result = multibed(self.iterations, self.n_subsets)
                return
            recons = [a(1, self.n_subsets, n_subset_specific=0) for a in algorithms]
            self.result = dicom.stitch_multibed(recons=torch.stack(recons), files_NM=multibed.files_NM)
    return MultiBedCase(None, p.get('n_subsets', 8))


BUILDERS = {'pet-sino': build_pet_sino, 'pet-lm': build_pet_lm, 'sss': build_sss, 'ct': build_ct, 'fbp': build_fbp,
            'spect': build_spect, 'starguide': build_starguide, 'kem': build_kem, 'motion': build_motion,
            'multibed': build_multibed}


# =============================================================================================== a case's process
def _is_oom(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return isinstance(error, MemoryError) or any(s in text for s in ('out of memory', 'not enough memory', 'outofmemory',
                                                                    'cuda error: out of memory', 'cannot allocate',
                                                                    'failed to allocate', 'paging file'))


def run_child(spec: dict) -> dict:
    t_start = time.perf_counter()
    capped = _cap_this_process(spec['ram_cap_gb'])
    result = {'id': spec['id'], 'ram_cap_gb': spec['ram_cap_gb'], 'gpu_cap_gb': spec['gpu_cap_gb'], 'capped': capped}
    marks = {}
    import torch
    import pytomography
    from pytomography.utils import memory
    if spec.get('device') == 'cpu':
        pytomography.set_device(torch.device('cpu'))
    cuda = torch.device(pytomography.device).type == 'cuda'
    result['device'] = str(pytomography.device)
    result['cuda_available'] = torch.cuda.is_available()

    def mark(name):
        entry = {'commit_gb': _commit_bytes() / 1e9, 'peak_gb': _peak_commit_bytes() / 1e9, 't': time.perf_counter() - t_start}
        if cuda:
            entry['gpu_reserved_gb'] = torch.cuda.max_memory_reserved() / 1e9
        marks[name] = entry
    if cuda:
        total = torch.cuda.get_device_properties(0).total_memory
        cap = spec['gpu_cap_gb'] * 1e9
        if spec.get('plan'):        # a plan's GPU budget holds the CUDA context too, which is outside PyTorch's allocator
            cap -= getattr(memory, 'CUDA_CONTEXT_GPU_BYTES', 0.0)
        result['pytorch_gpu_cap_gb'] = cap / 1e9
        torch.cuda.set_per_process_memory_fraction(max(0.0, min(1.0, cap / total)))
    mark('imported')
    p = dict(spec['params'])
    plan_budget = spec.get('plan')                       # {'ram_gb', 'gpu_gb'} for a plan case
    if plan_budget:
        pytomography.set_memory_budget(plan_budget['ram_gb'])
    elif spec.get('memory_budget_gb'):
        pytomography.set_memory_budget(spec['memory_budget_gb'])
    stage = 'build'
    try:
        t0 = time.perf_counter()
        case = BUILDERS[spec['model']](p)
        case.iterations = int(p.get('iterations', 0))
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()                     # the data's making is not the reconstruction's
        result['t_build_s'] = time.perf_counter() - t0
        mark('built')
        stage = 'estimate'
        if spec.get('plans'):                            # plan every budget, run nothing
            result['plans'] = []
            for ram_gb, gpu_gb in spec['plans']:
                pytomography.set_memory_budget(ram_gb)
                plan = case.plan(ram_gb, gpu_gb)
                result['plans'].append({'ram_budget': ram_gb, 'gpu_budget': gpu_gb, 'fits': plan.fits, 'settings': plan.settings,
                                        'ram_gb': plan.estimate.ram_gb, 'gpu_gb': plan.estimate.gpu_gb, 'tried': len(plan.options)})
            pytomography.set_memory_budget(None)
            result['skipped'] = 'plans only'
            return result
        if plan_budget:
            plan = case.plan(plan_budget['ram_gb'], plan_budget['gpu_gb'])
            result['plan'] = {'fits': plan.fits, 'settings': plan.settings, 'ram_gb': plan.estimate.ram_gb,
                              'gpu_gb': plan.estimate.gpu_gb, 'tried': len(plan.options)}
            settings = spec.get('settings') or plan.settings      # the settings the driver asks to run (a plan's, or the closest)
            case.apply(settings)
            result['ran_settings'] = settings
        estimate = case.estimate()
        result['estimate'] = estimate.to_dict()
        result['subiterations_per_iteration'] = case.subiterations
        if spec.get('estimate_only'):
            result['skipped'] = 'estimate only'
            return result
        ram_room = spec['ram_cap_gb']
        if not plan_budget and (estimate.ram_gb > ram_room or estimate.gpu_gb > spec['gpu_cap_gb']):
            result['skipped'] = (f"estimate {estimate.ram_gb:.1f} GB RAM, {estimate.gpu_gb:.1f} GB GPU is above the caps "
                                 f"{ram_room:.1f} GB, {spec['gpu_cap_gb']:.1f} GB")
            return result
        stage = 'setup'
        t0 = time.perf_counter()
        case.setup()
        if cuda:
            torch.cuda.synchronize()
        result['t_setup_s'] = time.perf_counter() - t0
        mark('setup')
        stage = 'subiteration'
        times = []
        for k in range(2):
            t0 = time.perf_counter()
            case.step()
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            mark(f'step{k + 1}')
            if times[0] >= 2.0 or spec['model'] in ('sss', 'fbp') or spec.get('one_step') or case.iterations:
                break
        per_call = max(1, case.iterations * case.subiterations)     # subiterations in a call of step()
        result['t_sub_first_s'] = times[0] / per_call
        result['t_sub_s'] = min(times) / per_call
        result['t_iter_s'] = min(times) / per_call * case.subiterations
        result['ok'] = True
    except BaseException as error:                       # an allocation past a cap is a failure of that case
        result['error'] = f"{type(error).__name__}: {str(error)[:400]}"
        result['error_stage'] = stage
        result['oom'] = _is_oom(error)
    finally:
        measured = {'process_peak_gb': _peak_commit_bytes() / 1e9}
        if cuda:
            measured['gpu_reserved_gb'] = torch.cuda.max_memory_reserved() / 1e9
            measured['gpu_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9
        result['measured'] = measured
        result['marks'] = marks
        result['t_total_s'] = time.perf_counter() - t_start
    return result


# =============================================================================================== the cases
def _c(id, model, part, size, params, gpu=True, **extra):
    return dict(id=id, model=model, part=part, size=size, params=params, gpu=gpu, **extra)


def cases(full: bool) -> list[dict]:
    """The cases. Part 'estimate' compares estimates with measured peaks (sizes, then each setting swept at one size:
    the speed tables); 'plan' runs plans under caps of their budgets; 'cpu' runs on the CPU device."""
    C = []
    # ---------------------------------------------------------------- PET sinograms
    for s in (['S', 'M', 'L'] if full else ['S']):
        C.append(_c(f'pet-sino/{s}/nontof', 'pet-sino', 'estimate', s, dict(scanner=s, n_subsets=8 if s != 'L' else 14)))
    if full:
        C.append(_c('pet-sino/L/nontof/data-on-cpu', 'pet-sino', 'estimate', 'L', dict(scanner='L', n_subsets=14, out='cpu')))
    for s, T, out in ([('S', 9, 'cuda'), ('M', 9, 'cpu'), ('M', 15, 'cpu'), ('M', 21, 'cpu')] if full else [('S', 9, 'cuda')]):
        C.append(_c(f'pet-sino/{s}/tof{T}' + ('/data-on-cpu' if out == 'cpu' else ''), 'pet-sino', 'estimate', s,
                    dict(scanner=s, tof=T, n_subsets=16 if s != 'S' else 8, **({'out': 'cpu'} if out == 'cpu' else {}))))
    for s, T, ev, n in ([('M', 15, 1e7, 16), ('L', 21, 1e8, 28)] if full else [('M', 15, 1e7, 16)]):
        C.append(_c(f'pet-sino/{s}/tof{T}/lazy/{ev:.0e}', 'pet-sino', 'estimate', s,
                    dict(scanner=s, tof=T, lazy=True, events=ev, n_subsets=n, out='cuda')))
    if full:   # settings swept at M (non-TOF): subsets, N_splits, a memory budget, a prior
        for n in (1, 4, 16, 32):
            C.append(_c(f'pet-sino/M/nontof/subsets{n}', 'pet-sino', 'estimate', 'M', dict(scanner='M', n_subsets=n), speed='pet-sino M non-TOF'))
        for k in (4, 16):
            C.append(_c(f'pet-sino/M/nontof/splits{k}', 'pet-sino', 'estimate', 'M', dict(scanner='M', n_subsets=8, N_splits=k), speed='pet-sino M non-TOF'))
        C.append(_c('pet-sino/M/nontof/budget6', 'pet-sino', 'estimate', 'M', dict(scanner='M', n_subsets=8), memory_budget_gb=6, speed='pet-sino M non-TOF'))
        C.append(_c('pet-sino/M/nontof/bsrem-rdp', 'pet-sino', 'estimate', 'M', dict(scanner='M', n_subsets=8, algorithm='BSREM'), speed='pet-sino M non-TOF'))
        for n in (14, 28, 56, 112):
            C.append(_c(f'pet-sino/L/tof21/lazy/subsets{n}', 'pet-sino', 'estimate', 'L',
                        dict(scanner='L', tof=21, lazy=True, events=3e7, n_subsets=n, out='cuda'), speed='pet-sino L TOF21 lazy'))
        C.append(_c('pet-sino/L/tof21/lazy/splits4', 'pet-sino', 'estimate', 'L',
                    dict(scanner='L', tof=21, lazy=True, events=3e7, n_subsets=28, N_splits=4, out='cuda'), speed='pet-sino L TOF21 lazy'))
    # ---------------------------------------------------------------- PET list mode
    lm = [('S', 0, 1e6, 4), ('S', 9, 1e6, 4), ('M', 0, 1e7, 8), ('M', 15, 1e7, 8), ('L', 0, 1e8, 10), ('L', 21, 1e8, 10)] if full \
        else [('S', 9, 1e6, 4), ('M', 15, 1e7, 8)]
    for s, T, ev, n in lm:
        C.append(_c(f'pet-lm/{s}/{"tof" + str(T) if T else "nontof"}/{ev:.0e}', 'pet-lm', 'estimate', s,
                    dict(scanner=s, tof=T, events=ev, n_subsets=n)))
    if full:
        C.append(_c('pet-lm/L/tof21/1e+08/256cubed', 'pet-lm', 'estimate', 'L', dict(scanner='L', tof=21, events=1e8, n_subsets=10, nz=256, dz=1.0)))
        sweep = dict(scanner='M', tof=15, events=3e7)
        for n in (1, 4, 16, 32):
            C.append(_c(f'pet-lm/M/tof15/3e7/subsets{n}', 'pet-lm', 'estimate', 'M', dict(sweep, n_subsets=n), speed='pet-lm M TOF15 3e7'))
        for k in (4, 16):
            C.append(_c(f'pet-lm/M/tof15/3e7/splits{k}', 'pet-lm', 'estimate', 'M', dict(sweep, n_subsets=8, N_splits=k), speed='pet-lm M TOF15 3e7'))
        C.append(_c('pet-lm/M/tof15/3e7/subsets8', 'pet-lm', 'estimate', 'M', dict(sweep, n_subsets=8), speed='pet-lm M TOF15 3e7'))
        C.append(_c('pet-lm/M/tof15/3e7/events-on-cpu', 'pet-lm', 'estimate', 'M', dict(sweep, n_subsets=8, lor_device='cpu'), speed='pet-lm M TOF15 3e7'))
        C.append(_c('pet-lm/M/tof15/3e7/unsorted', 'pet-lm', 'estimate', 'M', dict(sweep, n_subsets=8, sort_events=False), speed='pet-lm M TOF15 3e7'))
        C.append(_c('pet-lm/M/tof15/3e7/budget8', 'pet-lm', 'estimate', 'M', dict(sweep, n_subsets=8), memory_budget_gb=8, speed='pet-lm M TOF15 3e7'))
        C.append(_c('pet-lm/M/tof15/3e7/norm-weights', 'pet-lm', 'estimate', 'M', dict(sweep, n_subsets=8, weights_sensitivity=True)))
        C.append(_c('pet-lm/M/tof15/3e7/bsrem-rdp', 'pet-lm', 'estimate', 'M', dict(sweep, n_subsets=8, algorithm='BSREM'), speed='pet-lm M TOF15 3e7'))
    # ---------------------------------------------------------------- SSS
    C.append(_c('sss/S/tof9/lazy', 'sss', 'estimate', 'S', dict(scanner='S', tof=9, lazy=True)))
    if full:
        C.append(_c('sss/S/tof9/dense', 'sss', 'estimate', 'S', dict(scanner='S', tof=9, lazy=False)))
        C.append(_c('sss/S/tof9/torch-kernel/splits1', 'sss', 'estimate', 'S', dict(scanner='S', tof=9, fused=False, N_splits=1), speed='sss S TOF9'))
        C.append(_c('sss/S/tof9/torch-kernel/splits3', 'sss', 'estimate', 'S', dict(scanner='S', tof=9, fused=False, N_splits=3), speed='sss S TOF9'))
        C.append(_c('sss/M/tof15/lazy', 'sss', 'estimate', 'M', dict(scanner='M', tof=15, lazy=True)))
        C.append(_c('sss/M/tof15/dense', 'sss', 'estimate', 'M', dict(scanner='M', tof=15, lazy=False)))
        C.append(_c('sss/M/tof15/listmode/1e7', 'sss', 'estimate', 'M', dict(scanner='M', tof=15, listmode=True, events=1e7)))
        C.append(_c('sss/L/tof21/lazy/lazy-data', 'sss', 'estimate', 'L', dict(scanner='L', tof=21, lazy=True, lazy_data=True, events=3e7)))
        C.append(_c('sss/L/tof21/lazy/lazy-data/step8', 'sss', 'estimate', 'L', dict(scanner='L', tof=21, lazy=True, lazy_data=True, events=3e7, step=8)))
    # ---------------------------------------------------------------- CT
    helical = {'S': dict(n=256, nz=64, views=576, cols=368, rows=32), 'M': dict(n=512, nz=150, views=1152, cols=736, rows=64),
               'L': dict(n=512, nz=300, views=1152, cols=736, rows=64)}
    cone = {'S': dict(n=256, nz=128, views=360, cols=512, rows=192, du=0.8), 'L': dict(n=512, nz=300, views=720, cols=1024, rows=384)}
    for s in (['S', 'M', 'L'] if full else ['S']):
        C.append(_c(f'ct-helical/{s}/sart', 'ct', 'estimate', s, dict(helical[s], geometry='helical', n_subsets=24 if s != 'L' else 12)))
        C.append(_c(f'ct-helical/{s}/fbp', 'fbp', 'estimate', s, dict(helical[s], geometry='helical')))
    for s in (['S', 'L'] if full else ['S']):
        C.append(_c(f'ct-cone/{s}/sart', 'ct', 'estimate', s, dict(cone[s], geometry='cone', n_subsets=24 if s != 'L' else 8)))
        C.append(_c(f'ct-cone/{s}/fdk', 'fbp', 'estimate', s, dict(cone[s], geometry='cone')))
    if full:
        for n in (1, 8, 48):
            C.append(_c(f'ct-helical/M/sart/subsets{n}', 'ct', 'estimate', 'M', dict(helical['M'], geometry='helical', n_subsets=n), speed='ct-helical M SART'))
        C.append(_c('ct-helical/M/sart/splits4', 'ct', 'estimate', 'M', dict(helical['M'], geometry='helical', n_subsets=24, N_splits=4), speed='ct-helical M SART'))
        C.append(_c('ct-helical/M/sart/projections-on-cpu', 'ct', 'estimate', 'M', dict(helical['M'], geometry='helical', n_subsets=24, out='cpu'), speed='ct-helical M SART'))
        for g in (0.75e9, 0.1875e9):
            C.append(_c(f'ct-helical/M/fbp/gpu-budget{g / 1e9:g}', 'fbp', 'estimate', 'M', dict(helical['M'], geometry='helical', gpu_budget=g), speed='ct-helical M FBP'))
        C.append(_c('ct-helical/M/fbp/budget1', 'fbp', 'estimate', 'M', dict(helical['M'], geometry='helical'), memory_budget_gb=1, speed='ct-helical M FBP'))
        for n in (1, 8):
            C.append(_c(f'ct-cone/S/sart/subsets{n}', 'ct', 'estimate', 'S', dict(cone['S'], geometry='cone', n_subsets=n), speed='ct-cone S SART'))
    # ---------------------------------------------------------------- SPECT, StarGuide, KEM, multi-bed, motion
    for n in ([128, 192, 256] if full else [128]):
        C.append(_c(f'spect/{n}/att-psf', 'spect', 'estimate', str(n), dict(n=n, n_subsets=8)))
    C.append(_c('spect/128/no-psf', 'spect', 'estimate', '128', dict(n=128, n_subsets=8, psf=None)))
    C.append(_c('spect/128/psf-operator', 'spect', 'estimate', '128', dict(n=128, n_subsets=8, psf='operator')))
    if full:
        C.append(_c('spect/128/cached-attenuation', 'spect', 'estimate', '128', dict(n=128, n_subsets=8, cache_probabilities=True)))
        for n in (1, 4, 15, 30):
            C.append(_c(f'spect/128/att-psf/subsets{n}', 'spect', 'estimate', '128', dict(n=128, n_subsets=n), speed='spect 128 att+PSF'))
        C.append(_c('spect/128/att-psf/subsets8', 'spect', 'estimate', '128', dict(n=128, n_subsets=8), speed='spect 128 att+PSF'))
        C.append(_c('spect/128/mlem', 'spect', 'estimate', '128', dict(n=128, algorithm='MLEM'), speed='spect 128 att+PSF'))
        C.append(_c('spect/128/bsrem-rdp', 'spect', 'estimate', '128', dict(n=128, n_subsets=8, algorithm='BSREM'), speed='spect 128 att+PSF'))
        C.append(_c('spect/128/osmaposl-rdp', 'spect', 'estimate', '128', dict(n=128, n_subsets=8, algorithm='OSMAPOSL'), speed='spect 128 att+PSF'))
        C.append(_c('spect/192/bsrem-rdp', 'spect', 'estimate', '192', dict(n=192, n_subsets=8, algorithm='BSREM')))
        C.append(_c('spect/128/fbp', 'fbp', 'estimate', '128', dict(n=128, geometry='spect')))
    for n, lz in ([(128, 100), (192, 160), (256, 200)] if full else [(128, 100)]):
        C.append(_c(f'starguide/{n}x{lz}', 'starguide', 'estimate', f'{n}x{lz}', dict(n=n, lz=lz, n_subsets=5)))
    for n in ([128, 192] if full else [128]):
        C.append(_c(f'kem/{n}', 'kem', 'estimate', str(n), dict(n=n, n_subsets=8, psf=None)))
    for beds in ([2, 3] if full else [2]):
        C.append(_c(f'multibed/{beds}x128', 'multibed', 'estimate', f'{beds}x128', dict(n=128, beds=beds, n_subsets=8)))
    if full:
        C.append(_c('motion/4x128', 'motion', 'estimate', '4x128', dict(n=128, gates=4, n_subsets=8)))
    # ---------------------------------------------------------------- two whole iterations in one call (the steady state)
    if full:
        steady = [('pet-sino/L/nontof', 'pet-sino', dict(scanner='L', n_subsets=14)),
                  ('pet-sino/L/tof21/lazy/subsets56', 'pet-sino', dict(scanner='L', tof=21, lazy=True, events=3e7, n_subsets=56, out='cuda')),
                  ('pet-lm/L/tof21/1e+08', 'pet-lm', dict(scanner='L', tof=21, events=1e8, n_subsets=10)),
                  ('pet-lm/M/tof15/3e7/subsets16', 'pet-lm', dict(scanner='M', tof=15, events=3e7, n_subsets=16)),
                  ('ct-helical/M/sart', 'ct', dict(helical['M'], geometry='helical', n_subsets=24)),
                  ('ct-cone/S/sart', 'ct', dict(cone['S'], geometry='cone', n_subsets=24)),
                  ('spect/256/att-psf', 'spect', dict(n=256, n_subsets=8)),
                  ('spect/192/bsrem-rdp', 'spect', dict(n=192, n_subsets=8, algorithm='BSREM')),
                  ('starguide/256x200', 'starguide', dict(n=256, lz=200, n_subsets=5)),
                  ('kem/192', 'kem', dict(n=192, n_subsets=8, psf=None)),
                  ('multibed/3x128', 'multibed', dict(n=128, beds=3, n_subsets=8)),
                  ('motion/4x128', 'motion', dict(n=128, gates=4, n_subsets=8))]
        for name, model, params in steady:
            C.append(_c(f'{name}/2-iterations', model, 'estimate', name.split('/')[1], dict(params, iterations=2)))
    # ---------------------------------------------------------------- plans (run under caps of their budgets)
    plans = [('pet-lm/L/tof21/3e7', 'pet-lm', dict(scanner='L', tof=21, events=3e7, n_subsets=8)),
             ('pet-sino/L/tof21/lazy/3e7', 'pet-sino', dict(scanner='L', tof=21, lazy=True, events=3e7, n_subsets=28, out='cuda'))]
    if full:
        plans += [('ct-helical/M/sart', 'ct', dict(helical['M'], geometry='helical', n_subsets=24)),
                  ('ct-helical/L/fbp', 'fbp', dict(helical['L'], geometry='helical')),
                  ('spect/192/bsrem-rdp', 'spect', dict(n=192, n_subsets=8, algorithm='BSREM')),
                  ('sss/M/tof15', 'sss', dict(scanner='M', tof=15, lazy=None))]
    for name, model, params in plans:
        C.append(_c(f'plan/{name}', model, 'plan', name, params, budgets=[(r, g) for r in PLAN_RAM_GB for g in PLAN_GPU_GB]))
    # ---------------------------------------------------------------- the CPU device (GPU budget 0)
    cpu = [('pet-sino/S-64/nontof', 'pet-sino', dict(scanner='S', n=64, nz=16, dz=4.0, n_subsets=4, out='cpu')),
           ('pet-lm/S-64/tof9/1e6', 'pet-lm', dict(scanner='S', n=64, nz=16, dz=4.0, tof=9, events=1e6, n_subsets=4, lor_device='cpu'))]
    if full:
        cpu += [('pet-sino/S-128/tof9', 'pet-sino', dict(scanner='S', n=128, nz=32, tof=9, n_subsets=4, out='cpu')),
                ('pet-lm/M/tof15/1e7', 'pet-lm', dict(scanner='M', tof=15, events=1e7, n_subsets=4, lor_device='cpu'))]
    cpu.append(('spect/64/att-psf', 'spect', dict(n=64, views=60, n_subsets=4)))
    for name, model, params in cpu:
        # PyTomography's device set to the CPU on this machine, whose GPU stays visible: the parallelproj build here is
        # the CUDA one, which refuses CPU arrays when no CUDA device is visible ("cudaSetDevice failed")
        C.append(_c(f'cpu/{name}/gpu-visible', model, 'cpu', name, params, gpu=True, device='cpu', budgets=[(r, 0) for r in (6, 8)]))
        if model == 'spect':
            # no CUDA device at all (CUDA_VISIBLE_DEVICES=-1): SPECT is PyTorch only
            C.append(_c(f'cpu/{name}/no-cuda', model, 'cpu', name, params, gpu=False, device='cpu', budgets=[(r, 0) for r in (6, 8)]))
    return C


# =============================================================================================== the driver
class GPUSampler(threading.Thread):
    """The GPU's used memory every 100 ms, from nvidia-smi, with the time each was read."""
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.process = None

    def run(self):
        try:
            self.process = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-lms', '100'],
                                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return
        for line in self.process.stdout:
            try:
                self.samples.append((time.time(), float(line.strip()) / 1024 * 1.073741824))     # MiB -> GB
            except ValueError:
                pass

    def window(self, start, end):
        before = [g for t, g in self.samples if start - 1.5 <= t < start]
        during = [g for t, g in self.samples if start <= t <= end]
        if not before or not during:
            return None
        return max(during) - min(before)

    def stop(self):
        if self.process:
            self.process.terminate()


class Lock:
    def __init__(self, path: str | None, what: str):
        self.path, self.what, self.held = path, what, False

    def acquire(self):
        if not self.path or self.held:
            return
        while True:
            try:
                os.mkdir(self.path)
                break
            except FileExistsError:
                print(f"  waiting for the GPU lock {self.path} ...", flush=True)
                time.sleep(60)
        Path(self.path, 'owner').write_text(f"memval-agent {self.what} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        self.held = True

    def release(self):
        if self.held:
            for f in Path(self.path).glob('*'):
                f.unlink()
            os.rmdir(self.path)
            self.held = False


def _launch(spec: dict, sampler: GPUSampler | None) -> dict:
    env = os.environ.copy()
    if not spec['gpu']:
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    start = time.time()
    process = subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--child', json.dumps(spec)],
                             capture_output=True, text=True, env=env)
    end = time.time()
    lines = [l for l in process.stdout.splitlines() if l.startswith('RESULT ')]
    if lines:
        result = json.loads(lines[-1][7:])
    else:
        result = {'id': spec['id'], 'error': f"process ended ({process.returncode}) without a result: "
                                             + ' | '.join((process.stderr.strip().splitlines() or ['no output'])[-3:])[:600],
                  'oom': process.returncode != 0}
    result.update({k: spec[k] for k in ('model', 'part', 'size', 'params') if k in spec})
    result['spec'] = {k: v for k, v in spec.items() if k not in ('params',)}
    result['wall_s'] = end - start
    if sampler is not None and spec['gpu']:
        delta = sampler.window(start, end)
        if delta is not None:
            result.setdefault('measured', {})['nvidia_smi_delta_gb'] = delta
    return result


def _cap_now() -> float:
    return max(0.0, min(RAM_LIMIT_GB, _machine_free_gb() - KEEP_FREE_GB))


def _summary_line(r: dict) -> str:
    if r.get('skipped'):
        return f"{r['id']}: skipped ({r['skipped']})"
    e, m = r.get('estimate'), r.get('measured', {})
    text = r['id'] + ': '
    if e:
        text += (f"RAM est {e['ram_gb']:.2f} meas {m.get('process_peak_gb', 0):.2f} ({m.get('process_peak_gb', 0) / max(e['ram_gb'], 1e-9):.2f})  "
                 f"GPU est {e['gpu_gb']:.2f} res {m.get('gpu_reserved_gb', 0):.2f}"
                 + (f" ({m.get('gpu_reserved_gb', 0) / e['gpu_gb']:.2f})" if e['gpu_gb'] else '') + '  ')
    if 't_sub_s' in r:
        text += f"sub {r['t_sub_s']:.2f}s setup {r.get('t_setup_s', 0):.1f}s"
    if 'error' in r:
        text += f"ERROR[{r.get('error_stage')}] {r['error'][:200]}"
    return text


def drive(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / 'results.jsonl'
    selected = [c for c in cases(args.full) if not args.only or any(o in c['id'] for o in args.only)]
    done = set()
    if args.resume and jsonl.exists():
        done = {json.loads(l)['run_id'] for l in jsonl.read_text().splitlines() if l.strip()}
    lock = Lock(args.lock, 'validate_memory')
    sampler = GPUSampler()
    sampler.start()
    results_lock = threading.Lock()

    def record(result):
        with results_lock:
            with open(jsonl, 'a') as f:
                f.write(json.dumps(result) + '\n')
        print('  ' + _summary_line(result), flush=True)

    def run_spec(spec, share: int = 1):
        """Runs one process, capped at the smaller of 25 GB (or its plan's RAM budget) and its share of what the
        machine has free beyond 16 GB; waits for memory if that is not enough."""
        if spec['run_id'] in done:
            return None
        budget = spec.get('ram_cap_gb')                  # a plan's RAM budget: its cap, exactly
        while (_machine_free_gb() - KEEP_FREE_GB) / share < (budget if budget is not None else 6.0):
            print(f"  waiting for memory ({_machine_free_gb():.1f} GB free) ...", flush=True)
            time.sleep(20)
        room = (_machine_free_gb() - KEEP_FREE_GB) / share
        spec['ram_cap_gb'] = float(budget) if budget is not None else min(RAM_LIMIT_GB, room)
        spec.setdefault('gpu_cap_gb', GPU_LIMIT_GB)
        if args.dry:
            spec['estimate_only'] = True
        result = _launch(spec, sampler)
        result['run_id'] = spec['run_id']
        record(result)
        return result

    def expand(case, share=1):
        """Runs a case: one process for an estimate case. For a plan or CPU case, one process that plans every budget,
        then a run of each distinct plan that fits under caps of the smallest RAM and GPU budgets that chose it (it fits
        all of them), with set_memory_budget at that RAM; and a run of the closest settings of the largest budget
        nothing fits, under its caps (it should fail)."""
        if case['part'] == 'estimate':
            run_spec(dict(case, run_id=case['id']), share)
            return
        r = run_spec(dict(case, run_id=f"{case['id']}@plans", plans=case['budgets']), share)
        if r is None:
            r = next((json.loads(l) for l in jsonl.read_text().splitlines() if l.strip() and json.loads(l)['run_id'] == f"{case['id']}@plans"), {})
        plans = r.get('plans') or []
        if args.dry:
            return
        chosen = {}
        for plan in plans:
            if plan['fits']:
                key = json.dumps(plan['settings'], sort_keys=True)
                ram, gpu = chosen.get(key, (math.inf, math.inf, None))[:2]
                chosen[key] = (min(ram, plan['ram_budget']), min(gpu, plan['gpu_budget']), plan['settings'])
        for ram, gpu, settings in chosen.values():
            run_spec(dict(case, run_id=f"{case['id']}@run{ram}/{gpu}", plan={'ram_gb': ram, 'gpu_gb': gpu}, settings=settings,
                          ram_cap_gb=float(ram), gpu_cap_gb=float(gpu) if gpu else GPU_LIMIT_GB, expect='fit'), share)
        misses = [p for p in plans if not p['fits']]
        if misses:
            p = max(misses, key=lambda p: (p['ram_budget'], p['gpu_budget']))
            if p['ram_gb'] <= RAM_LIMIT_GB and p['gpu_gb'] <= GPU_LIMIT_GB:
                run_spec(dict(case, run_id=f"{case['id']}@miss{p['ram_budget']}/{p['gpu_budget']}",
                              plan={'ram_gb': p['ram_budget'], 'gpu_gb': p['gpu_budget']}, settings=p['settings'],
                              ram_cap_gb=float(p['ram_budget']), gpu_cap_gb=float(p['gpu_budget']) if p['gpu_budget'] else GPU_LIMIT_GB,
                              expect='fail', one_step=True), share)

    gpu_cases = [c for c in selected if c['gpu']]
    cpu_cases = [c for c in selected if not c['gpu']]
    print(f"{len(selected)} cases ({len(gpu_cases)} on the GPU) -> {out}", flush=True)
    if args.list:
        for c in selected:
            print(f"  {c['part']:8s} {c['id']}")
        sampler.stop()
        return
    try:
        # CPU-only cases (CUDA hidden) first, two at a time, each capped at half of what is free beyond 16 GB
        threads = [threading.Thread(target=lambda part: [expand(c, share=2) for c in part], args=(cpu_cases[k::2],)) for k in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if gpu_cases:
            lock.acquire()
        for case in gpu_cases:
            expand(case)
    finally:
        lock.release()
        sampler.stop()
    write_report(out, args)


# =============================================================================================== the report
def _load(out: Path) -> list[dict]:
    rows = {}
    for line in (out / 'results.jsonl').read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r['run_id']] = r                       # the latest run of each
    return list(rows.values())


def _settings_text(r: dict) -> str:
    p = dict(r.get('params', {}))
    spec = r.get('spec', {})
    keys = ['iterations', 'n_subsets', 'N_splits', 'tof', 'events', 'lazy', 'out', 'lor_device', 'sort_events', 'algorithm', 'psf',
            'cache_probabilities', 'gpu_budget', 'fused', 'beds', 'gates', 'weights_sensitivity', 'lazy_data', 'listmode']
    if r.get('ran_settings'):
        p.update(r['ran_settings'])
    if r.get('estimate', {}).get('settings'):
        for k, v in r['estimate']['settings'].items():
            p.setdefault(k, v)
    parts = []
    shown_when_false = ('sort_events', 'fused', 'lazy')        # settings whose False is not their default
    for k in keys:
        if k in p and p[k] not in (None, '') and (p[k] is not False or k in shown_when_false):
            v = p[k]
            parts.append(f"{k}={v:.0e}" if k == 'events' else f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    for k in ('memory_budget_gb', 'memory_budget'):
        if spec.get(k) or p.get(k):
            parts.append(f"budget={spec.get(k) or p.get(k)} GB")
    if spec.get('device') == 'cpu':
        parts.append('device=cpu' + (' (GPU visible)' if spec.get('gpu') else ''))
    return ', '.join(parts)


def _ratio(a, b):
    return a / b if b else None


def _fmt(x, digits=2):
    return '-' if x is None else f"{x:.{digits}f}"


def write_report(out: Path, args=None) -> None:
    results = _load(out)
    (out / 'results.json').write_text(json.dumps(results, indent=1))
    est = [r for r in results if r.get('part') == 'estimate' and r.get('ok')]
    lines = ['# Memory estimates and plans, validated on synthetic data', '',
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `src/pytomography/tests/validate_memory.py`. "
             "Each run is a fresh process: random data of the right shape, the system matrix, likelihood and algorithm "
             "built, the subsets set up (every subset's H_m^T 1), then one subiteration (FBP and SSS: once, whole). "
             "Measured: the process's peak committed memory (Windows peak pagefile usage, which includes the GPU memory "
             "PyTorch holds), PyTorch's peak reserved and allocated GPU memory, and the GPU's used memory above its level "
             "before the run (nvidia-smi; per-process numbers are not reported under Windows). Ratio = measured / "
             "estimate: at most 1.00 passes.", '']
    notes = out / 'notes.md'                            # findings written by hand, kept across reruns
    if notes.exists():
        lines += [notes.read_text(encoding='utf-8').rstrip(), '']
    # summary
    worst = {}
    failures, smi_over = [], []
    for r in est:
        e, m = r['estimate'], r['measured']
        rr = _ratio(m.get('process_peak_gb'), e['ram_gb'])
        rg = _ratio(m.get('gpu_reserved_gb'), e['gpu_gb']) if e['gpu_gb'] else None
        model = r['id'].split('/')[0]
        w = worst.setdefault(model, {'ram': 0, 'gpu': 0, 'ram_id': '', 'gpu_id': '', 'min_ram': 9, 'n': 0, 'smi': 0})
        if e['gpu_gb'] and m.get('nvidia_smi_delta_gb'):
            w['smi'] = max(w['smi'], m['nvidia_smi_delta_gb'] / e['gpu_gb'])
        w['n'] += 1
        if rr is not None and rr > w['ram']:
            w['ram'], w['ram_id'] = rr, r['id']
        if rr is not None:
            w['min_ram'] = min(w['min_ram'], rr)
        if rg is not None and rg > w['gpu']:
            w['gpu'], w['gpu_id'] = rg, r['id']
        if (rr or 0) > 1 or (rg or 0) > 1:
            failures.append(r)
        if e['gpu_gb'] and (m.get('nvidia_smi_delta_gb') or 0) > e['gpu_gb']:
            smi_over.append(r)
    lines += ['## Summary of the estimates', '',
              'Ratios are measured / estimated: peak commit / `ram_gb`, PyTorch reserved / `gpu_gb`, and the GPU use nvidia-smi '
              'showed above its level before the run / `gpu_gb` (device-wide, so other processes on the GPU can add to it).', '',
              '| model | runs | worst RAM ratio | (run) | loosest RAM ratio | worst GPU ratio | (run) | worst nvidia-smi ratio |',
              '|---|---|---|---|---|---|---|---|']
    for model, w in sorted(worst.items()):
        lines.append(f"| {model} | {w['n']} | {w['ram']:.2f} | {w['ram_id']} | {w['min_ram']:.2f} | {_fmt(w['gpu'] or None)} | {w['gpu_id']} | {_fmt(w['smi'] or None)} |")
    lines.append('')
    if failures:
        lines += ['**Estimates below the measured peak:**', '']
        for r in failures:
            e, m = r['estimate'], r['measured']
            lines.append(f"- {r['id']}: RAM {m['process_peak_gb']:.2f} GB measured vs {e['ram_gb']:.2f} estimated "
                         f"({m['process_peak_gb'] - e['ram_gb']:+.2f} GB); GPU reserved {m.get('gpu_reserved_gb', 0):.2f} vs {e['gpu_gb']:.2f} "
                         f"({m.get('gpu_reserved_gb', 0) - e['gpu_gb']:+.2f} GB)")
        lines.append('')
    else:
        lines += ['No estimate was below its measured peak (commit, PyTorch reserved).', '']
    if smi_over:
        lines += [f"**GPU use by nvidia-smi above the GPU estimate** ({len(smi_over)} runs; device-wide, so a job of another "
                  "process can add to it):", '']
        for r in smi_over:
            lines.append(f"- {r['id']}: {r['measured']['nvidia_smi_delta_gb']:.2f} GB against {r['estimate']['gpu_gb']:.2f} GB "
                         f"(PyTorch reserved {r['measured'].get('gpu_reserved_gb', 0):.2f} GB)")
        lines.append('')
    elif any(r['measured'].get('nvidia_smi_delta_gb') for r in est):
        lines += ['The GPU use nvidia-smi showed was within the GPU estimate in every run.', '']
    errors = [r for r in results if r.get('part') == 'estimate' and 'error' in r]
    skipped = [r for r in results if r.get('part') == 'estimate' and r.get('skipped')]
    if errors:
        lines += ['**Runs that failed:**', ''] + [f"- {r['id']} ({r.get('error_stage')}): {r['error'][:300]}" for r in errors] + ['']
    if skipped:
        lines += ['**Skipped (estimate above the 25 GB / 10 GB caps, or the machine had too little free):**', ''] + \
                 [f"- {r['id']}: {r['skipped']}" for r in skipped] + ['']
    # per-model tables
    lines += ['## Estimates against measured peaks', '']
    by_model = {}
    for r in est:
        by_model.setdefault(r['id'].split('/')[0], []).append(r)
    for model, rows in sorted(by_model.items()):
        lines += [f"### {model}", '', '| run | size | settings | est RAM | peak commit | ratio | est GPU | reserved | ratio | allocated | nvidia-smi | ratio | setup s | subiteration s | iteration s |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for r in rows:
            e, m = r['estimate'], r['measured']
            lines.append(f"| {r['id']} | {r.get('size', '')} | {_settings_text(r)} | {e['ram_gb']:.2f} | {m['process_peak_gb']:.2f} | "
                         f"{_fmt(_ratio(m['process_peak_gb'], e['ram_gb']))} | {e['gpu_gb']:.2f} | {_fmt(m.get('gpu_reserved_gb'))} | "
                         f"{_fmt(_ratio(m.get('gpu_reserved_gb'), e['gpu_gb']) if e['gpu_gb'] else None)} | {_fmt(m.get('gpu_allocated_gb'))} | "
                         f"{_fmt(m.get('nvidia_smi_delta_gb'))} | {_fmt(_ratio(m.get('nvidia_smi_delta_gb'), e['gpu_gb']) if e['gpu_gb'] and m.get('nvidia_smi_delta_gb') else None)} | "
                         f"{_fmt(r.get('t_setup_s'), 1)} | {_fmt(r.get('t_sub_s'))} | {_fmt(r.get('t_iter_s'), 1)} |")
        lines.append('')
    # speed tables
    groups = {}
    for r in est:
        g = r.get('spec', {}).get('speed')
        if g:
            groups.setdefault(g, []).append(r)
    if groups:
        lines += ['## Speed against settings (one size per model)', '',
                  'Time of one subiteration (the first subset; a second run when the first took under 2 s) and of a full iteration '
                  '(subsets x that), with the peak each setting measured.', '']
        for g, rows in sorted(groups.items()):
            lines += [f"### {g}", '', '| settings | subiteration s | iteration s | setup s | peak commit GB | GPU reserved GB |', '|---|---|---|---|---|---|']
            for r in sorted(rows, key=lambda r: r.get('t_iter_s') or 0):
                m = r['measured']
                lines.append(f"| {_settings_text(r)} | {_fmt(r.get('t_sub_s'))} | {_fmt(r.get('t_iter_s'), 1)} | {_fmt(r.get('t_setup_s'), 1)} | "
                             f"{m['process_peak_gb']:.2f} | {_fmt(m.get('gpu_reserved_gb'))} |")
            lines.append('')
    # plans
    plan_rows = [r for r in results if r.get('part') in ('plan', 'cpu')]
    if plan_rows:
        lines += ['## Plans under caps of their budgets', '',
                  'For each budget (RAM GB / GPU GB) `plan_memory` chose the settings (with `set_memory_budget(ram_gb)`). Each '
                  'distinct plan that fits was then run under caps of the smallest RAM and GPU budgets that chose it, which it '
                  "fits in all: committed memory capped by a Windows job object, PyTorch's GPU memory by "
                  '`set_per_process_memory_fraction`. A run that goes over a cap fails. Where nothing fit, the closest '
                  'settings of the largest such budget were run under its caps too, to check that they do not fit.', '']
        for case_id in sorted({r['run_id'].split('@')[0] for r in plan_rows}):
            runs = sorted((r for r in plan_rows if r['run_id'].split('@')[0] == case_id), key=lambda r: r['run_id'])
            planned = next((r for r in runs if r['run_id'].endswith('@plans')), None)
            lines += [f"### {case_id}", '']
            if planned and planned.get('plans'):
                lines += ['| RAM / GPU budget | fits | settings chosen | est RAM | est GPU |', '|---|---|---|---|---|']
                for q in planned['plans']:
                    lines.append(f"| {q['ram_budget']} / {q['gpu_budget']} | {'yes' if q['fits'] else 'no'} | {json.dumps(q['settings'])} | "
                                 f"{q['ram_gb']:.2f} | {q['gpu_gb']:.2f} |")
                lines.append('')
            elif planned and 'error' in planned:
                lines += [f"Planning failed: {planned['error'][:200]}", '']
            lines += ['| run under caps (RAM / GPU GB) | settings | est RAM | est GPU | peak commit | GPU reserved | result | subiteration s |',
                      '|---|---|---|---|---|---|---|---|']
            for r in runs:
                kind = r['run_id'].split('@')[-1]
                if kind == 'plans':
                    continue
                budget = r.get('spec', {}).get('plan', {})
                e, m = r.get('estimate') or {}, r.get('measured', {})
                if kind.startswith('run'):
                    outcome = 'fit' if r.get('ok') else ('BROKE ITS CAP' if r.get('oom') else 'ERROR') + ': ' + r.get('error', '')[:120]
                elif not r.get('ok'):
                    outcome = 'did not fit: stopped at its cap, as planned' if r.get('oom') else 'ERROR: ' + r.get('error', '')[:120]
                else:
                    outcome = 'fit after all (planned not to)'
                settings = json.dumps(r.get('ran_settings') or r.get('spec', {}).get('settings'))
                lines.append(f"| {budget.get('ram_gb')} / {budget.get('gpu_gb')} | {settings} | {_fmt(e.get('ram_gb'))} | {_fmt(e.get('gpu_gb'))} | "
                             f"{_fmt(m.get('process_peak_gb'))} | {_fmt(m.get('gpu_reserved_gb'))} | {outcome} | {_fmt(r.get('t_sub_s'))} |")
            lines.append('')
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f"wrote {out / 'report.md'} and {out / 'results.json'}")


# =============================================================================================== main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    size = parser.add_mutually_exclusive_group()
    size.add_argument('--quick', action='store_true', help='one case per model and two plans (the default)')
    size.add_argument('--full', action='store_true', help='every model at 2-3 sizes, settings sweeps, plans')
    parser.add_argument('--only', nargs='*', help='run only cases whose id contains one of these')
    parser.add_argument('--out', default='memval', help='folder for results.jsonl, results.json and report.md')
    parser.add_argument('--lock', help='GPU lock directory to hold while GPU cases run (e.g. D:/v4runs/gpu.lock)')
    parser.add_argument('--resume', action='store_true', help='skip runs already in results.jsonl')
    parser.add_argument('--dry', action='store_true', help='build every case and print its estimate, without reconstructing')
    parser.add_argument('--list', action='store_true', help='list the cases and exit')
    parser.add_argument('--report-only', action='store_true', help='rewrite report.md and results.json from results.jsonl')
    parser.add_argument('--child', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        print('RESULT ' + json.dumps(run_child(json.loads(args.child))), flush=True)
        return
    if args.report_only:
        write_report(Path(args.out), args)
        return
    drive(args)


if __name__ == '__main__':
    main()
