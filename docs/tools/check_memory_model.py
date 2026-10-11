"""Proof that the memory model of the docs (``docs/source/_static/memory/model.json``) is PyTomography's own.

For every model, over a grid of configurations (every tutorial's, realistic scanners and large ones, on a CUDA device
and on the CPU, on Windows and on Linux), this compares

* the model, evaluated by ``memory_model.py``, with the library's ``estimate_memory()`` (peak RAM and GPU memory) and
  ``plan_memory()`` (the settings chosen, and whether they fit), and
* the browser's evaluator, ``pytomo-memory.js`` run with ``node``, with ``memory_model.py``,

and reports the largest relative difference of each model. It fails (exit 1) if an estimate differs by more than 1%
or a plan chooses other settings.

It runs on the CPU and allocates no full-size array: the library's estimates are called on its own classes with their
metadata and tiny or 'meta' tensors standing in for the data (only shapes, dtypes and devices are read). Estimates for a
CUDA device are made without a GPU as PyTorch's tests of them are: ``pytomography.device`` is set to CUDA,
``torch.cuda.mem_get_info`` gives the free memory of the configuration, and tensors that would be on the GPU report it
(their ``device`` is CUDA, or they are marked). A few 'mapping checks' first build the real objects at small size
(sinogram metadata, a lazy sinogram from events, a list mode system matrix, a KEM transform) to show that the
stand-ins give the library's numbers.

Usage (Windows, Git Bash)::

    E=/c/Users/lukep/anaconda3/envs/pytomo-pp2; export PATH="$E/Library/bin:$PATH"
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<PyTomography #294>/src $E/python.exe docs/tools/check_memory_model.py [--quick] [--model ID] [--no-js]
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import pytomography                                  # noqa: E402
from pytomography.metadata import ObjectMeta         # noqa: E402
from pytomography.utils import memory                # noqa: E402
from memory_model import MemoryModel, DEFAULT_MODEL  # noqa: E402

TOLERANCE = 0.01
JS = os.path.join(HERE, '..', 'source', '_static', 'memory', 'pytomo-memory.js')


# ------------------------------------------------------------------------------------------------ simulated devices
class _FakeCuda(torch.Tensor):
    """A 'meta' tensor (shape and dtype only) that says it is on a CUDA device."""
    @property
    def device(self):
        return torch.device('cuda')


class _FakeCpu(torch.Tensor):
    """A 'meta' tensor (shape and dtype only) that says it is on the CPU."""
    @property
    def device(self):
        return torch.device('cpu')


def fake(shape, dtype=torch.float32, on_gpu: bool = False) -> torch.Tensor:
    """A stand-in for an array of ``shape``: no memory, but its storage size, dtype and device are those of the array."""
    base = torch.empty(tuple(int(s) for s in shape), dtype=dtype, device='meta')
    return torch.Tensor._make_subclass(_FakeCuda if on_gpu else _FakeCpu, base)


def mark(tensor: torch.Tensor, on_gpu: bool) -> torch.Tensor:
    """Marks a real (CPU) tensor as one a CUDA run keeps on the GPU, for the simulated estimates."""
    tensor._sim_cuda = bool(on_gpu)
    return tensor


_ORIGINAL_NBYTES = memory.nbytes


def _nbytes(x):
    """``memory.nbytes``, with tensors marked by :func:`mark` counted on the GPU."""
    if isinstance(x, torch.Tensor) and getattr(x, '_sim_cuda', False):
        return 0.0, float(x.untyped_storage().nbytes())
    return _ORIGINAL_NBYTES(x)


def _patch_nbytes():
    for name, module in list(sys.modules.items()):
        if name.startswith('pytomography') and getattr(module, 'nbytes', None) is _ORIGINAL_NBYTES:
            module.nbytes = _nbytes


@contextlib.contextmanager
def simulate(p: dict):
    """PyTomography as a configuration ``p`` sees it: its device, operating system, free GPU memory, memory budget and
    whether CuPy's fused kernels can run."""
    from pytomography.utils import _sss_cuda, sss
    from pytomography.projectors.CT import _wfbp_cuda
    saved = (pytomography.device, pytomography.memory_budget, memory.counts_gpu_as_ram, torch.cuda.mem_get_info,
             _sss_cuda.available, _wfbp_cuda.available)
    cuda = p.get('device', 'cuda') == 'cuda'
    cupy = bool(p.get('cupy', True))
    pytomography.device = torch.device('cuda' if cuda else 'cpu')
    budget = p.get('memory_budget_gb')
    pytomography.memory_budget = None if budget is None else float(budget) * 1e9
    windows = p.get('os', 'windows') == 'windows'
    memory.counts_gpu_as_ram = lambda: windows
    free = float(p.get('gpu_free_gb', 30)) * 1e9
    torch.cuda.mem_get_info = lambda device=None: (free, 96e9)
    _sss_cuda.available = lambda device, num_tof_bins: torch.device(device).type == 'cuda' and num_tof_bins <= _sss_cuda.MAX_TOF_BINS and cupy
    _wfbp_cuda.available = lambda device: torch.device(device).type == 'cuda' and cupy
    _patch_nbytes()
    try:
        yield
    finally:
        (pytomography.device, pytomography.memory_budget, memory.counts_gpu_as_ram, torch.cuda.mem_get_info,
         _sss_cuda.available, _wfbp_cuda.available) = saved


def _algorithm(name: str, system_matrix, projections=None, additive=None, prior: bool = False):
    """A reconstruction algorithm of the library around ``system_matrix``, without running its constructor (which makes
    an initial image): what ``estimate_memory`` and ``plan_memory`` read is its likelihood (system matrix, projections,
    additive term) and its prior."""
    from pytomography import algorithms
    from pytomography.priors import RelativeDifferencePrior
    cls = getattr(algorithms, name)
    algorithm = cls.__new__(cls)
    algorithm.likelihood = SimpleNamespace(system_matrix=system_matrix, projections=projections, additive_term=additive)
    algorithm.prior = RelativeDifferencePrior(beta=1.0) if prior else None
    return algorithm


def _result(estimate) -> dict:
    return {'ram_gb': estimate.ram_gb, 'gpu_gb': estimate.gpu_gb, 'settings': dict(estimate.settings),
            'parts': [(p.name, p.scope, p.ram_bytes / 1e9, p.gpu_bytes / 1e9) for p in estimate.parts]}


def _plan_result(plan) -> dict:
    return {'settings': {k: (v.item() if hasattr(v, 'item') else v) for k, v in plan.settings.items()}, 'fits': plan.fits,
            'ram_gb': plan.estimate.ram_gb, 'gpu_gb': plan.estimate.gpu_gb, 'tried': len(plan.options)}


# ------------------------------------------------------------------------------------------------------------- PET
def _pet_shape(p) -> list:
    """Sinogram shape (angles, radial bins, planes) as PETSinogramPolygonProjMeta computes it (checked in
    :func:`mapping_checks`)."""
    C, R = int(p['crystals_per_ring']), int(p['rings'])
    return [C // 2, C + 1, R * R]


def _pet_info(C: int, R: int, crystals_trans: int = 8, crystals_axial: int = 8) -> dict:
    """A scanner geometry dictionary of ``C`` crystals per ring and ``R`` rings, in blocks of ``crystals_trans`` by
    ``crystals_axial`` crystals (the mMR's layout for C=448, R=64)."""
    return dict(min_rsector_difference=0, crystal_length=20.0, radius=328.0, firstCrystalAxis=0,
                rsectorTransNr=C // crystals_trans, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=R // crystals_axial,
                moduleTransSpacing=0.0, moduleAxialSpacing=32.25, submoduleTransNr=1, submoduleAxialNr=1,
                submoduleTransSpacing=0.0, submoduleAxialSpacing=0.0, crystalTransNr=crystals_trans,
                crystalAxialNr=crystals_axial, crystalTransSpacing=4.0, crystalAxialSpacing=4.0,
                NrCrystalsPerRing=C, NrRings=R)


def _sss_rtheta(p, shape, T) -> float:
    """Host memory of a lazy SSS estimate, from the library's own estimate of it (its interpolation over angle and
    radius of the sampled ring pairs)."""
    from pytomography.utils import sss
    step = int(p.get('sss_step', 4))
    proj_meta = SimpleNamespace(info={'NrCrystalsPerRing': int(p['crystals_per_ring']), 'NrRings': int(p['rings'])})
    sizes = sss._sss_sizes(proj_meta, SimpleNamespace(num_bins=T) if T > 1 or p.get('tof') else None, step, step)
    return 4.0 * sizes['N_theta'] * sizes['N_r'] * sizes['sampled_planes'] * sizes['T']


def build_pet_sinogram(p):
    """OSEM (or the algorithm of ``p``) of a PETSinogramSystemMatrix, its data and additive term as the tutorials make
    them: a dense sinogram or a lazy one binned from the events; randoms and a lazy SSS scatter estimate divided by the
    sensitivity."""
    from pytomography.projectors.PET import PETSinogramSystemMatrix
    from pytomography.io.PET.shared import LazySinogram
    shape = _pet_shape(p)
    tof = bool(p['tof'])
    T = int(p['tof_bins']) if tof else 1
    cuda = p['device'] == 'cuda'
    sm = PETSinogramSystemMatrix.__new__(PETSinogramSystemMatrix)
    sm.object_meta = ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(int(p['Lx']), int(p['Ly']), int(p['Lz'])))
    sm.proj_meta = SimpleNamespace(N_angles=shape[0], shape=shape, tof_meta=SimpleNamespace(num_bins=T) if tof else None)
    sm.TOF = tof
    sm.output_device = p['device'] if p['proj_device'] == 'device' else 'cpu'
    out_gpu = torch.device(sm.output_device).type == 'cuda'
    sm.sinogram_sensitivity = fake(shape, on_gpu=out_gpu) if p['sensitivity'] else None
    sm.N_splits = int(p['N_splits'])
    full = shape + [T] if tof else shape
    if p['data'] == 'lazy':
        data = LazySinogram(None, full, memory_bytes=(6 if tof else 4) * int(p['events']) + 8 * (shape[0] + 1))
    else:
        data = fake(full, on_gpu=p['data'] == 'dense_gpu' and True)
    additive = None
    if p['additive'] == 'dense':
        additive = fake(full, on_gpu=p['data'] == 'dense_gpu')
    elif p['additive'] == 'lazy':
        scatter = LazySinogram(None, full, memory_bytes=_sss_rtheta(p, shape, T))
        randoms = fake(shape)
        sensitivity = fake(shape + [1] if tof else shape)
        additive = ((randoms.unsqueeze(-1) if tof else randoms) + scatter) / sensitivity
    return _algorithm(p['algorithm'], sm, data, additive, p['prior'])


def _pet_sinogram_settings(p):
    shape = _pet_shape(p)
    return [{'N_splits': int(p['N_splits']), 'n_subsets': 1}, {'N_splits': 10, 'n_subsets': 28},
            {'N_splits': 1, 'n_subsets': shape[0]}, {'N_splits': 3, 'n_subsets': 7}]


def _listmode_system_matrix(p):
    """A PETLMSystemMatrix as the library's estimate reads it: the events' detector IDs (their number, columns, dtype
    and device), the sensitivity weights of every crystal pair, the scanner geometry being known, the output device
    and N_splits (checked against a real one in :func:`mapping_checks`)."""
    from pytomography.projectors.PET import PETLMSystemMatrix
    cuda = p['device'] == 'cuda'
    E = int(p['events'])
    cols = 3 if p['tof'] else 2
    dtype = torch.int64 if int(p['id_bytes']) == 8 else torch.int32
    crystals = int(p['crystals_per_ring']) * int(p['rings'])
    sm = PETLMSystemMatrix.__new__(PETLMSystemMatrix)
    sm.object_meta = ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(int(p['Lx']), int(p['Ly']), int(p['Lz'])))
    weights = None if p['lm_weights'] == 'none' else fake((crystals * (crystals - 1) // 2,), on_gpu=p['lm_weights'] == 'gpu')
    sm.proj_meta = SimpleNamespace(detector_ids=fake((E, cols), dtype, on_gpu=cuda and p['lor_device'] == 'device'),
                                   weights_sensitivity=weights, info={'NrCrystalsPerRing': int(p['crystals_per_ring'])},
                                   shape=(E,))
    sm.sort_events = bool(p['sort_events'])
    sm.output_device = p['device'] if p['proj_device'] == 'device' else 'cpu'
    sm.N_splits = int(p['N_splits'])
    return sm


def build_pet_listmode(p):
    """OSEM (or the algorithm of ``p``) of a PETLMSystemMatrix; the likelihood's projections are the one-element tensor
    a list mode likelihood makes, on the device, and its additive term one value per event, moved there too."""
    cuda = p['device'] == 'cuda'
    sm = _listmode_system_matrix(p)
    additive = fake((int(p['events']),), on_gpu=cuda) if p['lm_additive'] else None
    return _algorithm(p['algorithm'], sm, fake((1,), on_gpu=cuda), additive, p['prior'])


def _pet_listmode_settings(p):
    return [{'N_splits': int(p['N_splits']), 'n_subsets': 1}, {'N_splits': 10, 'n_subsets': 14},
            {'N_splits': 1, 'n_subsets': 256}, {'N_splits': 3, 'n_subsets': 21}]


# --------------------------------------------------------------------------------------------------------- generic
def _estimate_algorithm(obj, settings):
    return _result(obj.estimate_memory(**settings))


def _plan_algorithm(obj, ram_gb, gpu_gb, fixed):
    return _plan_result(obj.plan_memory(ram_gb=ram_gb, gpu_gb=gpu_gb, **(fixed or {})))


class Checker:
    """How one model is checked: the library object of a configuration, its estimate and plan, the configurations, the
    settings to estimate and the budgets to plan for."""
    def __init__(self, model_id, build, grid, settings, estimate=_estimate_algorithm, plan=_plan_algorithm,
                 budgets=((16, 8), (25, 24), (64, 32), (8, 2), (1000, 1000)), fixed=({},)):
        self.model_id, self.build, self.grid, self.settings = model_id, build, grid, settings
        self.estimate, self.plan, self.budgets, self.fixed = estimate, plan, budgets, fixed


def _sample(base: dict, axes: dict, n: int, seed: int) -> list:
    """``n`` configurations: ``base`` with values of ``axes`` drawn at random (every axis value appears at least once
    when ``n`` allows)."""
    rng = random.Random(seed)
    names = list(axes)
    out = []
    for i in range(n):
        p = dict(base)
        for name in names:
            values = axes[name]
            p[name] = values[i % len(values)] if i < max(len(v) for v in axes.values()) else rng.choice(values)
        out.append(p)
    return out


# --------------------------------------------------------------------------------------------------- the grids
DEVICES = {'device': ['cuda', 'cpu'], 'os': ['windows', 'linux']}


def grid_pet_sinogram(quick: bool) -> list:
    base = dict(device='cuda', os='windows', gpu_free_gb=30, memory_budget_gb=25, Lx=128, Ly=128, Lz=96, crystals_per_ring=448,
                rings=64, tof=True, tof_bins=21, proj_device='device', sensitivity=True, data='lazy', events=50_000_000,
                additive='lazy', sss_step=6, algorithm='OSEM', prior=False, N_splits=10, n_subsets=28)
    tutorials = [dict(base),                                                        # t_PETGATE_scat_sinoTOF
                 dict(base, tof=False, proj_device='cpu', data='dense_ram', additive='dense', N_splits=10, n_subsets=24)]  # t_PETGATE_scat_sino
    axes = dict(DEVICES, gpu_free_gb=[30, 4, 11.5], memory_budget_gb=[None, 25, 8],
                scanner=[(448, 64), (504, 64), (64, 8), (760, 80), (640, 120)], image=[(128, 128, 96), (200, 200, 127), (440, 440, 159), (64, 64, 32)],
                tof=[False, True], tof_bins=[21, 13, 33], proj_device=['device', 'cpu'], sensitivity=[True, False],
                data=['dense_ram', 'dense_gpu', 'lazy'], additive=['none', 'dense', 'lazy'], sss_step=[4, 6, 2],
                algorithm=['OSEM', 'BSREM', 'MLEM', 'OSMAPOSL', 'RBIEM'], prior=[False, True], N_splits=[1, 10, 3],
                events=[50_000_000, 2_000_000, 400_000_000])
    configs = tutorials + _sample(base, axes, 12 if quick else 60, 1)
    for p in configs:
        if 'scanner' in p:
            p['crystals_per_ring'], p['rings'] = p.pop('scanner')
        if 'image' in p:
            p['Lx'], p['Ly'], p['Lz'] = p.pop('image')
    return configs


def grid_pet_listmode(quick: bool) -> list:
    base = dict(device='cuda', os='windows', gpu_free_gb=30, memory_budget_gb=25, Lx=128, Ly=128, Lz=96, crystals_per_ring=448,
                rings=64, tof=False, events=50_000_000, id_bytes=4, lor_device='device', proj_device='device', sort_events=True,
                lm_weights='ram', lm_additive=True, algorithm='OSEM', prior=False, N_splits=10, n_subsets=14)
    tutorials = [dict(base, events=97_000_000),                                         # t_PETGATE_scat_lm
                 dict(base, tof=True, events=97_000_000),                               # t_PETGATE_scat_lmTOF
                 dict(base, crystals_per_ring=504, Lx=200, Ly=200, Lz=127, events=37_709_254, id_bytes=8, N_splits=4, n_subsets=21)]  # t_PETSIRD
    axes = dict(DEVICES, gpu_free_gb=[30, 4], memory_budget_gb=[None, 25],
                scanner=[(448, 64), (504, 64), (64, 8), (760, 80), (840, 672)], image=[(128, 128, 96), (200, 200, 127), (440, 440, 159)],
                tof=[False, True], events=[50_000_000, 2_000_000, 400_000_000, 1_500_000_000], id_bytes=[4, 8],
                lor_device=['device', 'cpu'], proj_device=['device', 'cpu'], sort_events=[True, False],
                lm_weights=['ram', 'gpu', 'none'], lm_additive=[True, False], algorithm=['OSEM', 'BSREM', 'MLEM', 'OSMAPOSL'],
                prior=[False, True], N_splits=[1, 10, 3])
    configs = tutorials + _sample(base, axes, 12 if quick else 60, 2)
    for p in configs:
        if 'scanner' in p:
            p['crystals_per_ring'], p['rings'] = p.pop('scanner')
        if 'image' in p:
            p['Lx'], p['Ly'], p['Lz'] = p.pop('image')
    return configs


CHECKERS = {
    'pet_sinogram': Checker('pet_sinogram', build_pet_sinogram, grid_pet_sinogram, _pet_sinogram_settings),
    'pet_listmode': Checker('pet_listmode', build_pet_listmode, grid_pet_listmode, _pet_listmode_settings),
}


# ------------------------------------------------------------------------------------------------- mapping checks
def mapping_checks() -> list:
    """The assumptions the stand-ins make, checked against the library's real objects at small size."""
    from pytomography.metadata.PET import PETSinogramPolygonProjMeta, PETTOFMeta
    from pytomography.io.PET.shared import listmode_to_sinogram
    lines = []
    ok = True
    # 1. the sinogram shape of a scanner, as the metadata computes it (loops over every crystal pair: small scanners)
    for C, R, ct, ca in ((64, 8, 4, 4), (448, 64, 8, 8), (128, 16, 8, 4)):
        meta = PETSinogramPolygonProjMeta(_pet_info(C, R, ct, ca))
        expected = _pet_shape({'crystals_per_ring': C, 'rings': R})
        same = list(meta.shape) == expected and meta.N_angles == expected[0]
        ok &= same
        lines.append(f"sinogram shape of {C} crystals x {R} rings: library {list(meta.shape)}, model {expected}: {'same' if same else 'DIFFERENT'}")
    # 2. what a lazy sinogram binned from events keeps: 4 bytes an event (6 with TOF) and the angle offsets
    info = _pet_info(64, 8, 4, 4)
    gen = torch.Generator().manual_seed(0)
    n = 20000
    for tof in (False, True):
        c1 = torch.randint(0, 64 * 8, (n,), generator=gen)
        c2 = (c1 + 32 + torch.randint(-8, 8, (n,), generator=gen)) % (64 * 8)
        ids = torch.stack([c1, c2] + ([torch.randint(0, 5, (n,), generator=gen)] if tof else []), dim=1).to(torch.int32)
        tof_meta = PETTOFMeta(5, 300.0, 60.0) if tof else None
        lazy = listmode_to_sinogram(ids, info, tof_meta=tof_meta, lazy=True)
        kept = int(lazy.description.split()[2].replace(',', ''))           # events inside the sinogram
        formula = (6 if tof else 4) * kept + 8 * (64 // 2 + 1)
        same = lazy.memory_bytes == formula
        ok &= same
        lines.append(f"lazy sinogram of {kept} events ({'TOF' if tof else 'non-TOF'}): library keeps {lazy.memory_bytes:.0f} bytes, model {formula}: {'same' if same else 'DIFFERENT'}")
    # 3. real sinogram and list mode system matrices (a small scanner, on the CPU) against their stand-ins
    from pytomography.metadata.PET import PETLMProjMeta
    from pytomography.projectors.PET import PETSinogramSystemMatrix, PETLMSystemMatrix
    object_meta = ObjectMeta(dr=(4.0, 4.0, 4.0), shape=(32, 32, 6))
    base = dict(device='cpu', os='windows', gpu_free_gb=30, memory_budget_gb=None, Lx=32, Ly=32, Lz=6, crystals_per_ring=64,
                rings=8, tof=True, tof_bins=5, events=n, id_bytes=4, lor_device='device', proj_device='device', sort_events=True,
                lm_weights='ram', lm_additive=False, algorithm='OSEM', prior=False, N_splits=3, n_subsets=4,
                sensitivity=True, data='dense_ram', additive='none')
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3)
    with simulate(base):
        proj_meta = PETSinogramPolygonProjMeta(info, tof_meta)
        real = PETSinogramSystemMatrix(object_meta, proj_meta, sinogram_sensitivity=torch.rand(proj_meta.shape), N_splits=3, device='cpu')
        stand_in = build_pet_sinogram(base).likelihood.system_matrix
        a, b = real.estimate_memory(4), stand_in.estimate_memory(4)
    same = a.ram_gb == b.ram_gb and a.gpu_gb == b.gpu_gb and [(p.name, p.ram_bytes, p.gpu_bytes) for p in a.parts] == [(p.name, p.ram_bytes, p.gpu_bytes) for p in b.parts]
    ok &= same
    lines.append(f"PETSinogramSystemMatrix (real, 64 x 8 crystals, TOF) and its stand-in: {a.ram_gb:.6f} / {b.ram_gb:.6f} GB RAM, {a.gpu_gb:.6f} / {b.gpu_gb:.6f} GB GPU: {'same' if same else 'DIFFERENT'}")
    saved = PETLMSystemMatrix._backward_full
    PETLMSystemMatrix._backward_full = lambda self, N_splits=20: torch.zeros(self.object_meta.shape)   # its sensitivity image: a projector call
    try:
        c1 = torch.randint(0, 64 * 8, (n,), generator=gen)
        ids = torch.stack([c1, (c1 + 32) % (64 * 8), torch.randint(0, 5, (n,), generator=gen)], dim=1).to(torch.int32)
        weights = torch.rand(64 * 8 * (64 * 8 - 1) // 2)
        with simulate(base):
            real = PETLMSystemMatrix(object_meta, PETLMProjMeta(ids, info, tof_meta=tof_meta, weights_sensitivity=weights),
                                     N_splits=3, device='cpu', lor_device='cpu')
            stand_in = _listmode_system_matrix(base)
            a, b = real.estimate_memory(4), stand_in.estimate_memory(4)
    finally:
        PETLMSystemMatrix._backward_full = saved
    same = a.ram_gb == b.ram_gb and a.gpu_gb == b.gpu_gb and [(p.name, p.ram_bytes, p.gpu_bytes) for p in a.parts] == [(p.name, p.ram_bytes, p.gpu_bytes) for p in b.parts]
    ok &= same
    lines.append(f"PETLMSystemMatrix (real, {n} events, TOF, sensitivity weights) and its stand-in: {a.ram_gb:.6f} / {b.ram_gb:.6f} GB RAM, {a.gpu_gb:.6f} / {b.gpu_gb:.6f} GB GPU: {'same' if same else 'DIFFERENT'}")
    return ok, lines


# ---------------------------------------------------------------------------------------------------------- running
def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def _settings_equal(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool):
            if not math.isclose(float(x), float(y), rel_tol=1e-12, abs_tol=0):
                return False
        elif x != y:
            return False
    return True


def check_model(model: MemoryModel, checker: Checker, quick: bool, js_cases: list) -> dict:
    """Compares the model with the library over the checker's grid; returns the summary and adds the cases for the
    JavaScript comparison to ``js_cases``."""
    worst = {'ram': 0.0, 'gpu': 0.0}
    worst_case = None
    estimates = plans = plan_mismatches = 0
    failures = []
    configs = checker.grid(quick)
    for p in configs:
        params = {k: v for k, v in p.items() if k in {v['name'] for v in model.data['variables']} | {'cupy'}}
        with simulate(p):
            obj = checker.build(p)
            for s in checker.settings(p):
                lib = checker.estimate(obj, s)
                mine = model.estimate(checker.model_id, params, s)
                js_cases.append({'kind': 'estimate', 'model': checker.model_id, 'params': params, 'settings': s})
                estimates += 1
                for key in ('ram', 'gpu'):
                    a, b = mine[f'{key}_gb'], lib[f'{key}_gb']
                    d = _rel(a, b) if b > 1e-6 else abs(a - b)
                    if d > worst[key]:
                        worst[key] = d
                        worst_case = (p, s, key, a, b)
                    if d > TOLERANCE:
                        failures.append(f"estimate {key}: model {a:.4f} GB, library {b:.4f} GB for {s} in {params}\n"
                                        f"    library parts {[(n, round(r, 4), round(g, 4)) for n, _, r, g in lib['parts']]}\n"
                                        f"    model parts   {[(q['name'], round(q['ram_gb'], 4), round(q['gpu_gb'], 4)) for q in mine['parts']]}")
                if not _settings_equal(mine['settings'], lib['settings']):
                    failures.append(f"estimate settings: model {mine['settings']}, library {lib['settings']} for {s} in {params}")
            for (ram_gb, gpu_gb), fixed in itertools.product(checker.budgets[:3] if quick else checker.budgets, checker.fixed):
                lib = checker.plan(obj, ram_gb, gpu_gb, fixed)
                mine = model.plan(checker.model_id, params, ram_gb, gpu_gb, fixed)
                js_cases.append({'kind': 'plan', 'model': checker.model_id, 'params': params, 'ram_gb': ram_gb, 'gpu_gb': gpu_gb, 'fixed': fixed})
                plans += 1
                if not _settings_equal(mine['settings'], lib['settings']) or mine['fits'] != lib['fits']:
                    plan_mismatches += 1
                    failures.append(f"plan for {ram_gb} GB RAM, {gpu_gb} GB GPU (fixed {fixed}): model {mine['settings']} "
                                    f"fits={mine['fits']} ({mine['estimate']['ram_gb']:.3f}, {mine['estimate']['gpu_gb']:.3f} GB), "
                                    f"library {lib['settings']} fits={lib['fits']} ({lib['ram_gb']:.3f}, {lib['gpu_gb']:.3f} GB) in {params}")
    return {'model': checker.model_id, 'configurations': len(configs), 'estimates': estimates, 'plans': plans,
            'max_rel_ram': worst['ram'], 'max_rel_gpu': worst['gpu'], 'plan_mismatches': plan_mismatches,
            'failures': failures, 'worst_case': worst_case}


def run_js(cases: list) -> list:
    """The JavaScript evaluator's results for ``cases``, run with node."""
    driver = r"""
const fs = require('fs');
const M = require(process.argv[2]);
M.load(JSON.parse(fs.readFileSync(process.argv[3], 'utf8')));
const cases = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
const out = cases.map(c => {
  if (c.kind === 'estimate') { const e = M.estimate(c.model, c.params, c.settings); return {ram_gb: e.ram_gb, gpu_gb: e.gpu_gb, settings: e.settings, parts: e.parts.length}; }
  const p = M.plan(c.model, c.params, c.ram_gb, c.gpu_gb, c.fixed);
  return {settings: p.settings, fits: p.fits, ram_gb: p.estimate.ram_gb, gpu_gb: p.estimate.gpu_gb, tried: p.options.length};
});
fs.writeFileSync(process.argv[5], JSON.stringify(out));
"""
    with tempfile.TemporaryDirectory() as tmp:
        paths = [os.path.join(tmp, name) for name in ('driver.js', 'cases.json', 'out.json')]
        with open(paths[0], 'w') as f:
            f.write(driver)
        with open(paths[1], 'w') as f:
            json.dump(cases, f)
        subprocess.run(['node', paths[0], os.path.abspath(JS), os.path.abspath(DEFAULT_MODEL), paths[1], paths[2]], check=True)
        with open(paths[2]) as f:
            return json.load(f)


def compare_js(model: MemoryModel, cases: list) -> dict:
    """Largest difference between the JavaScript and the Python evaluators, per model."""
    results = run_js(cases)
    summary = {}
    for c, js in zip(cases, results):
        s = summary.setdefault(c['model'], {'cases': 0, 'max_rel': 0.0, 'mismatches': []})
        s['cases'] += 1
        if c['kind'] == 'estimate':
            py = model.estimate(c['model'], c['params'], c['settings'])
            d = max(_rel(js['ram_gb'], py['ram_gb']), _rel(js['gpu_gb'], py['gpu_gb']) if py['gpu_gb'] else abs(js['gpu_gb']))
            s['max_rel'] = max(s['max_rel'], d)
            if d > 1e-9 or not _settings_equal(js['settings'], py['settings']) or js['parts'] != len(py['parts']):
                s['mismatches'].append(f"estimate {c}: js {js}, python {py['ram_gb']}, {py['gpu_gb']}, {py['settings']}")
        else:
            py = model.plan(c['model'], c['params'], c['ram_gb'], c['gpu_gb'], c['fixed'])
            same = _settings_equal(js['settings'], py['settings']) and js['fits'] == py['fits'] and js['tried'] == len(py['options'])
            if not same:
                s['mismatches'].append(f"plan {c}: js {js}, python {py['settings']} fits={py['fits']} tried={len(py['options'])}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--quick', action='store_true', help='a small grid')
    parser.add_argument('--model', action='append', help='check only this model (repeatable)')
    parser.add_argument('--no-js', action='store_true', help='skip the JavaScript comparison')
    parser.add_argument('--json', help='write the summary to this file')
    args = parser.parse_args(argv)
    model = MemoryModel.load()
    t0 = time.time()
    ok, lines = mapping_checks()
    print('Mapping checks (stand-ins against the real objects):')
    for line in lines:
        print('  ' + line)
    failed = not ok
    js_cases = []
    rows = []
    for model_id, checker in CHECKERS.items():
        if args.model and model_id not in args.model:
            continue
        t = time.time()
        r = check_model(model, checker, args.quick, js_cases)
        r['seconds'] = time.time() - t
        rows.append(r)
        bad = max(r['max_rel_ram'], r['max_rel_gpu']) > TOLERANCE or r['plan_mismatches'] or r['failures']
        failed |= bool(bad)
        print(f"{model_id:22s} {r['configurations']:4d} configurations, {r['estimates']:5d} estimates: max difference "
              f"RAM {100 * r['max_rel_ram']:.3f}%, GPU {100 * r['max_rel_gpu']:.3f}%; {r['plans']:4d} plans, "
              f"{r['plan_mismatches']} different ({r['seconds']:.0f} s)")
        for f in r['failures'][:8]:
            print('    ' + f)
        if len(r['failures']) > 8:
            print(f"    ... {len(r['failures']) - 8} more")
    js = {}
    if not args.no_js and js_cases:
        js = compare_js(model, js_cases)
        for model_id, s in js.items():
            print(f"{model_id:22s} JavaScript vs Python: {s['cases']} cases, max difference {s['max_rel']:.2e}, {len(s['mismatches'])} different")
            for m in s['mismatches'][:5]:
                print('    ' + m)
            failed |= bool(s['mismatches']) or s['max_rel'] > 1e-9
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'rows': [{k: v for k, v in r.items() if k != 'worst_case'} for r in rows], 'js': js,
                       'mapping': lines}, f, indent=1, default=str)
    print(f"{'FAILED' if failed else 'PASSED'} in {time.time() - t0:.0f} s")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
