"""Checks PyTomography's memory estimates against measured peaks: runs a few small reconstructions with each system
model and prints, for each, the estimate (``estimate_memory``) next to what was measured. It is not a test (pytest
does not collect it) and it needs the GPU: run it when the GPU is free, one case at a time in processes of their own::

    python -m pytomography.tests.calibrate_memory --list            # the cases
    python -m pytomography.tests.calibrate_memory                   # every case
    python -m pytomography.tests.calibrate_memory spect pet-sinogram-tof --json results.json
    python -m pytomography.tests.calibrate_memory --scale 2         # larger images and data (about 4x the memory)

What is measured, for each case, in a process of its own (so that its peaks are its own):

* ``gpu_allocated``: the most GPU memory PyTorch's tensors took (``torch.cuda.max_memory_allocated``);
* ``gpu_reserved``: the most GPU memory PyTorch held, its cache included (``torch.cuda.max_memory_reserved``), which
  is what other jobs on the GPU lose;
* ``process_peak``: the process's peak committed memory (Windows: peak pagefile usage, which includes the GPU memory
  PyTorch holds; elsewhere: peak resident size), the measure the estimates' ``ram_gb`` is for;
* ``simind_peak``: for the SIMIND case, the largest total resident memory of the SIMIND processes at once.

Each estimate should be above what is measured (ratio measured / estimated at most 1) and not far above it (a ratio
under about 0.5 means the estimate is wasteful). The ratios to look at are ``process_peak / ram_gb`` and
``gpu_reserved / gpu_gb``.

Safety: a case's estimate is checked against :func:`pytomography.utils.memory.available_memory` before it runs, and
against ``--gpu-limit-gb`` (default 4 GB, the GPU is shared); cases that would not fit are skipped. Each case's process
also caps PyTorch's GPU memory at that limit (``torch.cuda.set_per_process_memory_fraction``), so that a low estimate
ends in an out-of-memory error instead of filling the GPU. The SIMIND case runs only if ``simind`` is on the PATH.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

import pytomography
from pytomography.utils import memory
from pytomography.io.PET.scanner import scanner_info


# ------------------------------------------------------------------ measurement
def _process_peak_bytes() -> float:
    """This process's peak committed memory: peak pagefile usage on Windows, peak resident size elsewhere."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        if hasattr(info, 'peak_pagefile'):
            return float(info.peak_pagefile)
    except Exception:
        pass
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        class COUNTERS(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t),
                        ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t),
                        ('PeakPagefileUsage', ctypes.c_size_t)]
        counters = COUNTERS()
        counters.cb = ctypes.sizeof(COUNTERS)
        kernel32 = ctypes.WinDLL('kernel32')
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(COUNTERS), wintypes.DWORD]
        kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return float(counters.PeakPagefileUsage)
    import resource
    scale = 1 if sys.platform == 'darwin' else 1024          # ru_maxrss is in bytes on macOS, in kB on Linux
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


class _ChildMemory:
    """Polls the memory of this process's child processes named ``name`` (e.g. SIMIND) while it is open (resident, or
    committed on Windows if more), and keeps the largest total seen at once (``peak``, bytes). Needs psutil."""
    def __init__(self, name: str, interval: float = 0.05):
        self.name, self.interval, self.peak = name.lower(), interval, 0.0
        self._stop = threading.Event()

    def _poll(self):
        import psutil
        me = psutil.Process()
        while not self._stop.is_set():
            total = 0.0
            for child in me.children(recursive=True):
                try:
                    if self.name in child.name().lower():
                        info = child.memory_info()
                        total += max(info.rss, getattr(info, 'private', 0))      # resident, or committed on Windows
                except psutil.Error:
                    pass
            self.peak = max(self.peak, total)
            time.sleep(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        return False


# ------------------------------------------------------------------ cases: each returns (estimate, run)
def _spect_metas(scale: float, n_views: int = 120, n: int = 128):
    from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
    n = int(round(n * math.sqrt(scale) / 2)) * 2
    object_meta = SPECTObjectMeta([4.0] * 3, (n, n, n))
    radii = 200.0 + 50 * np.sin(np.linspace(0, 2 * np.pi, n_views, endpoint=False))       # a non-circular orbit
    proj_meta = SPECTProjMeta((n, n), [4.0, 4.0], list(np.linspace(0, 360, n_views, endpoint=False)), radii=radii)
    return object_meta, proj_meta


def _spect_system_matrix(scale: float, psf: str = 'gaussian', cache_probabilities: bool = False):
    from pytomography.metadata.SPECT import SPECTPSFMeta
    from pytomography.projectors.SPECT import SPECTSystemMatrix
    from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
    object_meta, proj_meta = _spect_metas(scale, n=96 if psf == 'operator' else 128)
    transforms = [SPECTAttenuationTransform((0.015 * torch.rand(object_meta.shape)).to(pytomography.device), cache_probabilities=cache_probabilities)]
    if psf == 'gaussian':
        transforms.append(SPECTPSFTransform(SPECTPSFMeta((0.03235, 1.324))))
    elif psf == 'operator':
        class _GaussianOperator:
            """A PSF operator that exposes its kernel (as SPECTSPSFToolbox operators do): a Gaussian widening with distance."""
            def _get_kernel(self, xv, yv, distances):
                sigma = (0.03235 * distances + 1.324).reshape(-1, 1, 1)
                k = torch.exp(-(xv ** 2 + yv ** 2) / (2 * sigma ** 2))
                return k / k.sum(dim=(1, 2), keepdim=True)

            def __call__(self, input, xv, yv, distances, normalize=False):
                kernel = self._get_kernel(xv, yv, distances).unsqueeze(1)
                return torch.nn.functional.conv2d(input.unsqueeze(0), kernel, padding='same', groups=input.shape[0]).squeeze(0)
        transforms.append(SPECTPSFTransform(psf_operator=_GaussianOperator(), cache_kernel=True))
    return SPECTSystemMatrix(transforms, [], object_meta, proj_meta)


def _osem_case(sm, n_subsets: int, n_iters: int = 1, algorithm: str = 'OSEM', prior: bool = False, additive: bool = True, shape=None):
    """An ordered-subsets reconstruction of random data of ``shape`` (default: the projections') with ``sm``."""
    from pytomography import algorithms
    from pytomography.likelihoods import PoissonLogLikelihood
    from pytomography.priors import RelativeDifferencePrior
    shape = tuple(sm.proj_meta.shape) if shape is None else tuple(shape)
    projections = torch.poisson(10 * torch.rand(shape)).to(pytomography.device)
    additive_term = torch.rand(shape).to(pytomography.device) if additive else None
    likelihood = PoissonLogLikelihood(sm, projections, additive_term=additive_term)
    kwargs = {'prior': RelativeDifferencePrior(beta=1)} if prior else {}
    algorithm = getattr(algorithms, algorithm)(likelihood, **kwargs)
    return algorithm.estimate_memory(n_subsets=n_subsets), lambda: algorithm(n_iters=n_iters, n_subsets=n_subsets)


def case_spect(scale):
    """SPECT, attenuation and Gaussian PSF, 128^3, 120 views, OSEM 8 subsets."""
    return _osem_case(_spect_system_matrix(scale), 8)


def case_spect_cached(scale):
    """SPECT with the attenuation probabilities of every view cached, OSEM 8 subsets."""
    return _osem_case(_spect_system_matrix(scale, cache_probabilities=True), 8)


def case_spect_psf_operator(scale):
    """SPECT with a PSF operator (PSF2D, kernel spectra cached), 96^3, OSEM 4 subsets."""
    return _osem_case(_spect_system_matrix(scale, psf='operator'), 4)


def case_spect_bsrem(scale):
    """SPECT, BSREM with a relative difference prior, 8 subsets (stacked H_m^T 1 and the prior's arrays)."""
    return _osem_case(_spect_system_matrix(scale), 8, algorithm='BSREM', prior=True)


def case_spect_mlem(scale):
    """SPECT, MLEM (one subset: the largest subset arrays)."""
    from pytomography.algorithms import MLEM
    from pytomography.likelihoods import PoissonLogLikelihood
    sm = _spect_system_matrix(scale)
    likelihood = PoissonLogLikelihood(sm, torch.poisson(10 * torch.rand(sm.proj_meta.shape)).to(pytomography.device))
    algorithm = MLEM(likelihood)
    return algorithm.estimate_memory(), lambda: algorithm(n_iters=1)


def case_starguide(scale):
    """StarGuide: 30 angles of 12 views (16 columns each), 128 x 128 x 100, attenuation and Gaussian PSF, OSEM 5 subsets."""
    from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTPSFMeta, StarGuideProjMeta
    from pytomography.projectors.SPECT import StarGuideSystemMatrix
    from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
    n = int(round(128 * math.sqrt(scale) / 2)) * 2
    lz = int(100 * math.sqrt(scale))
    object_meta = SPECTObjectMeta((2.5, 2.5, 2.5), (n, n, lz))
    n_angles, per_angle = 30, 12
    angles = torch.tensor(np.repeat(np.linspace(0, 360, n_angles, endpoint=False), per_angle), dtype=torch.float32)
    views = n_angles * per_angle
    offsets = (torch.rand(views) * 2 - 1) * 80
    proj_meta = StarGuideProjMeta((views, 16, lz), angles.to(pytomography.device), (0.5 + torch.rand(views)).to(pytomography.device),   # s
                                  offsets.to(pytomography.device), 150.0 + 100 * torch.rand(views).numpy())
    transforms = [SPECTAttenuationTransform((0.015 * torch.rand(n, n, lz)).to(pytomography.device), assume_padded=False),
                  SPECTPSFTransform(SPECTPSFMeta((0.02, 0.5)), assume_padded=False)]
    return _osem_case(StarGuideSystemMatrix(object_meta, proj_meta, obj2obj_transforms=transforms), 5)


def case_kem(scale):
    """KEM on SPECT (kernel on the host), 96^3, 4 subsets."""
    from pytomography.algorithms import KEM
    from pytomography.likelihoods import PoissonLogLikelihood
    from pytomography.projectors.SPECT import SPECTSystemMatrix
    from pytomography.projectors.shared import KEMSystemMatrix
    from pytomography.transforms.shared import KEMTransform
    object_meta, proj_meta = _spect_metas(scale, n=96)
    sm = KEMSystemMatrix(SPECTSystemMatrix([], [], object_meta, proj_meta), KEMTransform([torch.rand(object_meta.shape)], size=3))
    algorithm = KEM(PoissonLogLikelihood(sm, torch.poisson(10 * torch.rand(proj_meta.shape)).to(pytomography.device)))
    return algorithm.estimate_memory(n_subsets=4), lambda: algorithm(n_iters=1, n_subsets=4)


def case_motion(scale):
    """Motion correction: 4 gates of SPECT (each its own system matrix), 96^3, 4 subsets."""
    from pytomography.projectors.SPECT import SPECTSystemMatrix
    from pytomography.projectors.shared import MotionSystemMatrix
    from pytomography.transforms.shared import DVFMotionTransform
    object_meta, proj_meta = _spect_metas(scale, n=96)
    gates = 4
    dvf = lambda: torch.randn(*object_meta.shape, 3)
    sm = MotionSystemMatrix([SPECTSystemMatrix([], [], object_meta, proj_meta) for _ in range(gates)],
                            [DVFMotionTransform(dvf(), dvf()) for _ in range(gates)])
    from pytomography.algorithms import OSEM
    from pytomography.likelihoods import PoissonLogLikelihood
    projections = torch.poisson(10 * torch.rand(gates, *proj_meta.shape)).to(pytomography.device)
    algorithm = OSEM(PoissonLogLikelihood(sm, projections))
    return algorithm.estimate_memory(n_subsets=4), lambda: algorithm(n_iters=1, n_subsets=4)


def _pet_info(scale):
    """A scanner of 256 crystals per ring and 16 rings (32 at scale 4)."""
    rings = 16 if scale < 4 else 32
    return scanner_info(min_rsector_difference=0, crystal_length=20.0, radius=300.0, rsector0_angle_deg=90.0, transaxial_direction=-1, axial_direction=-1,
                rsectorTransNr=32, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=rings // 8, moduleTransSpacing=0.0,
                moduleAxialSpacing=33.0, submoduleTransNr=1, submoduleAxialNr=1, submoduleTransSpacing=0.0,
                submoduleAxialSpacing=0.0, crystalTransNr=8, crystalAxialNr=8, crystalTransSpacing=4.0,
                crystalAxialSpacing=4.0, NrCrystalsPerRing=256, NrRings=rings)


def _pet_object(scale):
    from pytomography.metadata import ObjectMeta
    n = int(round(128 * math.sqrt(scale) / 2)) * 2
    return ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(n, n, 32))


def _pet_sinogram_case(scale, tof: bool, n_subsets: int = 8):
    from pytomography.metadata.PET import PETSinogramPolygonProjMeta, PETTOFMeta
    from pytomography.projectors.PET import PETSinogramSystemMatrix
    tof_meta = PETTOFMeta(9, 500.0, 60.0, n_sigmas=3) if tof else None
    proj_meta = PETSinogramPolygonProjMeta(_pet_info(scale), tof_meta)
    object_meta = _pet_object(scale)
    attenuation = (0.0096 * torch.rand(object_meta.shape)).to(pytomography.device)
    sm = PETSinogramSystemMatrix(object_meta, proj_meta, attenuation_map=attenuation,
                                 sinogram_sensitivity=torch.rand(proj_meta.shape), N_splits=1, device=pytomography.device)
    return _osem_case(sm, n_subsets, shape=tuple(proj_meta.shape) + ((tof_meta.num_bins,) if tof else ()))


def case_pet_sinogram(scale):
    """PET sinogram, non-TOF, 256 crystals x 16 rings, 128 x 128 x 32, OSEM 8 subsets, data on the GPU."""
    return _pet_sinogram_case(scale, tof=False)


def case_pet_sinogram_tof(scale):
    """PET sinogram with 9 TOF bins, OSEM 8 subsets, data on the GPU."""
    return _pet_sinogram_case(scale, tof=True)


def _pet_listmode_case(scale, tof: bool, n_subsets: int = 4):
    from pytomography.algorithms import OSEM
    from pytomography.likelihoods import PoissonLogLikelihood
    from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta
    from pytomography.projectors.PET import PETLMSystemMatrix
    info = _pet_info(scale)
    n_crystals = info['NrCrystalsPerRing'] * info['NrRings']
    n_events = int(5e6 * scale)
    d0 = torch.randint(0, n_crystals, (n_events,))
    d1 = (d0 + n_crystals // 2 + torch.randint(-40, 40, (n_events,))) % n_crystals
    tof_meta = PETTOFMeta(9, 500.0, 60.0, n_sigmas=3) if tof else None
    ids = torch.stack([d0, d1] + ([torch.randint(0, 9, (n_events,))] if tof else []), dim=1)
    proj_meta = PETLMProjMeta(ids, info, tof_meta=tof_meta)
    sm = PETLMSystemMatrix(_pet_object(scale), proj_meta, N_splits=4)
    algorithm = OSEM(PoissonLogLikelihood(sm, additive_term=torch.rand(n_events).to(pytomography.device)))
    return algorithm.estimate_memory(n_subsets=n_subsets), lambda: algorithm(n_iters=1, n_subsets=n_subsets)


def case_pet_listmode(scale):
    """PET list mode, 5 million events, non-TOF, OSEM 4 subsets."""
    return _pet_listmode_case(scale, tof=False)


def case_pet_listmode_tof(scale):
    """PET list mode, 5 million events with 9 TOF bins, OSEM 4 subsets."""
    return _pet_listmode_case(scale, tof=True)


def _ct(which: str, scale: float):
    from pytomography.metadata import ObjectMeta
    from pytomography.projectors.CT import CTConeBeamFlatPanelSystemMatrix, CTGen3SystemMatrix
    from pytomography.tests.test_ct_fbp import _conebeam, _helix, _line_integrals
    if which == 'helical':
        meta = _helix(rotations=2)
        n = int(round(192 * math.sqrt(scale) / 2)) * 2
        return CTGen3SystemMatrix(ObjectMeta(dr=(1.0, 1.0, 1.0), shape=(n, n, 32)), meta), _line_integrals(meta)
    meta = _conebeam(n_views=240, n_cols=256)
    n = int(round(128 * math.sqrt(scale) / 2)) * 2
    return CTConeBeamFlatPanelSystemMatrix(ObjectMeta(dr=(0.5, 0.5, 0.5), shape=(n, n, 64)), meta), torch.rand(meta.N_angles, *meta.shape)


def _sart_case(which: str, scale: float, n_subsets: int = 8):
    from pytomography.algorithms import SART
    sm, projections = _ct(which, scale)
    projections = projections.to(sm.device)
    algorithm = SART(sm, projections)
    return algorithm.estimate_memory(n_subsets=n_subsets), lambda: algorithm(n_iters=1, n_subsets=n_subsets)


def case_ct_helical(scale):
    """Helical CT (the test helix, two rotations), 192 x 192 x 32, SART 8 subsets."""
    return _sart_case('helical', scale)


def case_ct_cone(scale):
    """Cone-beam CT, 240 views of 256 columns, 128 x 128 x 64, SART 8 subsets."""
    return _sart_case('cone', scale)


def _fbp_case(sm, projections, **options):
    from pytomography.algorithms import FilteredBackProjection
    fbp = FilteredBackProjection(projections, sm, **options)
    return fbp.estimate_memory(), fbp


def case_fbp_helical(scale):
    """Helical FBP (WFBP) of the test helix, 192 x 192 x 32."""
    return _fbp_case(*_ct('helical', scale))


def case_fbp_cone(scale):
    """Cone-beam FBP (FDK), 128 x 128 x 64."""
    return _fbp_case(*_ct('cone', scale))


def case_fbp_spect(scale):
    """SPECT FBP, 128^3, 120 views."""
    from pytomography.projectors.SPECT import SPECTSystemMatrix
    object_meta, proj_meta = _spect_metas(scale)
    return _fbp_case(SPECTSystemMatrix([], [], object_meta, proj_meta), torch.rand(proj_meta.shape))


def _sss_case(scale, N_splits: int, fused: bool = True):
    """Single scatter simulation of a cylinder, every second ring and crystal sampled. ``fused`` False uses PyTorch's
    TOF kernel (whose size N_splits bounds) instead of the fused CuPy kernel (which ignores N_splits)."""
    from pytomography.metadata.PET import PETSinogramPolygonProjMeta, PETTOFMeta
    from pytomography.projectors.PET import PETSinogramSystemMatrix
    from pytomography.utils import sss
    if not fused:
        sss._sss_cuda.available = lambda *args, **kwargs: False         # this process only
    tof_meta = PETTOFMeta(9, 500.0, 60.0, n_sigmas=3)
    proj_meta = PETSinogramPolygonProjMeta(_pet_info(scale), tof_meta)
    object_meta = _pet_object(scale)
    x = (torch.arange(object_meta.shape[0]) - object_meta.shape[0] / 2) * object_meta.dr[0]
    body = ((x[:, None] ** 2 + x[None, :] ** 2) < 100 ** 2).float()[:, :, None].expand(object_meta.shape)
    attenuation, image = 0.0096 * body, body * (1 + torch.rand(object_meta.shape))
    sm = PETSinogramSystemMatrix(object_meta, proj_meta, N_splits=1, device='cpu')
    data = torch.poisson(torch.rand(*proj_meta.shape, tof_meta.num_bins))
    arguments = dict(proj_data=data, tof_meta=tof_meta, sinogram_interring_stepsize=2, sinogram_intraring_stepsize=2,
                     N_splits=N_splits, lazy=True)
    estimate = sss.estimate_sss_memory(object_meta, proj_meta, sm, **arguments)
    return estimate, lambda: sss.get_sss_scatter_estimate(object_meta, proj_meta, image, attenuation, sm, **arguments)


def case_sss_tof(scale):
    """Single scatter simulation, 9 TOF bins, lazy, with the fused CuPy TOF kernel if CuPy is installed."""
    return _sss_case(scale, 1)


def case_sss_tof_torch(scale):
    """Single scatter simulation, 9 TOF bins, lazy, PyTorch's TOF kernel, N_splits 1."""
    return _sss_case(scale, 1, fused=False)


def case_sss_tof_torch_split(scale):
    """Single scatter simulation, 9 TOF bins, lazy, PyTorch's TOF kernel, N_splits 3."""
    return _sss_case(scale, 3, fused=False)


def case_simind(scale):
    """Monte Carlo hybrid OSEM with SIMIND, one iteration of 2 subsets (Tc-99m, 128^3, 60 views, 2 processes, 1e6
    photons): also measures SIMIND's own memory, which the estimate assumes (SIMIND_PROCESS_BYTES)."""
    import shutil
    if not shutil.which('simind'):
        raise RuntimeError('SIMIND is not on the PATH')
    from pytomography.projectors.SPECT import MonteCarloHybridSPECTSystemMatrix
    object_meta, proj_meta = _spect_metas(scale, n_views=60)
    sm = MonteCarloHybridSPECTSystemMatrix(object_meta, proj_meta, n_events=int(1e6), n_parallel=2, obj2obj_transforms=[],
                                           proj2proj_transforms=[], attenuation_map_140keV=0.15 * torch.rand(object_meta.shape),
                                           energy_window_params=['126,154,0'], primary_window_idx=0, isotope_names=['tc99m'],
                                           isotope_ratios=[1.0], collimator_type='SY-LEHR', crystal_thickness=0.95,
                                           cover_thickness=0.1, backscatter_thickness=5.0, energy_resolution_140keV=9.0)
    from pytomography.algorithms import OSEM
    from pytomography.likelihoods import MonteCarloHybridSPECTPoissonLogLikelihood
    projections = torch.poisson(10 * torch.rand(proj_meta.shape)).to(pytomography.device)
    algorithm = OSEM(MonteCarloHybridSPECTPoissonLogLikelihood(sm, projections))
    return algorithm.estimate_memory(n_subsets=2), lambda: algorithm(n_iters=1, n_subsets=2)


CASES = {name[5:].replace('_', '-'): f for name, f in sorted(globals().items()) if name.startswith('case_')}


# ------------------------------------------------------------------ running
def _run_case(name: str, scale: float, gpu_limit_gb: float) -> dict:
    """Runs one case in this process and returns what was estimated and measured."""
    cuda = torch.device(pytomography.device).type == 'cuda'
    if cuda:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gpu_limit_gb * 1e9 / total))
    estimate, run = CASES[name](scale)
    ram_available, gpu_available = memory.available_memory()
    result = {'case': name, 'device': str(pytomography.device), 'estimate': estimate.to_dict(), 'gpu_counts_as_ram': memory.counts_gpu_as_ram()}
    if estimate.ram_gb > ram_available or estimate.gpu_gb > min(gpu_available if cuda else math.inf, gpu_limit_gb):
        result['skipped'] = (f"needs ~{estimate.ram_gb:.1f} GB RAM, {estimate.gpu_gb:.1f} GB GPU; available "
                             f"{ram_available:.1f} GB RAM, {gpu_available:.1f} GB GPU (limit {gpu_limit_gb:g} GB)")
        return result
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with _ChildMemory('simind') as simind:
        run()
    if cuda:
        torch.cuda.synchronize()
    result['seconds'] = time.perf_counter() - start
    measured = {'process_peak_gb': _process_peak_bytes() / 1e9}
    if cuda:
        measured['gpu_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9
        measured['gpu_reserved_gb'] = torch.cuda.max_memory_reserved() / 1e9
    if name == 'simind':
        measured['simind_peak_gb'] = simind.peak / 1e9
    result['measured'] = measured
    return result


def _table(results: list[dict]) -> str:
    rows = [f"{'case':22s} {'est RAM':>8s} {'peak':>8s} {'ratio':>6s}   {'est GPU':>8s} {'alloc':>8s} {'reserved':>8s} {'ratio':>6s}   seconds"]
    for r in results:
        if 'error' in r or 'skipped' in r:
            rows.append(f"{r['case']:22s} {'error: ' + r['error'] if 'error' in r else 'skipped: ' + r['skipped']}")
            continue
        e, m = r['estimate'], r['measured']
        ratio = lambda a, b: f"{a / b:6.2f}" if b else '     -'
        rows.append(f"{r['case']:22s} {e['ram_gb']:8.2f} {m['process_peak_gb']:8.2f} {ratio(m['process_peak_gb'], e['ram_gb'])}   "
                    f"{e['gpu_gb']:8.2f} {m.get('gpu_allocated_gb', 0):8.2f} {m.get('gpu_reserved_gb', 0):8.2f} "
                    f"{ratio(m.get('gpu_reserved_gb', 0), e['gpu_gb'])}   {r['seconds']:7.1f}"
                    + (f"   SIMIND {m['simind_peak_gb']:.2f} GB" if 'simind_peak_gb' in m else ""))
    return "\n".join(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cases', nargs='*', help='cases to run (default: all; see --list)')
    parser.add_argument('--list', action='store_true', help='list the cases and exit')
    parser.add_argument('--scale', type=float, default=1.0, help='image and data size factor (memory grows about as its square)')
    parser.add_argument('--gpu-limit-gb', type=float, default=4.0, help='most GPU memory a case may take (the GPU is shared)')
    parser.add_argument('--json', help='also write the results to this JSON file')
    parser.add_argument('--run', help=argparse.SUPPRESS)                # internal: run one case in this process
    args = parser.parse_args(argv)
    if args.list:
        for name, f in CASES.items():
            print(f"{name:22s} {f.__doc__.splitlines()[0]}")
        return
    if args.run:
        try:
            result = _run_case(args.run, args.scale, args.gpu_limit_gb)
        except Exception as error:                                       # e.g. out of memory: a low estimate
            result = {'case': args.run, 'error': f"{type(error).__name__}: {error}"}
        print('RESULT ' + json.dumps(result))
        return
    names = args.cases or list(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        parser.error(f"unknown cases {unknown}; see --list")
    results = []
    for name in names:
        ram, gpu = memory.available_memory()
        print(f"{name}: running (available now ~{ram:.1f} GB RAM, {gpu:.1f} GB GPU)", flush=True)
        process = subprocess.run([sys.executable, '-m', 'pytomography.tests.calibrate_memory', '--run', name, '--scale', str(args.scale),
                                  '--gpu-limit-gb', str(args.gpu_limit_gb)], capture_output=True, text=True, env=os.environ.copy())
        lines = [line for line in process.stdout.splitlines() if line.startswith('RESULT ')]
        results.append(json.loads(lines[-1][7:]) if lines else {'case': name, 'error': (process.stderr.strip().splitlines() or ['no output'])[-1]})
    print()
    print(_table(results))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
