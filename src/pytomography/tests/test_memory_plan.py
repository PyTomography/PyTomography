"""Memory plans, the opposite question to memory estimates: given the memory there is, which settings fit. plan_memory
keeps the settings it is given, tries the others fastest first (the fewest projector splits, then the fewest subsets,
every number in order), returns the first that fits both budgets, and when nothing fits says so with the closest
settings instead of raising. Its choice agrees with estimate_memory. Estimates and plans convert to JSON. Without
budgets, a plan is for what is free on the machine, less a margin (available_memory)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.projectors import SystemMatrix
from pytomography.utils import memory
from pytomography.utils.memory import MemoryEstimate, MemoryPart, MemoryPlan, plan_settings
from pytomography.io.PET.scanner import scanner_info


@pytest.fixture(autouse=True)
def no_cuda_cache(monkeypatch):
    """No allowance for PyTorch's CUDA cache, so that the toy budgets below are plain arithmetic (it has its own test)."""
    monkeypatch.setattr(memory, "CUDA_CACHE_BYTES", 0.0)
    monkeypatch.setattr(memory, "CUDA_CACHE_FRACTION", 0.0)
    monkeypatch.setattr(memory, "CUDA_CONTEXT_GPU_BYTES", 0.0)


@pytest.fixture(params=[False, True], ids=["linux", "windows"])
def gpu_as_ram(request, monkeypatch):
    monkeypatch.setattr(memory, "counts_gpu_as_ram", lambda: request.param)
    return request.param


class _ToySystemMatrix(SystemMatrix):
    """Projections of 1 GB in subsets on the host, a normalisation image of 0.1 GB per subset, and projector chunks of
    0.4 GB on the GPU divided by N_splits; 40 views."""
    def __init__(self):
        self.object_meta = ObjectMeta(dr=(1, 1, 1), shape=(8, 8, 8))
        self.proj_meta = type('Meta', (), {'N_angles': 40})()
        self.N_splits = 4

    def _memory_parts(self, n_subsets, N_splits):
        splits = self.N_splits if N_splits is None else N_splits
        return [MemoryPart('sensitivity', ram_bytes=0.5e9),
                MemoryPart('normalisation images', ram_bytes=0.1e9 * n_subsets),
                MemoryPart('data and expected counts', ram_bytes=2 * 1e9 / n_subsets, scope='subset'),
                MemoryPart('LOR coordinates', gpu_bytes=0.4e9 / splits, scope='chunk')]

    def forward(self, object, **kwargs): ...
    def backward(self, proj, **kwargs): ...
    def get_subset_splits(self, n_subsets): ...


def _roundtrip(x) -> dict:
    """``x.to_dict()`` through JSON, checked to come back unchanged."""
    d = x.to_dict()
    back = json.loads(json.dumps(d))
    assert back == d
    return back


# ------------------------------------------------------------------ estimates as JSON, and the plan's search
def test_estimate_to_dict_round_trips_through_json(gpu_as_ram):
    sm = _ToySystemMatrix()
    estimate = sm.estimate_memory(n_subsets=4, held={'additive term': torch.zeros(1000)})
    d = _roundtrip(estimate)
    assert set(d) == {'title', 'settings', 'ram_gb', 'gpu_gb', 'gpu_counts_as_ram', 'parts', 'alternatives'}
    assert d['settings'] == {'N_splits': 4, 'n_subsets': 4} and d['gpu_counts_as_ram'] is gpu_as_ram
    assert d['ram_gb'] == pytest.approx(estimate.ram_gb, abs=1e-6) and d['gpu_gb'] == pytest.approx(0.1, abs=1e-6)
    assert {p['name'] for p in d['parts']} >= {'sensitivity', 'additive term', 'Python, PyTorch and CUDA'}
    assert all(set(p) == {'name', 'ram_gb', 'gpu_gb', 'scope'} for p in d['parts'])
    by_label = {a['label']: a for a in d['alternatives']}
    assert by_label['8 subsets']['settings'] == {'N_splits': 4, 'n_subsets': 8}
    assert by_label['N_splits 8']['settings'] == {'N_splits': 8, 'n_subsets': 4}
    # settings given to the estimate are those it is for; the title names N_splits only when it was given
    assert 'N_splits' not in estimate.title and 'N_splits 2' in sm.estimate_memory(4, N_splits=2).title


def test_a_large_budget_gets_the_fastest_settings(gpu_as_ram):
    plan = _ToySystemMatrix().plan_memory(ram_gb=1000, gpu_gb=1000)
    assert plan.fits and plan.settings == {'N_splits': 1, 'n_subsets': 1}
    assert len(plan.options) == 1 and plan.options[0]['fits'] and plan.fixed == []
    assert plan.budget_from == {'ram_gb': 'given', 'gpu_gb': 'given'}


def test_the_plan_keeps_the_settings_it_is_given(gpu_as_ram):
    sm = _ToySystemMatrix()
    plan = sm.plan_memory(ram_gb=1000, gpu_gb=1000, n_subsets=14)
    assert plan.settings == {'N_splits': 1, 'n_subsets': 14} and plan.fixed == ['n_subsets']
    plan = sm.plan_memory(ram_gb=1000, gpu_gb=0.15, n_subsets=14)            # 0.4 / 4 GB fits, 0.4 / 2 does not
    assert plan.settings == {'N_splits': 4, 'n_subsets': 14}
    assert all(o['settings']['n_subsets'] == 14 for o in plan.options)
    assert sm.plan_memory(ram_gb=1000, gpu_gb=1000, N_splits=16).settings == {'N_splits': 16, 'n_subsets': 1}
    assert sm.plan_memory(ram_gb=1000, gpu_gb=1000, N_splits=None).settings['N_splits'] == 1      # None: not kept
    with pytest.raises(ValueError, match="has no setting n_iters"):
        sm.plan_memory(ram_gb=10, gpu_gb=10, n_iters=3)


def test_the_plan_tries_fewer_splits_first_then_the_fewest_subsets(gpu_as_ram):
    sm = _ToySystemMatrix()
    # the GPU budget needs N_splits of at least 4; the RAM budget then needs the fewest subsets n with
    # 0.5 + 0.1 n + 2 / n GB of arrays (and the overheads) under it, every number tried in order
    ram_gb = sm.estimate_memory(n_subsets=5, N_splits=4).ram_gb
    plan = sm.plan_memory(ram_gb=ram_gb, gpu_gb=0.15)
    assert plan.fits and plan.settings['N_splits'] == 4
    tried = [o['settings'] for o in plan.options]
    assert tried[:40] == [{'N_splits': 1, 'n_subsets': n} for n in range(1, 41)]        # nothing fits with 1 split
    assert tried[-1] == plan.settings
    n = plan.settings['n_subsets']
    assert n <= 5 and all(not sm.estimate_memory(k, 4).ram_gb <= ram_gb for k in range(1, n))


def test_the_plan_agrees_with_estimate_memory(gpu_as_ram):
    sm = _ToySystemMatrix()
    held = {'additive term': torch.zeros(250_000_000 // 4)}
    for n_subsets, N_splits, gpu_gb in ((2, 2, 0.2), (3, 8, 0.06), (1, 1, 1.0)):
        ram_gb = sm.estimate_memory(n_subsets, N_splits, held).ram_gb + 1e-6       # those settings fit, so the plan does
        plan = sm.plan_memory(ram_gb=ram_gb, gpu_gb=gpu_gb, held=held)
        assert plan.fits
        estimate = sm.estimate_memory(held=held, **plan.settings)
        assert estimate.ram_gb <= ram_gb and estimate.gpu_gb <= gpu_gb
        assert plan.estimate.ram_gb == pytest.approx(estimate.ram_gb) and plan.estimate.gpu_gb == pytest.approx(estimate.gpu_gb)
        assert plan.estimate.settings == plan.settings and plan.estimate.alternatives
        assert all(not o['fits'] for o in plan.options[:-1]) and plan.options[-1]['fits']


def test_nothing_fits_returns_the_closest_settings_without_raising(gpu_as_ram):
    sm = _ToySystemMatrix()
    plan = sm.plan_memory(ram_gb=0.5, gpu_gb=0.0)
    assert not plan.fits
    assert len(plan.options) == len(memory.split_candidates(4)) * 40                  # every combination tried
    excess = lambda o: max(0, o['ram_gb'] - 0.5) + max(0, o['gpu_gb'] - 0.0)
    assert excess({'ram_gb': plan.estimate.ram_gb, 'gpu_gb': plan.estimate.gpu_gb}) == pytest.approx(min(excess(o) for o in plan.options))
    text = str(plan)
    assert text.startswith('Memory plan for _ToySystemMatrix') and 'Nothing fits; closest:' in text and '<- chosen' in text
    assert len(text.splitlines()) < 20                                                  # a short table, not 360 rows
    text.encode('cp1252')
    d = _roundtrip(plan)
    assert d['fits'] is False and len(d['options']) == len(plan.options) and d['settings'] == plan.settings
    assert d['budget'] == {'ram_gb': 0.5, 'gpu_gb': 0.0, 'from': {'ram_gb': 'given', 'gpu_gb': 'given'}}
    assert d['estimate']['settings'] == plan.settings


def test_fewest_subsets_is_a_plan_with_the_splits_kept():
    sm = _ToySystemMatrix()
    target = sm.estimate_memory(n_subsets=3).ram_gb                    # 0.5 + 0.1 n + 2 / n GB: falls until 4 subsets
    assert sm.fewest_subsets(target) == 3
    plan = sm.plan_memory(ram_gb=target, gpu_gb=math.inf, N_splits=4)
    assert plan.settings == {'N_splits': 4, 'n_subsets': 3}
    assert sm.fewest_subsets(sm.estimate_memory(n_subsets=3, N_splits=1).ram_gb, N_splits=1) == 3
    assert sm.fewest_subsets(0.1) is None
    assert _roundtrip(plan)['budget']['gpu_gb'] is None                                # no limit: None in JSON


def test_plan_settings_with_nothing_to_choose():
    estimate = lambda settings, alternatives: MemoryEstimate('fixed', [MemoryPart('a', ram_bytes=1e9)], settings=settings)
    plan = plan_settings('fixed', {}, estimate, ram_gb=2, gpu_gb=0)
    assert plan.fits and plan.settings == {} and len(plan.options) == 1
    assert 'Fits: its present settings' in str(plan)
    assert not plan_settings('fixed', {}, estimate, ram_gb=0.5, gpu_gb=0).fits


def test_available_memory_keeps_a_margin(monkeypatch):
    monkeypatch.setattr(memory, "_gpu_memory", lambda device: (20e9, 32e9))
    monkeypatch.setattr(memory, "_host_memory", lambda: (64e9, 40e9, 4e9))     # 64 GB machine, 40 GB free, this process 4 GB
    ram, gpu = memory.available_memory()
    assert ram == pytest.approx(40 - 16 + memory.fixed_overhead_bytes() / 1e9)  # 16 GB kept free; the overhead is ours
    assert gpu == pytest.approx(20 - 0.05 * 32)
    monkeypatch.setattr(memory, "_host_memory", lambda: (16e9, 10e9, 1e9))     # a 16 GB laptop: a quarter kept free
    assert memory.available_memory()[0] == pytest.approx(10 - 4 + 1)
    monkeypatch.setattr(memory, "_host_memory", lambda: (16e9, 2e9, 1e9))
    assert memory.available_memory()[0] == 0
    # plans without budgets are for what is available
    monkeypatch.setattr(memory, "_host_memory", lambda: (64e9, 40e9, 4e9))
    plan = _ToySystemMatrix().plan_memory()
    assert plan.ram_gb == pytest.approx(ram) and plan.gpu_gb == pytest.approx(gpu)
    assert plan.budget_from == {'ram_gb': 'available', 'gpu_gb': 'available'} and 'available now' in str(plan)
    plan = _ToySystemMatrix().plan_memory(ram_gb=7)
    assert plan.ram_gb == 7 and plan.gpu_gb == pytest.approx(gpu) and plan.budget_from == {'ram_gb': 'given', 'gpu_gb': 'available'}


def test_estimates_add_the_overheads_of_the_device(monkeypatch, gpu_as_ram):
    """Python, PyTorch and CUDA take more with a CUDA device (its libraries are loaded), and GPU arrays bring
    PyTorch's CUDA cache with them: a fixed amount and a fraction of the arrays."""
    monkeypatch.setattr(memory, "CUDA_CACHE_BYTES", 0.1e9)
    monkeypatch.setattr(memory, "CUDA_CACHE_FRACTION", 0.15)
    assert memory.fixed_overhead_bytes('cuda') == memory.FIXED_OVERHEAD_BYTES > memory.fixed_overhead_bytes('cpu') == memory.FIXED_OVERHEAD_CPU_BYTES
    for device in ('cpu', 'cuda'):
        monkeypatch.setattr(pytomography, "device", torch.device(device))
        estimate = memory.memory_estimate('test', [MemoryPart('image', gpu_bytes=2e9), MemoryPart('data', ram_bytes=1e9)])
        parts = {p.name: p for p in estimate.parts}
        assert parts['Python, PyTorch and CUDA'].ram_bytes == memory.fixed_overhead_bytes(device)
        assert parts["PyTorch's GPU cache (rounded and kept blocks)"].gpu_bytes == pytest.approx(0.1e9 + 0.15 * 2e9)
        assert estimate.gpu_gb == pytest.approx(2 + 0.1 + 0.3)
    host_only = memory.memory_estimate('test', [MemoryPart('data', ram_bytes=1e9)])
    assert all('GPU cache' not in p.name for p in host_only.parts) and host_only.gpu_gb == 0


def test_gpu_estimates_count_the_cuda_context(monkeypatch, gpu_as_ram):
    """On a CUDA device, the CUDA context and its libraries hold GPU memory outside PyTorch's allocator, which nvidia-smi
    shows and torch.cuda.memory_reserved does not (0.57 GB measured): GPU estimates count it. On Windows, where it is
    committed memory too and already in the fixed overhead, ram_gb stays the same. Found by validate_memory.py: the GPU's
    use ran above estimates without it in 37 of 121 reconstructions, by up to 0.54 GB."""
    monkeypatch.setattr(memory, "CUDA_CONTEXT_GPU_BYTES", 0.6e9)
    arrays = [MemoryPart('image', gpu_bytes=2e9), MemoryPart('data', ram_bytes=1e9)]
    monkeypatch.setattr(pytomography, "device", torch.device('cuda'))
    with_context = memory.memory_estimate('test', arrays)
    monkeypatch.setattr(memory, "CUDA_CONTEXT_GPU_BYTES", 0.0)
    without = memory.memory_estimate('test', arrays)
    assert with_context.gpu_gb == pytest.approx(without.gpu_gb + 0.6)
    assert with_context.ram_gb == pytest.approx(without.ram_gb)           # Windows: moved from RAM to GPU; elsewhere: GPU only
    context = {p.name: p for p in with_context.parts}['CUDA context and libraries on the GPU (outside PyTorch)']
    assert context.gpu_bytes == 0.6e9 and context.scope == 'overhead'
    fixed = {p.name: p for p in with_context.parts}['Python, PyTorch and CUDA']
    assert fixed.ram_bytes == pytest.approx(memory.FIXED_OVERHEAD_BYTES - (0.6e9 if gpu_as_ram else 0.0))
    # not on the CPU device
    monkeypatch.setattr(memory, "CUDA_CONTEXT_GPU_BYTES", 0.6e9)
    monkeypatch.setattr(pytomography, "device", torch.device('cpu'))
    assert memory.cuda_context() is None
    assert all('CUDA context' not in p.name for p in memory.memory_estimate('test', arrays).parts)


def test_available_memory_reads_this_machine():
    total, available, process = memory._host_memory()
    assert total > 0 and 0 < available <= total and process >= 0   # 0: unknown here, so nothing is added back
    ram, gpu = memory.available_memory()
    assert 0 <= ram <= total / 1e9 and gpu >= 0
    if not torch.cuda.is_available():
        assert gpu == 0 and memory.available_memory('cpu')[1] == 0


# ------------------------------------------------------------------ PET and CT system matrices
INFO = scanner_info(min_rsector_difference=0, crystal_length=20.0, radius=120.0, rsector0_angle_deg=90.0, transaxial_direction=-1, axial_direction=-1,
            rsectorTransNr=16, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=2, moduleTransSpacing=0.0, moduleAxialSpacing=17.0,
            submoduleTransNr=1, submoduleAxialNr=1, submoduleTransSpacing=0.0, submoduleAxialSpacing=0.0,
            crystalTransNr=4, crystalAxialNr=4, crystalTransSpacing=4.0, crystalAxialSpacing=4.0,
            NrCrystalsPerRing=64, NrRings=8)
PET_OBJECT = ObjectMeta(dr=(4, 4, 4), shape=(32, 32, 6))


def _pet_sinogram_system_matrix(tof=True, N_splits=3):
    pytest.importorskip("parallelproj_core")
    from pytomography.metadata.PET import PETSinogramPolygonProjMeta, PETTOFMeta
    from pytomography.projectors.PET import PETSinogramSystemMatrix
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3) if tof else None
    return PETSinogramSystemMatrix(PET_OBJECT, PETSinogramPolygonProjMeta(INFO, tof_meta), N_splits=N_splits, device='cpu')


def _pet_listmode_system_matrix(monkeypatch, n_events=5000):
    """A list mode system matrix, without computing its sensitivity image (a projector call)."""
    pytest.importorskip("parallelproj_core")
    from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta
    from pytomography.projectors.PET import PETLMSystemMatrix
    monkeypatch.setattr(PETLMSystemMatrix, "_backward_full", lambda self, N_splits=20: torch.zeros(self.object_meta.shape))
    gen = torch.Generator().manual_seed(0)
    n = INFO['NrCrystalsPerRing'] * INFO['NrRings']
    d0 = torch.randint(0, n, (n_events,), generator=gen)
    ids = torch.stack([d0, (d0 + n // 2) % n, torch.randint(0, 5, (n_events,), generator=gen)], dim=1)
    proj_meta = PETLMProjMeta(ids, INFO, tof_meta=PETTOFMeta(5, 300.0, 60.0, n_sigmas=3))
    return PETLMSystemMatrix(PET_OBJECT, proj_meta, N_splits=2, device='cpu', lor_device='cpu')


def test_pet_sinogram_plan(gpu_as_ram):
    sm = _pet_sinogram_system_matrix()
    settings = sm._memory_settings()
    assert list(settings) == ['N_splits', 'n_subsets'] and settings['N_splits'][0] == 3
    assert settings['n_subsets'][1] == list(range(1, sm.proj_meta.N_angles + 1))      # one angle per subset at most
    data = torch.zeros(*sm.proj_meta.shape, 5)
    held = {'data': data, 'additive term': data}
    plan = sm.plan_memory(ram_gb=1000, gpu_gb=1000, held=held)
    assert plan.settings == {'N_splits': 1, 'n_subsets': 1}
    # a RAM budget between the estimates of 1 and 4 subsets: the plan needs subsets, and agrees with estimate_memory
    ram_gb = (sm.estimate_memory(1, 1, held).ram_gb + sm.estimate_memory(4, 1, held).ram_gb) / 2
    plan = sm.plan_memory(ram_gb=ram_gb, gpu_gb=1000, held=held)
    assert plan.fits and 1 < plan.settings['n_subsets'] <= 4 and plan.settings['N_splits'] == 1
    assert sm.estimate_memory(held=held, **plan.settings).ram_gb <= ram_gb
    assert _roundtrip(plan)['estimate']['settings'] == plan.settings


def test_pet_listmode_plan(monkeypatch, gpu_as_ram):
    sm = _pet_listmode_system_matrix(monkeypatch)
    estimate = sm.estimate_memory(4)
    assert estimate.settings == {'N_splits': 2, 'n_subsets': 4}
    assert sm._memory_settings()['n_subsets'][1][-1] == memory.PLAN_MAX_SUBSETS        # events: no angles to limit them
    plan = sm.plan_memory(ram_gb=1000, gpu_gb=1000, n_subsets=7)
    assert plan.settings == {'N_splits': 1, 'n_subsets': 7}
    _roundtrip(plan)


def _on(monkeypatch, device: str):
    """Estimates for a CUDA or a CPU device, whatever this machine has: the device, and for CUDA the free GPU memory
    the GPU budget of a chunked step reads (30 GB)."""
    monkeypatch.setattr(pytomography, "device", torch.device(device))
    if device == 'cuda':
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (30e9, 32e9))


@pytest.mark.parametrize("model", ["pet sinogram", "pet list mode"])
def test_pet_estimates_on_a_cpu_device_are_in_ram_and_plans_fit_without_a_gpu(model, monkeypatch):
    """With PyTomography on the CPU, the images and the projector's chunk of a PET reconstruction are in RAM: the
    estimate has no GPU memory (and no allowance for PyTorch's CUDA cache), and a plan fits with a GPU budget of 0. The
    parts are the same arrays as on a CUDA device, where they are GPU memory; and on Windows, where that GPU memory
    counts as RAM too, a CUDA plan chooses the settings the CPU plan chooses for the same arrays."""
    sm = _pet_sinogram_system_matrix() if model == "pet sinogram" else _pet_listmode_system_matrix(monkeypatch)
    monkeypatch.setattr(memory, "counts_gpu_as_ram", lambda: True)
    _on(monkeypatch, 'cpu')
    cpu = {s: sm.estimate_memory(*s) for s in ((1, 1), (4, 2), (8, 1))}
    assert all(e.gpu_gb == 0 and all('GPU cache' not in p.name for p in e.parts) for e in cpu.values())
    plan = sm.plan_memory(ram_gb=1000, gpu_gb=0)
    assert plan.fits and plan.settings == {'N_splits': 1, 'n_subsets': 1}
    _on(monkeypatch, 'cuda')
    cuda = {s: sm.estimate_memory(*s) for s in cpu}
    for s in cpu:
        arrays = lambda e: {p.name: p.ram_bytes + p.gpu_bytes for p in e.parts if p.scope != 'overhead'}
        assert arrays(cuda[s]) == arrays(cpu[s])                      # the same arrays, only where they are differs
        on_gpu = {p.name for p in cuda[s].parts if p.gpu_bytes > 0 and p.scope != 'overhead'}
        assert any(name.startswith(('normalisation images', 'sensitivity and normalisation images')) for name in on_gpu)
        assert {'image, update and attenuation map', 'projector chunk'} <= on_gpu
    # the budgets of the arrays of 4 subsets and 2 splits on each device: both plans choose those settings
    _on(monkeypatch, 'cpu')
    plan_cpu = sm.plan_memory(ram_gb=cpu[(4, 2)].ram_gb + 1e-9, gpu_gb=0)
    _on(monkeypatch, 'cuda')
    plan_cuda = sm.plan_memory(ram_gb=cuda[(4, 2)].ram_gb + 1e-9, gpu_gb=math.inf)
    assert plan_cpu.fits and plan_cuda.fits and plan_cpu.settings == plan_cuda.settings
    assert sm.estimate_memory(**plan_cuda.settings).ram_gb <= cuda[(4, 2)].ram_gb + 1e-9


@pytest.mark.parametrize("uses_gpu", [False, True], ids=["no CUDA projector", "parallelproj CUDA build on a visible GPU"])
def test_parallelproj_on_the_cpu_device_counts_a_cuda_context(uses_gpu, monkeypatch):
    """With PyTomography on the CPU device, parallelproj's CUDA build still projects on a visible GPU, which makes a CUDA
    context: the estimates of the computations that project with parallelproj (PET, CT, the single scatter simulation)
    then count the overhead of a CUDA process, and SPECT's (PyTorch only) does not. Found by validate_memory.py: a PET
    sinogram OSEM on the CPU device committed 4.06 GB against 3.82 GB estimated with the CPU overhead."""
    pytest.importorskip("parallelproj_core")
    from pytomography.metadata.PET import PETSinogramPolygonProjMeta, PETTOFMeta
    from pytomography.projectors.CT import CTGen3SystemMatrix
    from pytomography.tests.test_ct_fbp import OBJECT_META, _helix
    from pytomography.utils import sss
    projector_uses_gpu = memory.projector_uses_gpu
    monkeypatch.setattr(memory, "projector_uses_gpu", lambda device=None: uses_gpu)
    _on(monkeypatch, 'cpu')
    overhead = lambda estimate: _parts(estimate)['Python, PyTorch and CUDA'].ram_bytes
    expected = memory.FIXED_OVERHEAD_BYTES if uses_gpu else memory.FIXED_OVERHEAD_CPU_BYTES
    pet = _pet_sinogram_system_matrix()
    assert overhead(pet.estimate_memory(4)) == expected
    assert overhead(_pet_listmode_system_matrix(monkeypatch).estimate_memory(4)) == expected
    assert overhead(CTGen3SystemMatrix(OBJECT_META, _helix(), device='cpu').estimate_memory(4)) == expected
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3)
    assert overhead(sss.estimate_sss_memory(PET_OBJECT, PETSinogramPolygonProjMeta(INFO, tof_meta), tof_meta=tof_meta)) == expected
    assert overhead(_spect().estimate_memory(4)) == memory.FIXED_OVERHEAD_CPU_BYTES
    # a plan with no GPU budget still fits: the context is counted as RAM, where Windows counts it
    assert pet.plan_memory(ram_gb=1000, gpu_gb=0).fits
    assert not projector_uses_gpu('cuda')                            # on a CUDA device the overhead is the CUDA one anyway


def test_list_mode_counts_the_detector_ids_of_its_sensitivity_weights(monkeypatch):
    """A list mode system matrix given the lines of response of its sensitivity (detector_ids_sensitivity, which the
    GE HDF5 reader gives) keeps their detector IDs: they are counted, where they are."""
    pytest.importorskip("parallelproj_core")
    from pytomography.metadata.PET import PETLMProjMeta
    from pytomography.projectors.PET import PETLMSystemMatrix
    monkeypatch.setattr(PETLMSystemMatrix, "_backward_full", lambda self, N_splits=20: torch.zeros(self.object_meta.shape))
    gen = torch.Generator().manual_seed(0)
    n = INFO['NrCrystalsPerRing'] * INFO['NrRings']
    d0 = torch.randint(0, n, (3000,), generator=gen)
    ids = torch.stack([d0, (d0 + n // 2) % n], dim=1)
    pairs = torch.combinations(torch.arange(n, dtype=torch.int32))[:20000]
    proj_meta = PETLMProjMeta(ids, INFO, detector_ids_sensitivity=pairs, weights_sensitivity=torch.rand(20000))
    sm = PETLMSystemMatrix(PET_OBJECT, proj_meta, N_splits=2, device='cpu', lor_device='cpu')
    parts = _parts(sm.estimate_memory(4))
    assert parts['detector IDs of the sensitivity weights'].ram_bytes == 20000 * 2 * 4
    assert parts['sensitivity weights'].ram_bytes == 20000 * 4
    without = PETLMSystemMatrix(PET_OBJECT, PETLMProjMeta(ids, INFO), N_splits=2, device='cpu', lor_device='cpu')
    assert 'detector IDs of the sensitivity weights' not in _parts(without.estimate_memory(4))


def test_a_view_counts_at_its_own_size():
    """A view of a larger tensor counts at its own size: the larger tensor is the caller's (e.g. the projections of
    every energy window, of which a reconstruction gets one). An expanded view counts no more than what it repeats."""
    windows = torch.zeros(3, 60, 32, 16)
    assert memory.nbytes(windows[1]) == (4.0 * 60 * 32 * 16, 0.0)
    assert memory.nbytes(windows) == (4.0 * 3 * 60 * 32 * 16, 0.0)
    x = torch.arange(5.0)
    holder = type('Holder', (), {})()
    holder.grid = x[None].expand(1000, 5)
    holder.ids = torch.zeros(100, 3, dtype=torch.int64)[:, :2]
    assert memory.tensor_bytes(holder) == (4.0 * 5 + 8.0 * 100 * 2, 0.0)
    sm = _spect(psf=False)                                    # a reconstruction given one window of three
    assert _parts(sm.estimate_memory(4, held={'photopeak': windows[1]}))['photopeak'].ram_bytes == 4.0 * 60 * 32 * 16


def test_ct_plan_finds_the_fewest_subsets_when_memory_rises_with_them():
    pytest.importorskip("parallelproj_core")
    from pytomography.projectors.CT import CTGen3SystemMatrix
    from pytomography.tests.test_ct_fbp import _helix
    meta = _helix()
    big = CTGen3SystemMatrix(ObjectMeta(dr=(1.0, 1.0, 1.0), shape=(256, 256, 64)), meta)
    ram_gb = big.estimate_memory(n_subsets=8).ram_gb
    plan = big.plan_memory(ram_gb=ram_gb, gpu_gb=1000)
    assert plan.settings == {'N_splits': 1, 'n_subsets': 1}                             # fewer subsets take less here
    assert big.fewest_subsets(ram_gb) == 1


# ------------------------------------------------------------------ SPECT, StarGuide, KEM and motion
def _spect(n_views=60, shape=(32, 32, 16), cache_probabilities=False, psf=True, n_radii=1):
    from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta
    from pytomography.projectors.SPECT import SPECTSystemMatrix
    from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
    object_meta = SPECTObjectMeta([4.0] * 3, shape)
    radii = 200.0 + 10 * (np.arange(n_views) % n_radii)
    proj_meta = SPECTProjMeta((shape[1], shape[2]), [4.0, 4.0], list(np.linspace(0, 360, n_views, endpoint=False)), radii=radii)
    transforms = [SPECTAttenuationTransform(torch.rand(shape) * 0.01, cache_probabilities=cache_probabilities)]
    if psf:
        transforms.append(SPECTPSFTransform(SPECTPSFMeta((0.03, 1.0))))
    return SPECTSystemMatrix(transforms, [], object_meta, proj_meta)


def _parts(estimate) -> dict:
    return {p.name: p for p in estimate.parts}


def _bytes(part) -> float:
    return part.ram_bytes + part.gpu_bytes


def test_spect_estimate_counts_its_arrays():
    sm = _spect()
    image = 4 * 32 * 32 * 16
    padded = 4 * np.prod(sm.object_meta.padded_shape)
    four, eight = _parts(sm.estimate_memory(4)), _parts(sm.estimate_memory(8))
    assert _bytes(four['H_m^T 1 of 4 subsets']) == 4 * image and _bytes(eight['H_m^T 1 of 8 subsets']) == 8 * image
    subset = 'projections of a subset: measured, expected, additive term, of ones, padded copy'
    assert _bytes(eight[subset]) < _bytes(four[subset])
    kept = _bytes(four['what the object transforms keep'])
    assert kept >= image                                                                 # the attenuation map, and PSF kernels
    cached = _parts(_spect(cache_probabilities=True).estimate_memory(4))
    assert _bytes(cached['what the object transforms keep']) == pytest.approx(kept + 60 * padded)
    view = _bytes(four["one view: rotated object and the transforms' temporaries"])
    assert view == padded + 6 * padded                     # the rotated object, and the attenuation's six (the PSF's three)
    assert sm.estimate_memory(4).settings == {'n_subsets': 4} and list(sm._memory_settings()) == ['n_subsets']
    assert sm._memory_settings()['n_subsets'][1] == list(range(1, 61))
    with pytest.raises(ValueError, match="has no setting N_splits"):
        sm.plan_memory(ram_gb=10, gpu_gb=10, N_splits=2)


def test_spect_normalisation_images_are_images_of_their_own():
    """A SPECT reconstruction keeps an H_m^T 1 image for each subset, which its estimate counts as one image each: each
    is an image of its own, not a view of the padded object back projection works in, which kept about twice the image
    for each subset (found by validate_memory.py: a 128^3 OSEM with 30 subsets peaked at 1.14 times its GPU estimate)."""
    sm = _spect(psf=False)
    image = 4 * np.prod(sm.object_meta.shape)
    assert np.prod(sm.object_meta.padded_shape) > np.prod(sm.object_meta.shape)
    likelihood = _likelihood(sm)
    likelihood._set_n_subsets(4)
    assert len(likelihood.norm_BPs) == 4
    assert all(norm.untyped_storage().nbytes() == image for norm in likelihood.norm_BPs)
    likelihood._set_n_subsets(1)
    assert likelihood.norm_BP.untyped_storage().nbytes() == image


def test_spect_psf_operator_and_its_kernel_cache():
    from pytomography.transforms.SPECT import SPECTPSFTransform
    from pytomography.tests.test_psf2d_operator import _KernelOperator
    sm = _spect(psf=False)

    class _Operator(_KernelOperator):
        def _get_kernel(self, xv, yv, a):
            return torch.ones(len(a), xv.shape[0], xv.shape[1]) / xv.numel()
    without = _bytes(_parts(sm.estimate_memory(2))['what the object transforms keep'])
    for cache in (False, True):
        transform = SPECTPSFTransform(psf_operator=_Operator(None), cache_kernel=cache)
        transform.configure(sm.object_meta, sm.proj_meta)
        sm.obj2obj_transforms = sm.obj2obj_transforms[:1] + [transform]
        parts = _parts(sm.estimate_memory(2))
        Lx, Ly, Lz = sm.object_meta.padded_shape
        k = sm.object_meta.shape[0] - 1
        width = Lz + k - 1 + (Lz + k - 1) % 2
        spectrum = 8 * Lx * (Ly + k - 1) * (width // 2 + 1)
        extra = _bytes(parts['what the object transforms keep']) - without
        assert (extra >= spectrum) == cache
        assert _bytes(parts["one view: rotated object and the transforms' temporaries"]) >= 8 * 4 * Lx * (Ly + k - 1) * width


def test_monte_carlo_hybrid_adds_simind(monkeypatch):
    from pytomography.projectors.SPECT import dualhead_system_matrix, MonteCarloHybridSPECTSystemMatrix
    monkeypatch.setattr(dualhead_system_matrix.shutil, "which", lambda name: "simind")
    base = _spect(psf=False)
    sm = MonteCarloHybridSPECTSystemMatrix(base.object_meta, base.proj_meta, n_events=10**6, n_parallel=6,
                                           obj2obj_transforms=base.obj2obj_transforms, proj2proj_transforms=[],
                                           attenuation_map_140keV=torch.rand(base.object_meta.shape), energy_window_params=['a', 'b', 'c'],
                                           primary_window_idx=1, isotope_names=['lu177'], isotope_ratios=[1.0], collimator_type='SY-ME',
                                           crystal_thickness_mm=9.5, cover_thickness_mm=1.0, backscatter_thickness_mm=50.0)
    parts = _parts(sm.estimate_memory(4))
    simind = parts['SIMIND processes (6)']
    assert simind.ram_bytes >= 6 * sm.SIMIND_PROCESS_BYTES and simind.scope == 'chunk'
    assert parts['attenuation map at 140 keV'].ram_bytes == 4 * np.prod(base.object_meta.shape)
    assert sm.estimate_memory(4).ram_gb > base.estimate_memory(4).ram_gb


def test_extended_system_matrix_adds_up_its_system_matrices():
    """ExtendedSystemMatrix (two photopeaks in one reconstruction, as t_dualpeak): the first system matrix's arrays,
    with the projections of a subset counted for each photopeak, kept and stacked; what the other system matrix holds
    of its own (its attenuation map, not the shared images and caches); and the projections of ones of both, for
    H^T 1. A system matrix given twice is counted once. The algorithms estimate it through it, as for any system
    matrix."""
    from pytomography.projectors import ExtendedSystemMatrix
    from pytomography.likelihoods import PoissonLogLikelihood
    from pytomography.algorithms import OSEM
    a, b = _spect(psf=True), _spect(psf=False)
    ext = ExtendedSystemMatrix([a, b])
    alone, both = _parts(a.estimate_memory(8)), _parts(ext.estimate_memory(8))
    name = 'projections of a subset: measured, expected, additive term, of ones, padded copy'
    assert _bytes(both[f'{name}, for each of 2, kept and stacked']) == 2 * 2 * _bytes(alone[name])
    own = [p for p in b.estimate_memory(8).parts if p.scope == 'held' and p.name == 'what the object transforms keep']
    assert _bytes(both['system matrix 2: what it holds']) == sum(_bytes(p) for p in own) > 0
    assert _bytes(both['projections of ones of all 2 (for H^T 1)']) == 4 * 2 * np.prod(a.proj_meta.shape)
    for key, part in alone.items():                  # the rest is the first system matrix's (but the overheads, which grow)
        if key != name and part.scope != 'overhead':
            assert key in both and _bytes(both[key]) == _bytes(part), key
    assert 'system matrix 2: what it holds' not in _parts(ExtendedSystemMatrix([a, a]).estimate_memory(8))
    single = a.estimate_memory(8)
    joint = ext.estimate_memory(8)
    assert joint.ram_gb + joint.gpu_gb > single.ram_gb + single.gpu_gb
    projections = torch.ones((2, *a.proj_meta.shape))
    algorithm = OSEM(PoissonLogLikelihood(ext, projections, torch.zeros_like(projections)))
    estimate = algorithm.estimate_memory(n_subsets=8)
    assert estimate.ram_gb + estimate.gpu_gb >= joint.ram_gb + joint.gpu_gb - 1e-9


def _starguide(psf=True):
    from pytomography.tests.test_starguide_projector import _system
    sm, f, g = _system(psf_params=(0.02, 0.05) if psf else None)
    return sm


def test_starguide_estimate():
    sm = _starguide()
    assert sm._max_subsets() == 6 and sm._memory_settings()['n_subsets'][1] == list(range(1, 7))
    two, three = _parts(sm.estimate_memory(2)), _parts(sm.estimate_memory(3))
    subset = 'projections of a subset: measured, expected, additive term, scaled, of ones'
    assert _bytes(three[subset]) <= _bytes(two[subset])
    assert _bytes(three['H_m^T 1 of 3 subsets']) == 3 * 4 * 40 * 40 * 12
    angle = next(p for name, p in two.items() if name.startswith('one angle (4 views)'))
    without_psf = next(p for name, p in _parts(_starguide(psf=False).estimate_memory(2)).items() if name.startswith('one angle'))
    assert _bytes(angle) > _bytes(without_psf)                                          # the PSF widens the slab and convolves it
    plan = sm.plan_memory(ram_gb=1000, gpu_gb=1000, n_subsets=3)
    assert plan.settings == {'n_subsets': 3}


def test_kem_and_motion_pass_through_to_their_system_matrices():
    from pytomography.projectors.shared import KEMSystemMatrix, MotionSystemMatrix
    from pytomography.transforms.shared import KEMTransform, DVFMotionTransform
    sm = _spect(psf=False)
    image = 4 * np.prod(sm.object_meta.shape)
    kem = KEMSystemMatrix(sm, KEMTransform([torch.rand(sm.object_meta.shape)], size=3))
    parts = _parts(kem.estimate_memory(4))
    assert _bytes(parts['KEM kernel']) == 27 * image and _bytes(parts['KEM support images']) == image
    assert set(_parts(sm.estimate_memory(4))) < set(parts)
    assert kem._memory_settings().keys() == sm._memory_settings().keys()
    assert kem.plan_memory(ram_gb=1000, gpu_gb=1000).settings == {'n_subsets': 1}
    dvf = torch.zeros(*sm.object_meta.shape, 3)
    others = [_spect(psf=False) for _ in range(2)]
    motion = MotionSystemMatrix([sm] + others, [DVFMotionTransform(dvf.clone(), dvf.clone()) for _ in range(3)])
    parts = _parts(motion.estimate_memory(4))
    assert _bytes(parts['projections of ones of all 3 (for H^T 1)']) == 3 * 4 * np.prod(sm.proj_meta.shape)
    assert 'system matrix 2: what it holds' in parts and 'system matrix 3: what it holds' in parts
    assert _bytes(parts['arrays of the 3 transforms']) == 3 * 2 * 3 * image
    same = _parts(MotionSystemMatrix([sm, sm, sm], [DVFMotionTransform(dvf, dvf) for _ in range(3)]).estimate_memory(4))
    assert 'system matrix 2: what it holds' not in same                                 # one system matrix, counted once
    assert motion.plan_memory(ram_gb=1000, gpu_gb=1000).settings == {'n_subsets': 1}


@pytest.mark.parametrize("model", ["spect", "starguide", "kem", "motion", "pet sinogram", "pet list mode", "ct helical", "ct cone beam"])
def test_every_system_model_estimates_and_plans(model, monkeypatch, gpu_as_ram):
    if model == "spect":
        sm = _spect()
    elif model == "starguide":
        sm = _starguide()
    elif model == "kem":
        from pytomography.projectors.shared import KEMSystemMatrix
        from pytomography.transforms.shared import KEMTransform
        sm = KEMSystemMatrix(_spect(), KEMTransform([torch.rand(32, 32, 16)], size=3))
    elif model == "motion":
        from pytomography.projectors.shared import MotionSystemMatrix
        from pytomography.transforms.shared import DVFMotionTransform
        dvf = torch.zeros(32, 32, 16, 3)
        sm = MotionSystemMatrix([_spect(), _spect()], [DVFMotionTransform(dvf, dvf) for _ in range(2)])
    elif model == "pet sinogram":
        sm = _pet_sinogram_system_matrix()
    elif model == "pet list mode":
        sm = _pet_listmode_system_matrix(monkeypatch)
    else:
        pytest.importorskip("parallelproj_core")
        from pytomography.projectors.CT import CTGen3SystemMatrix, CTConeBeamFlatPanelSystemMatrix
        from pytomography.tests.test_ct_fbp import OBJECT_META, _conebeam, _helix
        sm = CTGen3SystemMatrix(OBJECT_META, _helix()) if model == "ct helical" else \
            CTConeBeamFlatPanelSystemMatrix(ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(32, 32, 24)), _conebeam(120))
    estimate = sm.estimate_memory(n_subsets=2)
    own = sum(_bytes(p) for p in estimate.parts if p.scope != 'overhead')
    assert own > 0 and estimate.settings['n_subsets'] == 2
    _roundtrip(estimate)
    str(estimate).encode('cp1252')
    settings = sm._memory_settings()
    plan = sm.plan_memory(ram_gb=1000, gpu_gb=1000)
    assert plan.fits and plan.settings == {name: values[0] for name, (_, values) in settings.items()}
    tight = sm.plan_memory(ram_gb=estimate.ram_gb, gpu_gb=estimate.gpu_gb)
    assert tight.fits and tight.estimate.ram_gb <= estimate.ram_gb + 1e-9 and tight.estimate.gpu_gb <= estimate.gpu_gb + 1e-9
    assert not sm.plan_memory(ram_gb=0.1, gpu_gb=0).fits
    _roundtrip(tight)


# ------------------------------------------------------------------ reconstruction algorithms
def _likelihood(sm, additive=True):
    from pytomography.likelihoods import PoissonLogLikelihood
    gen = torch.Generator().manual_seed(0)
    projections = torch.rand(sm.proj_meta.shape, generator=gen)
    return PoissonLogLikelihood(sm, projections, additive_term=torch.rand(sm.proj_meta.shape, generator=gen) if additive else None)


def test_algorithms_delegate_to_their_system_matrix(gpu_as_ram):
    from pytomography.algorithms import OSEM, BSREM, MLEM, OSMAPOSL
    from pytomography.priors import RelativeDifferencePrior
    sm = _spect(psf=False)
    likelihood = _likelihood(sm)
    held = {'projections (the data)': likelihood.projections, 'additive term': likelihood.additive_term}
    osem = OSEM(likelihood)
    estimate = osem.estimate_memory(n_subsets=4)
    assert estimate.title.startswith('OSEM (SPECTSystemMatrix): 4 subsets') and estimate.settings == {'n_subsets': 4}
    assert estimate.ram_gb == pytest.approx(sm.estimate_memory(4, held=held).ram_gb)
    assert estimate.gpu_gb == pytest.approx(sm.estimate_memory(4, held=held).gpu_gb)
    osem.n_subsets = 6                                        # what the last call used, by default
    assert osem.estimate_memory().settings == {'n_subsets': 6}
    # a prior's arrays, and BSREM's stacked H_m^T 1, are the algorithm's own
    image = 4 * np.prod(sm.object_meta.shape)
    own = lambda e: sum(_bytes(p) for p in e.parts if p.scope != 'overhead')
    bsrem = BSREM(likelihood, prior=RelativeDifferencePrior(beta=1))
    osmaposl = OSMAPOSL(likelihood, prior=RelativeDifferencePrior(beta=1))
    assert own(osmaposl.estimate_memory(4)) == pytest.approx(own(estimate) + 9 * image)
    assert own(bsrem.estimate_memory(4)) == pytest.approx(own(estimate) + 9 * image + 5 * image)
    assert own(bsrem.estimate_memory(1)) == pytest.approx(own(osem.estimate_memory(1)) + 9 * image)
    # MLEM has one subset, whatever it is asked
    mlem = MLEM(likelihood)
    assert mlem.estimate_memory(n_subsets=8).settings == {'n_subsets': 1}
    assert mlem.plan_memory(ram_gb=1000, gpu_gb=1000, n_subsets=8).settings == {'n_subsets': 1}


def test_algorithm_plans_keep_the_subsets_and_agree_with_their_estimates(gpu_as_ram):
    from pytomography.algorithms import BSREM
    from pytomography.priors import QuadraticPrior
    bsrem = BSREM(_likelihood(_spect()), prior=QuadraticPrior(beta=1))
    plan = bsrem.plan_memory(ram_gb=1000, gpu_gb=1000, n_subsets=8)
    assert plan.fits and plan.settings == {'n_subsets': 8} and plan.title == 'BSREM (SPECTSystemMatrix)'
    ram_gb = bsrem.estimate_memory(n_subsets=3).ram_gb
    plan = bsrem.plan_memory(ram_gb=ram_gb, gpu_gb=1000)
    assert plan.fits and bsrem.estimate_memory(**plan.settings).ram_gb <= ram_gb
    assert plan.estimate.ram_gb == pytest.approx(bsrem.estimate_memory(**plan.settings).ram_gb)
    assert not bsrem.plan_memory(ram_gb=0.1, gpu_gb=0).fits
    _roundtrip(plan)


def test_kem_sart_and_multibed_reconstructions():
    from pytomography.algorithms import KEM, SART, OSEM, PGAAMultiBedSPECT
    from pytomography.likelihoods import PoissonLogLikelihood
    from pytomography.projectors.shared import KEMSystemMatrix
    from pytomography.transforms.shared import KEMTransform
    sm = _spect(psf=False)
    kem_sm = KEMSystemMatrix(sm, KEMTransform([torch.rand(sm.object_meta.shape)], size=3))
    kem = KEM(PoissonLogLikelihood(kem_sm, torch.rand(sm.proj_meta.shape)))
    assert 'KEM kernel' in _parts(kem.estimate_memory(2)) and kem.plan_memory(ram_gb=1000, gpu_gb=1000).fits
    sart = SART(sm, torch.rand(sm.proj_meta.shape))
    assert sart.estimate_memory(3).settings == {'n_subsets': 3} and sart.plan_memory(ram_gb=1000, gpu_gb=1000, n_subsets=5).fits
    beds = [OSEM(_likelihood(_spect(psf=False))) for _ in range(3)]
    multibed = PGAAMultiBedSPECT(beds)
    estimate = multibed.estimate_memory(n_subsets=4)
    parts = _parts(estimate)
    assert {f'bed {k}: what its reconstruction holds' for k in (1, 2, 3)} <= set(parts)
    one = beds[0].estimate_memory(4)
    held = sum(_bytes(p) for p in one.parts if p.scope == 'held')
    assert _bytes(parts['bed 2: what its reconstruction holds']) == pytest.approx(held)
    assert estimate.ram_gb + estimate.gpu_gb > one.ram_gb + one.gpu_gb
    plan = multibed.plan_memory(ram_gb=1000, gpu_gb=1000, n_subsets=4)
    assert plan.settings == {'n_subsets': 4} and plan.estimate.ram_gb == pytest.approx(estimate.ram_gb)
    _roundtrip(plan)


# ------------------------------------------------------------------ filtered back projection
def test_fbp_plan_lowers_its_budgets_only_when_needed():
    pytest.importorskip("parallelproj_core")
    from pytomography.algorithms import FilteredBackProjection
    from pytomography.projectors.CT import CTGen3SystemMatrix
    from pytomography.tests.test_ct_fbp import OBJECT_META, _helix, _line_integrals
    from pytomography.utils.memory import DEFAULT_BUDGET, memory_budget_set
    meta = _helix()
    proj = _line_integrals(meta)
    fbp = FilteredBackProjection(proj, CTGen3SystemMatrix(OBJECT_META, meta))
    estimate = fbp.estimate_memory()
    assert estimate.settings == {'gpu_budget': DEFAULT_BUDGET, 'memory_budget_gb': None}
    plan = fbp.plan_memory(ram_gb=1000, gpu_gb=1000)
    assert plan.fits and plan.settings == estimate.settings and len(plan.options) == 1
    # a RAM budget the present chunk of rebinned views does not fit: a lower memory budget, which halves the chunk
    smaller = dict(estimate.alternatives)
    assert any(label.startswith('set_memory_budget') for label in smaller)
    ram_gb = min(e.ram_gb for e in smaller.values()) + 1e-9
    plan = fbp.plan_memory(ram_gb=ram_gb, gpu_gb=1000)
    assert plan.fits and plan.settings != estimate.settings
    with memory_budget_set(plan.settings['memory_budget_gb']):
        again = FilteredBackProjection(proj, fbp.system_matrix, gpu_budget=plan.settings['gpu_budget']).estimate_memory()
    assert again.ram_gb == pytest.approx(plan.estimate.ram_gb) and again.ram_gb <= ram_gb
    assert pytomography.memory_budget is None                                           # the plan sets nothing
    kept = fbp.plan_memory(ram_gb=1000, gpu_gb=1000, gpu_budget=0.1e9)
    assert kept.settings['gpu_budget'] == 0.1e9 and kept.fixed == ['gpu_budget']
    assert not fbp.plan_memory(ram_gb=0.1, gpu_gb=0).fits
    _roundtrip(plan)


def test_fbp_plans_with_nothing_to_choose():
    from pytomography.algorithms import FilteredBackProjection
    sm = _spect(psf=False)
    fbp = FilteredBackProjection(torch.zeros(sm.proj_meta.shape), sm)
    estimate = fbp.estimate_memory()
    assert estimate.settings == {}
    plan = fbp.plan_memory(ram_gb=estimate.ram_gb, gpu_gb=estimate.gpu_gb)
    assert plan.fits and plan.settings == {} and plan.estimate.ram_gb == pytest.approx(estimate.ram_gb)
    assert not fbp.plan_memory(ram_gb=estimate.ram_gb - 0.01, gpu_gb=estimate.gpu_gb).fits
    _roundtrip(plan)


# ------------------------------------------------------------------ single scatter simulation
BIG_INFO = dict(INFO, rsectorTransNr=32, crystalTransNr=8, NrCrystalsPerRing=256, moduleAxialNr=4, crystalAxialNr=8, NrRings=32)   # large enough to matter


def test_sss_estimate_follows_n_splits_and_lazy():
    pytest.importorskip("parallelproj_core")
    from pytomography.metadata.PET import PETSinogramPolygonProjMeta, PETTOFMeta
    from pytomography.utils import sss
    tof_meta = PETTOFMeta(21, 300.0, 60.0, n_sigmas=3)
    proj_meta = PETSinogramPolygonProjMeta(BIG_INFO, tof_meta)
    sizes = sss._sss_sizes(proj_meta, tof_meta, 4, 4)
    assert sizes['lors'] == (9 * 64) * (9 * 64 - 1) // 2 and sizes['sampled_planes'] == 9 ** 2   # rings 0, 4, ... 28 and 31
    # every other ring and crystal sampled: 2.4 million lines of response, whose TOF kernel is the largest step
    fine = dict(sinogram_interring_stepsize=2, sinogram_intraring_stepsize=2)
    one = sss.estimate_sss_memory(PET_OBJECT, proj_meta, tof_meta=tof_meta, N_splits=1, lazy=True, **fine)
    seven = sss.estimate_sss_memory(PET_OBJECT, proj_meta, tof_meta=tof_meta, N_splits=7, lazy=True, **fine)
    assert one.settings == {'lazy': True, 'N_splits': 1} and seven.settings == {'lazy': True, 'N_splits': 7}
    if not sss._sss_cuda.available(pytomography.device, 21):                            # the TOF kernel is built by PyTorch
        assert any('sparse estimate' in p.name for p in one.parts)
        assert one.ram_gb + one.gpu_gb > seven.ram_gb + seven.gpu_gb + 2
    dense = sss.estimate_sss_memory(PET_OBJECT, proj_meta, tof_meta=tof_meta, N_splits=7, lazy=False)
    sinogram = 4 * 128 * 257 * 32 ** 2 * 21
    assert _bytes(_parts(dense)['interpolated estimate (dense) and its scaled copy (host)']) == 2 * sinogram
    assert dense.ram_gb > seven.ram_gb
    _roundtrip(dense)
    str(dense).encode('cp1252')
    non_tof = sss.estimate_sss_memory(PET_OBJECT, PETSinogramPolygonProjMeta(BIG_INFO, None), proj_data=torch.zeros(128, 257, 32 ** 2))
    assert non_tof.settings == {'lazy': False} and _bytes(_parts(non_tof)['data']) == 4 * 128 * 257 * 32 ** 2


def test_sss_plan_chooses_lazy_and_n_splits():
    pytest.importorskip("parallelproj_core")
    from pytomography.metadata.PET import PETSinogramPolygonProjMeta, PETTOFMeta
    from pytomography.utils import sss
    tof_meta = PETTOFMeta(21, 300.0, 60.0, n_sigmas=3)
    arguments = dict(object_meta=PET_OBJECT, proj_meta=PETSinogramPolygonProjMeta(BIG_INFO, tof_meta), tof_meta=tof_meta)
    plan = sss.plan_sss_memory(**arguments, ram_gb=1000, gpu_gb=1000)
    assert plan.fits and plan.settings == {'lazy': False, 'N_splits': 1}               # dense, and the TOF bins together
    budget = sss.estimate_sss_memory(**arguments, lazy=True, N_splits=3)
    plan = sss.plan_sss_memory(**arguments, ram_gb=budget.ram_gb, gpu_gb=budget.gpu_gb)
    assert plan.fits and plan.settings['lazy'] is True and plan.settings['N_splits'] <= 3
    assert sss.estimate_sss_memory(**arguments, **plan.settings).ram_gb <= budget.ram_gb
    kept = sss.plan_sss_memory(**arguments, ram_gb=1000, gpu_gb=1000, N_splits=5)
    assert kept.settings == {'lazy': False, 'N_splits': 5} and kept.fixed == ['N_splits']
    nothing = sss.plan_sss_memory(**arguments, ram_gb=0.1, gpu_gb=0)
    assert not nothing.fits and len(nothing.options) == 2 * 21
    _roundtrip(nothing)
