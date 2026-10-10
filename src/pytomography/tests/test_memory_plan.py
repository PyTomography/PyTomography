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
    monkeypatch.setattr(memory, "_host_memory", lambda: (64e9, 40e9, 3e9))     # 64 GB machine, 40 GB free, this process 3 GB
    ram, gpu = memory.available_memory()
    assert ram == pytest.approx(40 - 16 + memory.FIXED_OVERHEAD_BYTES / 1e9)    # 16 GB kept free; the overhead is ours
    assert gpu == pytest.approx(20 - 0.05 * 32)
    monkeypatch.setattr(memory, "_host_memory", lambda: (16e9, 10e9, 1e9))     # a 16 GB laptop: a quarter kept free
    assert memory.available_memory()[0] == pytest.approx(10 - 4 + 1)
    monkeypatch.setattr(memory, "_host_memory", lambda: (16e9, 2e9, 1e9))
    assert memory.available_memory()[0] == 0
    # plans without budgets are for what is available
    monkeypatch.setattr(memory, "_host_memory", lambda: (64e9, 40e9, 3e9))
    plan = _ToySystemMatrix().plan_memory()
    assert plan.ram_gb == pytest.approx(ram) and plan.gpu_gb == pytest.approx(gpu)
    assert plan.budget_from == {'ram_gb': 'available', 'gpu_gb': 'available'} and 'available now' in str(plan)
    plan = _ToySystemMatrix().plan_memory(ram_gb=7)
    assert plan.ram_gb == 7 and plan.gpu_gb == pytest.approx(gpu) and plan.budget_from == {'ram_gb': 'given', 'gpu_gb': 'available'}


def test_available_memory_reads_this_machine():
    total, available, process = memory._host_memory()
    assert total > 0 and 0 < available <= total and process > 0
    ram, gpu = memory.available_memory()
    assert 0 <= ram <= total / 1e9 and gpu >= 0
    if not torch.cuda.is_available():
        assert gpu == 0 and memory.available_memory('cpu')[1] == 0


# ------------------------------------------------------------------ PET and CT system matrices
INFO = dict(min_rsector_difference=0, crystal_length=20.0, radius=120.0, firstCrystalAxis=0,
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
