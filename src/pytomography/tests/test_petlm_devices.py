"""The list mode PET system matrix keeps its LORs on the projection device and hands them to the projector in
sinogram order. Neither changes what is computed: these tests check that where the LORs are kept, and whether they
are ordered, does not change the projections, that subsets stay consistent, that the memory report does not
promise less than a projection actually uses, and that each event gets its own crystal pair's normalization weight."""
from __future__ import annotations

import re

import numpy as np
import pytest
import torch

import pytomography

parallelproj_core = pytest.importorskip("parallelproj_core", reason="PET projection requires parallelproj 2")
from pytomography.io.PET.shared import crystal_pair_index
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta
from pytomography.projectors.PET import PETLMSystemMatrix

DEV = pytomography.device
INFO = dict(min_rsector_difference=0, crystal_length=20.0, radius=120.0, firstCrystalAxis=0,
            rsectorTransNr=16, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=2, moduleTransSpacing=0.0,
            moduleAxialSpacing=17.0, submoduleTransNr=1, submoduleAxialNr=1, submoduleTransSpacing=0.0,
            submoduleAxialSpacing=0.0, crystalTransNr=4, crystalAxialNr=4, crystalTransSpacing=4.0,
            crystalAxialSpacing=4.0, NrCrystalsPerRing=64, NrRings=8)
N_EVENTS = 4000
N_DETECTORS = INFO['NrCrystalsPerRing'] * INFO['NrRings']


def _events(tof_meta=None, seed=0):
    gen = torch.Generator().manual_seed(seed)
    d0 = torch.randint(0, N_DETECTORS, (N_EVENTS,), generator=gen)
    d1 = (d0 + N_DETECTORS // 2 + torch.randint(-40, 40, (N_EVENTS,), generator=gen)) % N_DETECTORS
    columns = [d0, d1]
    if tof_meta is not None:
        columns.append(torch.randint(0, tof_meta.num_bins, (N_EVENTS,), generator=gen))
    return torch.stack(columns, dim=1)


def _system(tof=False, lor_device=DEV, sort_events=True, n_subsets=3, seed=0):
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3) if tof else None
    proj_meta = PETLMProjMeta(_events(tof_meta, seed), INFO, tof_meta=tof_meta)
    sm = PETLMSystemMatrix(ObjectMeta(dr=(4, 4, 4), shape=(24, 24, 16)), proj_meta,
                           lor_device=lor_device, sort_events=sort_events, N_splits=2)
    sm.set_n_subsets(n_subsets)
    return sm


def _object(seed=1):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(24, 24, 16, generator=gen).to(DEV)


@pytest.mark.parametrize("tof", [False, True])
def test_lor_device_does_not_change_the_projection(tof):
    """Keeping the LORs on the CPU is a memory/speed trade, not a different computation. Forward projections are
    identical; back projections agree to float32 rounding, since the projector accumulates into the image with
    atomic additions whose order varies between launches (repeating one back projection differs by ~1e-7 too)."""
    obj = _object()
    on_device, on_cpu = _system(tof, lor_device=DEV), _system(tof, lor_device='cpu')
    for subset_idx in (None, 1):
        a, b = on_device.forward(obj, subset_idx), on_cpu.forward(obj, subset_idx)
        assert torch.equal(a, b)
        ga = torch.rand(a.shape[0], generator=torch.Generator().manual_seed(2)).to(a.device)
        ba, bb = on_device.backward(ga, subset_idx), on_cpu.backward(ga, subset_idx)
        assert (ba - bb).abs().max() <= 1e-5 * bb.abs().max()


@pytest.mark.parametrize("tof", [False, True])
def test_event_ordering_does_not_change_the_projection(tof):
    """Ordering only changes the order the LORs are given to the projector, so forward projections are unchanged and
    back projections change only by the order of their atomic additions."""
    obj = _object()
    ordered, unordered = _system(tof, sort_events=True), _system(tof, sort_events=False)
    for subset_idx in (None, 0, 2):
        a, b = ordered.forward(obj, subset_idx), unordered.forward(obj, subset_idx)
        assert torch.equal(a, b), "forward projections must be identical, event for event"
        g = torch.rand(a.shape[0], generator=torch.Generator().manual_seed(3)).to(a.device)
        ba, bb = ordered.backward(g, subset_idx), unordered.backward(g, subset_idx)
        assert (ba - bb).abs().max() <= 1e-4 * bb.abs().max()


def test_subsets_partition_the_events():
    """Every event appears in exactly one subset, and a subset projection equals those events of the full one."""
    obj = _object()
    sm = _system(tof=True, n_subsets=3)
    full = sm.forward(obj)
    covered = torch.zeros(N_EVENTS, dtype=torch.bool)
    for k in range(3):
        idx = sm.subset_indices_array[k].cpu()
        assert not covered[idx].any()
        covered[idx] = True
        assert torch.equal(sm.forward(obj, k), full[idx.to(full.device)])
    assert covered.all()


def test_changing_the_number_of_subsets_reorders_the_events():
    """The order the events of a subset are handed to the projector in is kept per subset; a new partition (another
    number of subsets, as when a second reconstruction uses other subsets) must not reuse the old one's orders, which
    could raise an error or drop events."""
    obj = _object()
    sm = _system(tof=True, n_subsets=2)
    first = [sm.forward(obj, k) for k in range(2)]
    sm.set_n_subsets(3)
    full = sm.forward(obj)
    for k in range(3):
        idx = sm.subset_indices_array[k].cpu()
        assert torch.equal(sm.forward(obj, k), full[idx.to(full.device)])
    sm.set_n_subsets(2)
    assert all(torch.equal(sm.forward(obj, k), first[k]) for k in range(2))


@pytest.mark.parametrize("sort_events", [False, True])
def test_memory_report_is_not_optimistic(capsys, sort_events):
    """The reported peak must not be below what a projection actually uses."""
    if not str(DEV).startswith('cuda'):
        pytest.skip("needs CUDA to measure memory")
    # Memory the process holds before this system matrix exists is not part of what the report estimates. The cuBLAS
    # workspace (8.5 MB), which the first GPU matrix product of any earlier test allocates for the life of the process,
    # made this test fail whenever it ran after one.
    baseline = torch.cuda.memory_allocated()
    sm = _system(tof=True, sort_events=sort_events, n_subsets=2)
    obj = _object()
    sm.forward(obj, 0)                                  # build any ordering first
    sm.print_memory_usage()
    report = capsys.readouterr().out
    assert 'estimated peak' in report and 'resident on' in report
    estimates = [float(gb) for line in report.splitlines() if 'estimated peak' in line
                 for gb in re.findall(r'([0-9]+\.[0-9]+)\s*GB', line)]
    assert estimates
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    g = sm.forward(obj); sm.backward(g)
    torch.cuda.synchronize()
    measured = (torch.cuda.max_memory_allocated() - baseline) / 1e9
    # the report is printed to 3 decimals, so allow a rounding unit on top of it
    assert max(estimates) + 1e-3 >= measured, f"reported {max(estimates):.3f} GB but a projection used {measured:.3f} GB"


def test_falls_back_when_the_scanner_geometry_is_unavailable():
    """Without ``info`` the events cannot be put in sinogram order; the projector must still work."""
    from pytomography.io.PET.shared import get_scanner_LUT
    proj_meta = PETLMProjMeta(_events(), scanner_LUT=get_scanner_LUT(INFO))
    sm = PETLMSystemMatrix(ObjectMeta(dr=(4, 4, 4), shape=(24, 24, 16)), proj_meta, N_splits=2)
    sm.set_n_subsets(2)
    assert sm._event_order(0) is None
    assert sm.forward(_object(), 0).shape[0] == sm.subset_indices_array[0].shape[0]


@pytest.mark.parametrize("tof", [False, True])
def test_attenuation_scaling_without_weights_matches_attenuation_weights(tof):
    """With ``scale_projection_by_sensitivity`` and no ``weights``, the projections are scaled by the attenuation
    probabilities (#220): the same as passing those probabilities as ``weights``. The events are projected in sinogram
    order and put back, so the factors must be those of the events in the order they were given."""
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3) if tof else None
    events = _events(tof_meta)
    object_meta = ObjectMeta(dr=(4, 4, 4), shape=(24, 24, 16))
    gen = torch.Generator().manual_seed(3)
    attenuation = (0.01 * torch.rand(24, 24, 16, generator=gen)).to(DEV)
    by_map = PETLMSystemMatrix(object_meta, PETLMProjMeta(events, INFO, tof_meta=tof_meta), attenuation_map=attenuation,
                               scale_projection_by_sensitivity=True, N_splits=2)
    probabilities = by_map._compute_attenuation_probability_projection(by_map.proj_meta.detector_ids).cpu()
    by_weights = PETLMSystemMatrix(object_meta, PETLMProjMeta(events, INFO, tof_meta=tof_meta, weights=probabilities),
                                   scale_projection_by_sensitivity=True, N_splits=2)
    assert by_map._event_order(None) is not None             # the events really are reordered
    x = _object()
    assert torch.allclose(by_map.forward(x), by_weights.forward(x), rtol=1e-5)
    by_map.set_n_subsets(3), by_weights.set_n_subsets(3)
    y = torch.rand(by_map.subset_indices_array[1].shape[0], generator=gen).to(DEV)
    assert torch.allclose(by_map.forward(x, 1), by_weights.forward(x, 1), rtol=1e-5)
    assert torch.allclose(by_map.backward(y, 1), by_weights.backward(y, 1), rtol=1e-4, atol=1e-6)


def test_crystal_pair_index_is_the_combinations_order():
    """The index of a crystal pair in ``torch.combinations`` order, exactly, up to the 411 million pairs of the mMR (in
    float32, about 9 in 10 of the mMR's crystal pairs were given a neighbouring pair's normalization weight)."""
    pairs = torch.combinations(torch.arange(40), 2)
    assert torch.equal(crystal_pair_index(pairs[:, 0], pairs[:, 1], 40), torch.arange(pairs.shape[0]))
    n = 28672                                                 # crystals of the Siemens Biograph mMR
    gen = torch.Generator().manual_seed(2)
    a, b = torch.randint(0, n, (2, 100_000), generator=gen).sort(dim=0).values
    a, b = a[a < b], b[a < b]
    counts = n - 1 - torch.arange(n)                          # pairs (i, j > i) of each crystal i
    first_pair_of = torch.cumsum(counts, 0) - counts
    assert torch.equal(crystal_pair_index(a.to(torch.int32), b.to(torch.int32), n), first_pair_of[a] + b - a - 1)


def test_each_event_gets_its_crystal_pairs_normalization_weight():
    """``_compute_sensitivity_projection(all_ids=False)``: the normalization weight of each event's crystal pair, whichever
    crystal the event lists first."""
    events = _events()
    pairs = torch.combinations(torch.arange(N_DETECTORS), 2)
    weights = torch.rand(pairs.shape[0], generator=torch.Generator().manual_seed(5)) + 0.5
    pair_of = torch.full((N_DETECTORS, N_DETECTORS), -1)
    pair_of[pairs[:, 0], pairs[:, 1]] = pair_of[pairs[:, 1], pairs[:, 0]] = torch.arange(pairs.shape[0])
    assert (events[:, 0] > events[:, 1]).any() and (events[:, 0] < events[:, 1]).any()
    sm = PETLMSystemMatrix(ObjectMeta(dr=(4, 4, 4), shape=(24, 24, 16)),
                           PETLMProjMeta(events, INFO, weights_sensitivity=weights), N_splits=2)
    assert torch.equal(sm._compute_sensitivity_projection(all_ids=False), weights[pair_of[events[:, 0], events[:, 1]]])
