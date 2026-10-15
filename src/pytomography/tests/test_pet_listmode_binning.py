"""``listmode_to_sinogram`` finds the bins of list mode events on the events' own device and adds them into one
sinogram, instead of ``torch.histogramdd`` on the CPU, which allocated one sinogram per CPU thread. These tests check
that the sinograms are those ``torch.histogramdd`` gave: identical counts, weighted sums to float32 rounding, with and
without TOF, and the same on either device. The events are binned a chunk at a time, with chunks sized from the memory
budget: any chunk size gives, bit for bit, the sinogram of all the events binned at once with ``torch.bincount``. No
projector is needed."""
from __future__ import annotations

import math
import warnings

import pytest
import torch

import pytomography
from pytomography.io.PET import shared
from pytomography.metadata.PET import PETTOFMeta
from pytomography.utils import memory
from pytomography.io.PET.scanner import scanner_info, scanner_lut

INFO = scanner_info(min_rsector_difference=0, crystal_length=20.0, radius=120.0, rsector0_angle=90.0, transaxial_direction=-1, axial_direction=-1,
            rsectorTransNr=16, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=2, moduleTransSpacing=0.0,
            moduleAxialSpacing=17.0, submoduleTransNr=1, submoduleAxialNr=1, submoduleTransSpacing=0.0,
            submoduleAxialSpacing=0.0, crystalTransNr=4, crystalAxialNr=4, crystalTransSpacing=4.0,
            crystalAxialSpacing=4.0, NrCrystalsPerRing=64, NrRings=8)
N_DETECTORS = INFO['NrCrystalsPerRing'] * INFO['NrRings']
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def _events(n, tof_meta=None, seed=0):
    gen = torch.Generator().manual_seed(seed)
    d0 = torch.randint(0, N_DETECTORS, (n,), generator=gen)
    d1 = (d0 + torch.randint(1, N_DETECTORS, (n,), generator=gen)) % N_DETECTORS
    columns = [d0, d1] + ([torch.randint(0, tof_meta.num_bins, (n,), generator=gen)] if tof_meta else [])
    return torch.stack(columns, dim=1)


def _histogramdd_reference(detector_ids, info, weights=None, normalization=False, tof_meta=None):
    """The implementation before the change (torch.histogramdd on the CPU)."""
    lor_coordinates, sinogram_index = shared.sinogram_coordinates(info)
    tof_bins = detector_ids[:, 2].clone() if tof_meta is not None else None
    ids = detector_ids[:, :2]
    within_ring_id = (ids % info['NrCrystalsPerRing']).to(torch.long)
    ring_ids = (ids // info['NrCrystalsPerRing']).to(torch.long)
    within_ring_id, idx = within_ring_id.sort(axis=1, descending=True)
    ring_ids = ring_ids.gather(index=idx, dim=1)
    bin_edges = [torch.arange(int(info['NrCrystalsPerRing'] / 2) + 1).to(torch.float32) - 0.5,
                 torch.arange(int(info['NrCrystalsPerRing']) + 2).to(torch.float32) - 0.5,
                 torch.arange(int((info['moduleAxialNr'] * info['crystalAxialNr'])**2) + 1).to(torch.float32) - 0.5]
    def coordinates(a, b):
        return torch.concatenate([lor_coordinates[within_ring_id[:, a], within_ring_id[:, b]], sinogram_index[ring_ids[:, a], ring_ids[:, b]].unsqueeze(1)], dim=-1).to(torch.float32)
    if tof_meta is not None:
        tof_bins[idx[:, 0] == 1] = tof_meta.num_bins - 1 - tof_bins[idx[:, 0] == 1]
        data = coordinates(0, 1)
        return torch.stack([torch.histogramdd(data[tof_bins == b], bin_edges, weight=None if weights is None else weights[tof_bins == b])[0]
                            for b in range(tof_meta.num_bins)], dim=-1)
    sinogram = torch.histogramdd(coordinates(0, 1), bin_edges, weight=weights)[0]
    if normalization:
        sinogram += torch.histogramdd(coordinates(1, 0), bin_edges, weight=weights)[0]
        sinogram /= 2
    return sinogram


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tof", [False, True])
def test_counts_are_identical(tof, device):
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3) if tof else None
    events = _events(50000, tof_meta)
    sinogram = shared.listmode_to_sinogram(events.to(device), INFO, tof_meta=tof_meta)
    assert sinogram.device.type == 'cpu' and sinogram.dtype == torch.float32
    assert torch.equal(sinogram, _histogramdd_reference(events, INFO, tof_meta=tof_meta))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tof, normalization", [(False, False), (False, True), (True, False)])
def test_weighted_sums_agree(tof, normalization, device):
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3) if tof else None
    events = _events(50000, tof_meta, seed=1)
    weights = torch.rand(events.shape[0], generator=torch.Generator().manual_seed(2))
    sinogram = shared.listmode_to_sinogram(events.to(device), INFO, weights=weights, normalization=normalization, tof_meta=tof_meta)
    reference = _histogramdd_reference(events, INFO, weights=weights, normalization=normalization, tof_meta=tof_meta)
    assert sinogram.shape == reference.shape
    assert torch.allclose(sinogram, reference, rtol=1e-5, atol=1e-5)


N_TOF = 5


@pytest.fixture
def budget(monkeypatch):
    """Sets the memory budget for one test (in GB); it is None again afterwards."""
    def set_budget(gb):
        monkeypatch.setattr(pytomography, "memory_budget", None if gb is None else gb * 1e9)
    return set_budget


def _crowded_events(n, tof, seed=0):
    """Events between a few crystals on opposite sides of the scanner, so that each sinogram bin gets about a dozen and
    the order in which their weights are added shows in the float32 sums; the crystals are given in either order (and,
    with TOF, some events lie outside the TOF bins)."""
    gen = torch.Generator().manual_seed(seed)
    d0 = torch.randint(0, 12, (n,), generator=gen)
    d1 = torch.randint(N_DETECTORS // 2, N_DETECTORS // 2 + 12, (n,), generator=gen)
    swap = torch.rand(n, generator=gen) < 0.5
    columns = [torch.where(swap, d1, d0), torch.where(swap, d0, d1)]
    if tof:
        columns.append(torch.randint(-1, N_TOF + 1, (n,), generator=gen))
    return torch.stack(columns, dim=1)


def _bincount_all_at_once(detector_ids, info, weights=None, normalization=False, num_tof_bins=None):
    """All the events binned at once with torch.bincount on the CPU, one TOF bin at a time (the implementation before
    the events were binned in chunks)."""
    lor_coordinates, sinogram_index = shared.sinogram_coordinates(info)
    shape = shared._sinogram_shape(info)
    ids = detector_ids[:, :2]
    within_ring_id = (ids % info['NrCrystalsPerRing']).to(torch.long)
    ring_ids = (ids // info['NrCrystalsPerRing']).to(torch.long)
    within_ring_id, idx = within_ring_id.sort(axis=1, descending=True, stable=True)
    ring_ids = ring_ids.gather(index=idx, dim=1)
    weights = torch.ones(ids.shape[0]) if weights is None else weights.to(torch.float32)

    def bincount(a, b, keep):
        key, inside = shared._bin_keys(lor_coordinates[within_ring_id[:, a], within_ring_id[:, b]], sinogram_index[ring_ids[:, a], ring_ids[:, b]], shape)
        inside &= keep
        return torch.bincount(key[inside], weights=weights[inside], minlength=math.prod(shape)).reshape(shape)
    if num_tof_bins is not None:
        tof_bins = torch.where(idx[:, 0] == 1, num_tof_bins - 1 - detector_ids[:, 2], detector_ids[:, 2])
        return torch.stack([bincount(0, 1, tof_bins == t) for t in range(num_tof_bins)], dim=-1)
    every = torch.ones(ids.shape[0], dtype=torch.bool)
    sinogram = bincount(0, 1, every)
    if normalization:
        sinogram += bincount(1, 0, every)
        sinogram /= 2
    return sinogram


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("tof, normalization", [(False, False), (False, True), (True, False)])
def test_any_chunks_give_the_sinogram_of_all_events_at_once(tof, normalization, weighted, device):
    """Binned a chunk at a time, with chunks that leave a last partial chunk or not, one event per chunk, or one chunk
    for all, the events give bit for bit the sinogram of binning them all at once: counts and weighted sums alike (the
    events of each bin are added in their order), from the events on either device."""
    n = 2000
    events = _crowded_events(n, tof, seed=3)
    weights = torch.rand(n, generator=torch.Generator().manual_seed(4)) if weighted else None
    num_tof_bins = N_TOF if tof else None
    reference = _bincount_all_at_once(events, INFO, weights=weights, normalization=normalization, num_tof_bins=num_tof_bins)
    if weighted and not tof and not normalization:   # the sums are sensitive to the order of the events
        assert not torch.equal(reference, _bincount_all_at_once(events.flip(0), INFO, weights=weights.flip(0)))
    for events_per_chunk in (1, 7, 333, n - 1, n, 3 * n):
        sinogram = shared._listmode_to_dense_sinogram(events.to(device), INFO, weights=weights, normalization=normalization,
                                                      num_tof_bins=num_tof_bins, events_per_chunk=events_per_chunk)
        assert sinogram.device.type == 'cpu' and sinogram.dtype == torch.float32
        assert torch.equal(sinogram, reference)


def test_the_budget_sets_the_events_per_chunk(budget):
    budget(None)
    assert shared._events_per_binning_chunk() == 2**22
    budget(2)                                         # an eighth is 0.25 GB
    assert shared._events_per_binning_chunk() == int(0.25e9 // shared.BINNING_BYTES_PER_EVENT)
    budget(16)                                        # an eighth is 2 GB, above the 0.5 GB cap of a block's temporaries
    assert shared._events_per_binning_chunk() == int(memory.BLOCK_CAP_BYTES // shared.BINNING_BYTES_PER_EVENT)


@pytest.mark.parametrize("tof, normalization", [(False, False), (False, True), (True, False)])
def test_listmode_to_sinogram_bins_chunks_the_budget_sizes(tof, normalization, budget, monkeypatch):
    """With a budget whose eighth holds 777 events, listmode_to_sinogram bins 2000 events as 777 + 777 + 446, and gives
    the sinogram of binning them all at once."""
    n = 2000
    events = _crowded_events(n, tof, seed=5)
    weights = torch.rand(n, generator=torch.Generator().manual_seed(6))
    tof_meta = PETTOFMeta(N_TOF, 300.0, 60.0, n_sigmas=3) if tof else None
    reference = _bincount_all_at_once(events, INFO, weights=weights, normalization=normalization, num_tof_bins=N_TOF if tof else None)
    budget((777 + 0.5) * shared.BINNING_BYTES_PER_EVENT * 8 / 1e9)
    assert shared._events_per_binning_chunk() == 777
    chunks = []
    add_events = shared._add_events

    def counting_add_events(sinograms, ids, *args):
        chunks.append(ids.shape[0])
        add_events(sinograms, ids, *args)
    monkeypatch.setattr(shared, "_add_events", counting_add_events)
    # lazy=False: a sinogram over a quarter of this small budget would otherwise be lazy
    sinogram = shared.listmode_to_sinogram(events, INFO, weights=weights, normalization=normalization, tof_meta=tof_meta, lazy=False)
    assert chunks == [777, 777, 446]
    assert torch.equal(sinogram, reference)


def test_sinogram_to_spatial_raises_no_numpy_deprecation_warning():
    """Copying the crystals' positions from a tensor into the NumPy lookup table made NumPy 2 warn that Tensor.__array__
    takes no copy argument, twice per crystal pair (8,356 warnings in the test suite)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        detector_coordinates, ring_coordinates = shared.sinogram_to_spatial.__wrapped__(INFO)   # not the cached tables
    lut = scanner_lut(INFO)
    shape = shared._sinogram_shape(INFO)
    assert detector_coordinates.shape == (shape[0], shape[1], 2, 2) and detector_coordinates.dtype == torch.float32
    assert ring_coordinates.shape == (INFO['NrRings']**2, 2)
    # every crystal of the table is one of the scanner's (its x and y), and every ring position one of its z
    rows = {tuple(r) for r in lut[:, :2].tolist()}
    filled = detector_coordinates.reshape(-1, 2)[detector_coordinates.reshape(-1, 2).abs().sum(dim=1) > 0]
    assert filled.shape[0] > 0 and all(tuple(r) in rows for r in filled.tolist())
    assert set(ring_coordinates.flatten().tolist()) <= set(lut[:, 2].tolist())
