"""Memory estimates: MemoryEstimate adds up its parts (on Windows GPU memory counts as RAM too), memory_estimate adds the
same overheads to an estimate and to its alternatives, SystemMatrix.estimate_memory evaluates the "To use less"
settings and the arrays a reconstruction holds, fewest_subsets finds the fewest subsets that fit, and a LazySinogram
reports what it keeps rather than its full size."""
from __future__ import annotations

import pytest
import torch

import pytomography
from pytomography.io.PET import shared
from pytomography.metadata import ObjectMeta, ProjMeta
from pytomography.projectors import SystemMatrix
from pytomography.utils import memory
from pytomography.utils.memory import MemoryEstimate, MemoryPart, memory_budget_set, memory_estimate

INFO = dict(min_rsector_difference=0, crystal_length=20.0, radius=120.0, firstCrystalAxis=0,
            rsectorTransNr=16, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=2, moduleTransSpacing=0.0, moduleAxialSpacing=17.0,
            submoduleTransNr=1, submoduleAxialNr=1, submoduleTransSpacing=0.0, submoduleAxialSpacing=0.0,
            crystalTransNr=4, crystalAxialNr=4, crystalTransSpacing=4.0, crystalAxialSpacing=4.0,
            NrCrystalsPerRing=64, NrRings=8)


@pytest.fixture(params=[False, True], ids=["linux", "windows"])
def gpu_as_ram(request, monkeypatch):
    monkeypatch.setattr(memory, "counts_gpu_as_ram", lambda: request.param)
    return request.param


class _ToySystemMatrix(SystemMatrix):
    """Projections of 1 GB in subsets; projector chunks of 0.4 GB on the GPU divided by N_splits."""
    def __init__(self):
        self.object_meta = ObjectMeta(dr=(1, 1, 1), shape=(8, 8, 8))
        self.proj_meta = None
        self.N_splits = 4

    def _memory_parts(self, n_subsets, N_splits):
        splits = self.N_splits if N_splits is None else N_splits
        return [MemoryPart('sensitivity', ram_bytes=0.5e9),
                MemoryPart('data and expected counts', ram_bytes=2 * 1e9 / n_subsets, scope='subset'),
                MemoryPart('LOR coordinates', gpu_bytes=0.4e9 / splits, scope='chunk')]

    def forward(self, object, **kwargs): ...
    def backward(self, proj, **kwargs): ...
    def get_subset_splits(self, n_subsets): ...


def test_estimate_adds_up_its_parts(gpu_as_ram):
    estimate = MemoryEstimate('test', [MemoryPart('a', ram_bytes=1e9), MemoryPart('b', ram_bytes=0.5e9, gpu_bytes=2e9, scope='subset')])
    assert estimate.gpu_gb == pytest.approx(2.0)
    assert estimate.ram_gb == pytest.approx(3.5 if gpu_as_ram else 1.5)
    text = str(estimate)
    assert text.startswith('test\nPeak ~') and 'held for the whole run' in text and 'one subset at a time' in text
    text.encode('cp1252')                                     # printable on a Windows console or into a file
    with pytest.raises(ValueError):
        MemoryPart('c', scope='always')


def test_memory_estimate_adds_the_overheads_to_the_alternatives_too(gpu_as_ram):
    parts = [MemoryPart('data', ram_bytes=4e9)]
    estimate = memory_estimate('test', parts, alternatives=[('fewer', [MemoryPart('data', ram_bytes=2e9)])])
    overhead = memory.FIXED_OVERHEAD_BYTES / 1e9
    allocator = memory.WINDOWS_ALLOCATOR_FRACTION if gpu_as_ram else 0
    assert estimate.ram_gb == pytest.approx(4 * (1 + allocator) + overhead)
    (label, alternative), = estimate.alternatives
    assert label == 'fewer' and alternative.ram_gb == pytest.approx(2 * (1 + allocator) + overhead)
    assert 'To use less: fewer ~' in str(estimate)


def test_memory_budget_set_restores_the_budget():
    pytomography.set_memory_budget(25)
    try:
        with memory_budget_set(8):
            assert pytomography.memory_budget == 8e9
        with pytest.raises(RuntimeError):
            with memory_budget_set(4):
                raise RuntimeError
        assert pytomography.memory_budget == 25e9
    finally:
        pytomography.set_memory_budget(None)


def test_system_matrix_estimate_and_fewest_subsets(gpu_as_ram):
    sm = _ToySystemMatrix()
    held = torch.zeros(250_000_000 // 4)                      # 0.25 GB on the host
    estimate = sm.estimate_memory(n_subsets=4, held={'additive term': held})
    names = [p.name for p in estimate.parts]
    assert 'additive term' in names and 'sensitivity' in names and 'Python, PyTorch and CUDA' in names
    labels = [label for label, _ in estimate.alternatives]
    assert labels == ['8 subsets', 'N_splits 8']
    by_label = dict(estimate.alternatives)
    assert by_label['8 subsets'].ram_gb < estimate.ram_gb and by_label['N_splits 8'].gpu_gb < estimate.gpu_gb
    target = sm.estimate_memory(n_subsets=10, held=[held]).ram_gb
    assert sm.fewest_subsets(target, held=[held]) == 10
    assert sm.fewest_subsets(0.1) is None
    # memory lowest at a few subsets (a normalisation image per subset, as in CT: 4/n + 0.2 n GB, least at 4 and 5):
    # still found, though 1024 subsets do not fit
    sm._memory_parts = lambda n, s: [MemoryPart('subset arrays', ram_bytes=4e9 / n, scope='subset'), MemoryPart('normalisation images', ram_bytes=0.2e9 * n)]
    assert sm.fewest_subsets(sm.estimate_memory(4).ram_gb) == 4
    assert sm.fewest_subsets(sm.estimate_memory(3).ram_gb) == 3
    with pytest.raises(NotImplementedError):
        SystemMatrix._memory_parts(sm, 1, None)


def test_lazy_sinogram_reports_what_it_keeps():
    gen = torch.Generator().manual_seed(0)
    n_detectors = INFO['NrCrystalsPerRing'] * INFO['NrRings']
    d0 = torch.randint(0, n_detectors, (5000,), generator=gen)
    d1 = (d0 + n_detectors // 2 + torch.randint(-20, 20, (5000,), generator=gen)) % n_detectors
    lazy = shared.listmode_to_sinogram(torch.stack([d0, d1], dim=1), INFO, lazy=True)
    assert 0 < lazy.memory_bytes < 5000 * 8 + 4 * (lazy.shape[0] + 1) * 8      # positions (4 bytes each) and offsets
    assert memory.nbytes(lazy) == (lazy.memory_bytes, 0.0)
    randoms = torch.rand(lazy.shape)
    combined = (randoms + lazy) / 2
    assert combined.memory_bytes == lazy.memory_bytes + randoms.untyped_storage().nbytes()
