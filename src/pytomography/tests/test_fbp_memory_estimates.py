"""Memory estimates of filtered back projection (FilteredBackProjection.estimate_memory) for the helical, cone-beam and
SPECT system matrices, and of ordered-subset reconstructions with the CT system matrices (estimate_memory), where each
subset keeps an H_m^T 1 image, so that fewer subsets can take less memory as well as more. The helical estimate counts
the chunk of rebinned views the reconstruction holds, under any memory budget. On a GPU, the estimates bound the
peaks the reconstructions reach."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pytomography

parallelproj_core = pytest.importorskip("parallelproj_core", reason="CT system matrices require parallelproj 2")
from pytomography.algorithms import FilteredBackProjection
from pytomography.metadata import ObjectMeta
from pytomography.projectors import SystemMatrix
from pytomography.projectors.CT import CTConeBeamFlatPanelSystemMatrix, CTGen3SystemMatrix, _wfbp
from pytomography.tests.test_ct_fbp import OBJECT_META, _conebeam, _helix, _line_integrals
from pytomography.tests.test_spect_fbp import _system_matrix as _spect_system_matrix
from pytomography.utils import PeakMemory
from pytomography.utils.memory import MemoryEstimate, memory_budget_set

# PyTomography's device as conftest.py set it: PYTOMOGRAPHY_TEST_DEVICE=cpu keeps it on the CPU on a machine with a GPU
ON_GPU = torch.cuda.is_available() and torch.device(pytomography.device).type == 'cuda'


def _parts(estimate: MemoryEstimate) -> dict:
    return {p.name: p for p in estimate.parts}


def _gpu(estimate: MemoryEstimate) -> float:
    return sum(p.gpu_bytes for p in estimate.parts)


def test_the_helical_fbp_estimate_counts_the_chunk_it_holds():
    meta = _helix()
    sm = CTGen3SystemMatrix(OBJECT_META, meta)
    proj = _line_integrals(meta)
    whole = _parts(FilteredBackProjection(proj, sm).estimate_memory())
    assert whole['projections'].ram_bytes == proj.numel() * 4
    assert whole['image'].ram_bytes + whole['image'].gpu_bytes == np.prod(OBJECT_META.shape) * 4
    stats = {}
    with memory_budget_set(0.004):                               # chunks of at most 0.5 MB of rebinned views
        small = _parts(FilteredBackProjection(proj, sm).estimate_memory())
        FilteredBackProjection(proj, sm, stats=stats)()
    chunk = small['rebinned views of one chunk'].ram_bytes
    assert chunk <= 0.004e9 / 8 < whole['rebinned views of one chunk'].ram_bytes
    # the chunks the reconstruction went through are those of the estimate
    geo = _wfbp.group_geometry(meta, _wfbp.view_groups(meta)[0])
    theta, t, _ = _wfbp.parallel_grid(geo, float(meta.source_rhos.double().mean()) * abs(float(meta.col_det_spacing)))
    assert stats['groups'][0]['chunks'] == -(-len(theta) // (chunk // (len(t) * meta.shape[1] * 4)))


def test_a_lower_gpu_budget_is_offered_when_it_saves_memory():
    meta = _helix()
    proj = _line_integrals(meta)
    estimate = FilteredBackProjection(proj, CTGen3SystemMatrix(OBJECT_META, meta), gpu_budget=20e6).estimate_memory()
    alternatives = dict(estimate.alternatives)
    assert 'gpu_budget=0.01e9' in alternatives
    total = lambda e: sum(p.ram_bytes + p.gpu_bytes for p in e.parts)
    assert total(alternatives['gpu_budget=0.01e9']) < total(estimate)


def test_ct_subsets_can_cost_memory_either_way():
    meta = _helix()
    # a large image: the H_m^T 1 of each subset outweighs the smaller subsets, so fewer subsets take less memory
    big = CTGen3SystemMatrix(ObjectMeta(dr=(1.0, 1.0, 1.0), shape=(256, 256, 64)), meta)
    estimate = big.estimate_memory(n_subsets=8)
    assert _parts(estimate)['H_m^T 1 of 8 subsets'].ram_bytes == 8 * 256 * 256 * 64 * 4
    labels = [label for label, _ in estimate.alternatives]
    assert '4 subsets' in labels                                 # (on a GPU, 16 subsets may be offered too: less GPU)
    assert big.fewest_subsets(estimate.ram_gb) == 1
    # many views and a small image: more subsets take less memory
    small = CTGen3SystemMatrix(ObjectMeta(dr=(4.0, 4.0, 4.0), shape=(16, 16, 4)), _helix(rotations=24, n_cols=200, n_rows=32))
    labels = [label for label, _ in small.estimate_memory(n_subsets=8).alternatives]
    assert '16 subsets' in labels and '4 subsets' not in labels
    # cone beam: the H_m^T 1 stay on the device
    cone = CTConeBeamFlatPanelSystemMatrix(ObjectMeta(dr=(1.0, 1.0, 1.0), shape=(64, 64, 32)), _conebeam(120))
    part = _parts(cone.estimate_memory(n_subsets=4))['H_m^T 1 of 4 subsets']
    assert part.ram_bytes + part.gpu_bytes == 4 * 64 * 64 * 32 * 4
    assert (part.gpu_bytes > 0) == (torch.device(pytomography.device).type == 'cuda')


def test_every_fbp_geometry_estimates_its_memory():
    cone_meta = _conebeam(120)
    cone = CTConeBeamFlatPanelSystemMatrix(ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(32, 32, 24)), cone_meta)
    spect = _spect_system_matrix(np.linspace(0, 360, 60, endpoint=False))
    for sm, proj in ((cone, torch.zeros(cone_meta.N_angles, *cone_meta.shape)), (spect, torch.zeros(spect.proj_meta.shape))):
        estimate = FilteredBackProjection(proj, sm).estimate_memory()
        parts = _parts(estimate)
        assert parts['projections'].ram_bytes == proj.numel() * 4
        assert estimate.ram_gb > 0 and type(sm).__name__ in estimate.title
        str(estimate).encode('cp1252')

    class _NoFBP(SystemMatrix):
        def __init__(self):
            self.object_meta, self.proj_meta = OBJECT_META, None
        def forward(self, object, **kwargs): ...
        def backward(self, proj, **kwargs): ...
        def get_subset_splits(self, n_subsets): ...

    with pytest.raises(NotImplementedError):
        FilteredBackProjection(torch.zeros(3), _NoFBP()).estimate_memory()


def _gpu_peak(run) -> float:
    """Peak GPU memory PyTorch allocates while ``run`` runs, above the level it starts at."""
    with PeakMemory('cuda') as m:
        run()
    return m.peak


@pytest.mark.skipif(not ON_GPU, reason="measures CUDA memory, and PyTomography's device is not a GPU")
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_the_helical_fbp_estimate_bounds_its_gpu_peak(backend):
    from pytomography.projectors.CT import _wfbp_cuda
    if backend == 'cuda' and not _wfbp_cuda.available('cuda'):
        pytest.skip('the fused kernel needs CuPy')
    meta = _helix(ffs=True)
    proj = _line_integrals(meta)
    fbp = FilteredBackProjection(proj, CTGen3SystemMatrix(ObjectMeta(dr=(1.0, 1.0, 2.0), shape=(96, 96, 14)), meta),
                                 gpu_budget=12e6, backend=backend)
    estimate = _gpu(fbp.estimate_memory())
    assert 0.4 * estimate <= _gpu_peak(fbp) <= estimate


@pytest.mark.skipif(not ON_GPU, reason="measures CUDA memory, and PyTomography's device is not a GPU")
@pytest.mark.parametrize("which", ["cone beam", "SPECT"])
def test_the_fbp_estimate_bounds_its_gpu_peak(which):
    if which == "cone beam":                                   # small, but large enough that kilobytes don't count
        meta = _conebeam(n_views=120, n_cols=256)
        sm = CTConeBeamFlatPanelSystemMatrix(ObjectMeta(dr=(0.5, 0.5, 0.5), shape=(128, 128, 64)), meta)
        proj = torch.rand(meta.N_angles, *meta.shape)
    else:                                                      # a typical SPECT grid, 128 x 128 x 32
        from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
        from pytomography.projectors.SPECT import SPECTSystemMatrix
        sm = SPECTSystemMatrix([], [], SPECTObjectMeta([0.4] * 3, (128, 128, 32)),
                               SPECTProjMeta((128, 32), [0.4, 0.4], list(np.linspace(0, 360, 60, endpoint=False))))
        proj = torch.rand(sm.proj_meta.shape)
    fbp = FilteredBackProjection(proj, sm)
    estimate = _gpu(fbp.estimate_memory())
    assert 0.4 * estimate <= _gpu_peak(fbp) <= estimate


@pytest.mark.skipif(not ON_GPU, reason="measures CUDA memory, and PyTomography's device is not a GPU")
@pytest.mark.parametrize("which", ["helical", "cone beam"])
def test_the_ordered_subsets_estimate_bounds_its_gpu_peak(which):
    from pytomography.algorithms import SART
    if which == "helical":
        meta = _helix(rotations=2)
        sm = CTGen3SystemMatrix(ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(48, 48, 14)), meta)
    else:
        meta = _conebeam(120)
        sm = CTConeBeamFlatPanelSystemMatrix(ObjectMeta(dr=(1.0, 1.0, 1.0), shape=(64, 64, 32)), meta)
    proj = torch.rand(meta.N_angles, *meta.shape, device=sm.device)
    estimate = _gpu(sm.estimate_memory(n_subsets=4, held={'projections': proj}))
    peak = _gpu_peak(lambda: SART(sm, proj)(n_iters=1, n_subsets=4)) + float(proj.numel() * 4 if proj.is_cuda else 0)
    assert 0.4 * estimate <= peak <= estimate
