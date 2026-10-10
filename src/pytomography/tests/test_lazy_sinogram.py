"""A TOF sinogram of a clinical scanner is too large to hold several of (34.6 GB for the Siemens Biograph mMR with 21 TOF
bins), and a reconstruction reads one subset of angles at a time. A LazySinogram computes the angles it is asked for
when it is asked for them: list mode events binned by angle, the interpolated scatter estimate, and arithmetic of these
with dense sinograms (the additive term). These tests check that every lazy path gives exactly the dense result: the
sinogram itself, the events looked up in it, the scatter estimate and its scaling, and OSEM reconstructions."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

try:
    import parallelproj_core  # noqa: F401
except ImportError:                                   # only the tests marked needs_kernels call the projector kernels
    sys.modules["parallelproj_core"] = types.ModuleType("parallelproj_core")

import pytomography
from pytomography.algorithms import OSEM
from pytomography.io.PET import shared
from pytomography.io.PET.shared import LazySinogram
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETSinogramPolygonProjMeta, PETTOFMeta
from pytomography.projectors.PET import PETLMSystemMatrix, PETSinogramSystemMatrix
from pytomography.utils import sss

HAVE_KERNELS = hasattr(sys.modules["parallelproj_core"], "joseph3d_fwd")
if not HAVE_KERNELS:
    del sys.modules["parallelproj_core"]              # the modules above keep the stub
needs_kernels = pytest.mark.skipif(not HAVE_KERNELS, reason="requires parallelproj 2 (parallelproj_core)")

DEV = pytomography.device
INFO = dict(min_rsector_difference=0, crystal_length=20.0, radius=120.0, firstCrystalAxis=0,
            rsectorTransNr=16, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=2, moduleTransSpacing=0.0, moduleAxialSpacing=17.0,
            submoduleTransNr=1, submoduleAxialNr=1, submoduleTransSpacing=0.0, submoduleAxialSpacing=0.0,
            crystalTransNr=4, crystalAxialNr=4, crystalTransSpacing=4.0, crystalAxialSpacing=4.0,
            NrCrystalsPerRing=64, NrRings=8)
N_DETECTORS = INFO['NrCrystalsPerRing'] * INFO['NrRings']
SHAPE = (INFO['NrCrystalsPerRing'] // 2, INFO['NrCrystalsPerRing'] + 1, (INFO['moduleAxialNr'] * INFO['crystalAxialNr'])**2)
N_TOF = 5
OBJECT_META = ObjectMeta(dr=(4, 4, 4), shape=(32, 32, 6))      # within the scanner's axial field of view (+-14.5 mm)


def _tof_meta(tof: bool):
    return PETTOFMeta(N_TOF, 300.0, 60.0, n_sigmas=3) if tof else None


def _events(n: int, tof: bool, seed: int = 0, outside_tof: bool = False) -> torch.Tensor:
    """Random coincidences between roughly opposite crystals (and a TOF bin; with ``outside_tof``, some lie outside the TOF bins)."""
    gen = torch.Generator().manual_seed(seed)
    d0 = torch.randint(0, N_DETECTORS, (n,), generator=gen)
    d1 = (d0 + N_DETECTORS // 2 + torch.randint(-24, 24, (n,), generator=gen)) % N_DETECTORS
    columns = [d0, d1]
    if tof:
        low, high = (-1, N_TOF + 1) if outside_tof else (0, N_TOF)
        columns.append(torch.randint(low, high, (n,), generator=gen))
    return torch.stack(columns, dim=1)


def _phantom():
    """Water cylinder of radius 40 mm with uniform activity and a hot sphere: (activity, attenuation) on the device."""
    x, y, z = torch.meshgrid(*[(torch.arange(n) - n / 2 + 0.5) * d for n, d in zip(OBJECT_META.shape, OBJECT_META.dr)], indexing="ij")
    cylinder = (x**2 + y**2 < 40**2).float()
    sphere = (((x - 15)**2 + (y + 10)**2 + z**2) < 12**2).float()
    return (cylinder + 3 * sphere).to(DEV), (0.0096 * cylinder).to(DEV)


@pytest.fixture
def small_chunks(monkeypatch):
    """Compute lazy sinograms three angles at a time, so that reading many angles goes through several groups."""
    monkeypatch.setattr(LazySinogram, "chunk_bytes", 3 * 4 * int(np.prod(SHAPE[1:])) * N_TOF)


@pytest.mark.parametrize("tof", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_lazy_listmode_sinogram_equals_dense(tof, weighted, small_chunks):
    events = _events(30000, tof, outside_tof=True)
    weights = torch.rand(events.shape[0], generator=torch.Generator().manual_seed(1)) if weighted else None
    dense = shared.listmode_to_sinogram(events, INFO, weights=weights, tof_meta=_tof_meta(tof))
    lazy = shared.listmode_to_sinogram(events, INFO, weights=weights, tof_meta=_tof_meta(tof), lazy=True)
    assert isinstance(lazy, LazySinogram) and lazy.shape == dense.shape
    assert torch.equal(lazy.to_dense(), dense)          # counts are exact, and weights are added in the same order
    angles = torch.tensor([7, 0, 31, 7, 12, 3, 30])    # any order, repeats allowed
    assert torch.equal(lazy[angles], dense[angles])
    assert torch.equal(lazy[[5, 2]], dense[[5, 2]])
    assert torch.equal(lazy[-1], dense[-1])
    assert torch.equal(lazy[4:20:3], dense[4:20:3])
    mask = torch.zeros(SHAPE[0], dtype=torch.bool)
    mask[[1, 8, 9]] = True
    assert torch.equal(lazy[mask], dense[mask])
    if tof:
        assert torch.equal(lazy[:, :, :10, 2], dense[:, :, :10, 2])
        assert torch.equal(lazy[:, 3:40, :, [0, 4]], dense[:, 3:40, :, [0, 4]])
        assert torch.equal(lazy[:, :, :10, [1, 3]].sum(dim=(0, 2)), dense[:, :, :10, [1, 3]].sum(dim=(0, 2)))


def test_lazy_indexing_errors():
    lazy = shared.listmode_to_sinogram(_events(100, True), INFO, tof_meta=_tof_meta(True), lazy=True)
    with pytest.raises(IndexError):
        lazy[SHAPE[0]]
    with pytest.raises(IndexError):
        lazy[..., 0]
    with pytest.raises(IndexError):
        lazy[[1, 2], :, :, [0, 1]]       # two array indices: numpy would pair them up
    with pytest.raises(NotImplementedError):
        shared.listmode_to_sinogram(_events(100, False), INFO, normalization=True, lazy=True)


def test_lazy_arithmetic_equals_dense(small_chunks):
    gen = torch.Generator().manual_seed(3)
    lazy = shared.listmode_to_sinogram(_events(20000, True), INFO, tof_meta=_tof_meta(True), lazy=True)
    dense = lazy.to_dense()
    randoms = torch.rand(SHAPE, generator=gen)
    sensitivity = torch.rand(SHAPE, generator=gen) + 0.5
    # the additive term of the TOF tutorials
    additive = (randoms.unsqueeze(-1) + lazy) / sensitivity.unsqueeze(-1)
    assert isinstance(additive, LazySinogram) and additive.shape == dense.shape
    assert torch.equal(additive.to_dense(), (randoms.unsqueeze(-1) + dense) / sensitivity.unsqueeze(-1))
    assert torch.equal((2.5 * lazy - lazy / 4).to_dense(), 2.5 * dense - dense / 4)
    assert torch.equal((1 - lazy)[[3, 1]], (1 - dense)[[3, 1]])
    assert torch.equal((lazy * lazy)[5], (dense * dense)[5])
    with pytest.raises(ValueError):
        lazy + randoms                   # the TOF dimension must be added explicitly
    with pytest.raises(ValueError):
        lazy + torch.rand(N_TOF)         # no angle dimension


@pytest.mark.parametrize("tof", [False, True])
def test_events_read_from_lazy_sinogram_equal_dense(tof, small_chunks):
    sinogram = shared.listmode_to_sinogram(_events(20000, tof, seed=4), INFO, tof_meta=_tof_meta(tof))
    sinogram = sinogram + torch.rand(sinogram.shape, generator=torch.Generator().manual_seed(5))
    lazy = LazySinogram(lambda angles: sinogram[angles], sinogram.shape)
    events = _events(7000, tof, seed=6)
    expected = shared.sinogram_to_listmode(events, sinogram, INFO)
    assert torch.equal(shared.sinogram_to_listmode(events, lazy, INFO), expected)


@pytest.mark.parametrize("normalization", [False, True])
def test_all_pairs_sinogram_in_blocks_equals_all_at_once(normalization):
    n_pairs = N_DETECTORS * (N_DETECTORS - 1) // 2
    weights = torch.rand(n_pairs, generator=torch.Generator().manual_seed(7))
    pairs = torch.combinations(torch.arange(N_DETECTORS).to(torch.int32), 2)
    expected = shared.listmode_to_sinogram(pairs, INFO, weights=weights, normalization=normalization)
    for pairs_per_chunk in (5000, 2**24):
        assert torch.equal(shared.all_pairs_to_sinogram(weights, INFO, normalization=normalization, pairs_per_chunk=pairs_per_chunk), expected)
    with pytest.raises(ValueError):
        shared.all_pairs_to_sinogram(weights[:-1], INFO)


@pytest.mark.parametrize("tof", [False, True])
def test_lazy_scatter_interpolation_equals_dense(tof, small_chunks):
    gen = torch.Generator().manual_seed(8)
    tof_meta = _tof_meta(tof)
    proj_meta = PETSinogramPolygonProjMeta(INFO, tof_meta)
    idx_intraring, idx_ring, detector_ids = sss.get_sample_detector_ids(proj_meta, 4, 4)
    N = detector_ids.shape[0]
    sparse = sss.SparseSinogram(detector_ids, torch.rand((N_TOF, N) if tof else (N,), generator=gen).to(DEV), INFO, tof_meta=tof_meta)
    tof_bins = range(N_TOF) if tof else None
    dense = sss.interpolate_sparse_sinogram(sparse, proj_meta, idx_intraring, idx_ring, tof_bins=tof_bins)
    lazy = sss.interpolate_sparse_sinogram(sparse, proj_meta, idx_intraring, idx_ring, tof_bins=tof_bins, lazy=True)
    assert isinstance(lazy, LazySinogram) and lazy.shape == dense.shape
    assert torch.equal(lazy.to_dense(), dense)
    angles = torch.tensor([30, 2, 17])
    assert torch.equal(lazy[angles], dense[angles])


@pytest.mark.parametrize("tof", [False, True])
def test_scatter_interpolation_does_not_depend_on_the_angles_per_call(tof, monkeypatch):
    """The interpolation over the ring pairs handles a group of angles, with all their TOF bins, per call, as many as fit
    in the GPU budget; one angle per call gives the same values."""
    gen = torch.Generator().manual_seed(9)
    tof_meta = _tof_meta(tof)
    proj_meta = PETSinogramPolygonProjMeta(INFO, tof_meta)
    idx_intraring, idx_ring, detector_ids = sss.get_sample_detector_ids(proj_meta, 4, 4)
    N = detector_ids.shape[0]
    sparse = sss.SparseSinogram(detector_ids, torch.rand((N_TOF, N) if tof else (N,), generator=gen).to(DEV), INFO, tof_meta=tof_meta)
    tof_bins = range(N_TOF) if tof else None
    expected = sss.interpolate_sparse_sinogram(sparse, proj_meta, idx_intraring, idx_ring, tof_bins=tof_bins)
    monkeypatch.setattr(sss, "gpu_budget", lambda *args, **kwargs: 1.0)
    assert torch.equal(sss.interpolate_sparse_sinogram(sparse, proj_meta, idx_intraring, idx_ring, tof_bins=tof_bins), expected)


def test_likelihood_without_additive_term_holds_no_zeros():
    """It used to hold a tensor of zeros the size of the projections: 34.6 GB for a TOF sinogram of the mMR."""
    sm = PETSinogramSystemMatrix(OBJECT_META, PETSinogramPolygonProjMeta(INFO, _tof_meta(True)), device='cpu')
    lazy = shared.listmode_to_sinogram(_events(1000, True), INFO, tof_meta=_tof_meta(True), lazy=True)
    likelihood = PoissonLogLikelihood(sm, lazy)
    assert likelihood.additive_term is None and not likelihood.exists_additive_term
    assert torch.equal(likelihood._get_projection_subset(lazy, None), lazy.to_dense())
    likelihood = PoissonLogLikelihood(sm, lazy, additive_term=0.5 * lazy)
    assert isinstance(likelihood.additive_term, LazySinogram) and likelihood.exists_additive_term


def _close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5) -> bool:
    """Equal up to the order in which the projector adds up back projections: parallelproj adds each line of response
    into the image with atomic adds, so two runs of the same back projection differ in the last bits."""
    return torch.allclose(a, b, rtol=rtol, atol=rtol * float(b.abs().max()))


def _sinogram_system_matrix(tof: bool, attenuation: torch.Tensor) -> PETSinogramSystemMatrix:
    return PETSinogramSystemMatrix(OBJECT_META, PETSinogramPolygonProjMeta(INFO, _tof_meta(tof)), attenuation_map=attenuation, N_splits=3, device='cpu')


def _what_the_projector_gets(sm: PETSinogramSystemMatrix, likelihood: PoissonLogLikelihood, obj: torch.Tensor, n_subsets: int) -> tuple:
    """The sinograms each subset's gradient back projects, and the expected projections, at a fixed object."""
    likelihood._set_n_subsets(n_subsets)
    back_projected, original = [], sm.backward
    def backward(proj, *args, **kwargs):
        back_projected.append(proj.clone())
        return original(proj, *args, **kwargs)
    sm.backward = backward
    try:
        predicted = []
        for k in range(n_subsets):
            likelihood.compute_gradient(obj, k)
            predicted.append(likelihood.projections_predicted.clone())
    finally:
        del sm.backward
    return back_projected, predicted


@needs_kernels
@pytest.mark.parametrize("tof", [False, True])
def test_osem_with_lazy_data_and_additive_term_equals_dense(tof, small_chunks):
    """The reconstruction of the TOF sinogram tutorial, with the data and the additive term computed one subset at a
    time: at a fixed object every subset's gradient back projects exactly the same sinogram, and whole reconstructions
    agree to the projector's rounding (two dense runs differ by about 2e-6 of the maximum; the TOF reconstruction
    without an additive term divides by 1e-11 wherever the expected counts are 0, which spreads that a little more)."""
    activity, attenuation = _phantom()
    sm = _sinogram_system_matrix(tof, attenuation)
    events = _events(60000, tof, seed=9)
    data_lazy = shared.listmode_to_sinogram(events, INFO, tof_meta=_tof_meta(tof), lazy=True)
    data = data_lazy.to_dense()
    randoms = torch.rand(SHAPE, generator=torch.Generator().manual_seed(10)) * 0.01
    sensitivity = sm._compute_sensitivity_sinogram().cpu()
    randoms = randoms.unsqueeze(-1) if tof else randoms
    scatter_lazy = 0.1 * data_lazy + 0.001
    cases = {'dense': (data, (randoms + scatter_lazy.to_dense()) / sensitivity), 'lazy': (data_lazy, (randoms + scatter_lazy) / sensitivity),
             'dense, no additive term': (data, None), 'lazy, no additive term': (data_lazy, None)}
    gets, reconstructions = {}, {}
    for name, (projections, additive) in cases.items():
        gets[name] = _what_the_projector_gets(sm, PoissonLogLikelihood(sm, projections, additive_term=additive), activity, 4)
        reconstructions[name] = OSEM(PoissonLogLikelihood(sm, projections, additive_term=additive))(n_iters=2, n_subsets=4)
    for lazy, dense in (('lazy', 'dense'), ('lazy, no additive term', 'dense, no additive term')):
        for got, expected in zip(gets[lazy], gets[dense]):
            assert len(got) == len(expected) == 4 and all(torch.equal(a, b) for a, b in zip(got, expected))
        assert _close(reconstructions[lazy], reconstructions[dense], rtol=1e-4)


@needs_kernels
@pytest.mark.skipif(DEV == 'cpu' or not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("tof", [False, True])
def test_osem_with_projections_on_the_gpu(tof, small_chunks):
    """With a system matrix that puts its projections on the GPU (device='cuda'), each subset of the data and of the
    additive term is computed there (lazy) or copied there (dense): the reconstruction is that of the system matrix
    that puts them on the host, up to the projector's rounding."""
    activity, attenuation = _phantom()
    events = _events(60000, tof, seed=16)
    data_lazy = shared.listmode_to_sinogram(events, INFO, tof_meta=_tof_meta(tof), lazy=True)
    randoms = torch.rand(SHAPE, generator=torch.Generator().manual_seed(17)) * 0.01
    randoms = randoms.unsqueeze(-1) if tof else randoms
    results = {}
    for device in ('cpu', DEV):
        sm = PETSinogramSystemMatrix(OBJECT_META, PETSinogramPolygonProjMeta(INFO, _tof_meta(tof)), attenuation_map=attenuation, N_splits=3, device=device)
        sensitivity = sm._compute_sensitivity_sinogram().cpu()
        for name, (projections, additive) in {'lazy': (data_lazy, (randoms + 0.1 * data_lazy + 0.001) / sensitivity),
                                              'dense': (data_lazy.to_dense(), (randoms + 0.1 * data_lazy.to_dense() + 0.001) / sensitivity)}.items():
            likelihood = PoissonLogLikelihood(sm, projections, additive_term=additive)
            results[device, name] = OSEM(likelihood)(n_iters=2, n_subsets=4).cpu()
            assert likelihood.projections_predicted.device.type == torch.device(device).type
    for name in ('lazy', 'dense'):
        assert _close(results[DEV, name], results['cpu', name], rtol=1e-4)


@needs_kernels
@pytest.mark.parametrize("tof", [False, True])
def test_osem_result_unchanged_by_in_place_additive_term(tof):
    """The forward projection and the additive term are added in place now; the reconstruction is the same as adding them into a new tensor."""
    activity, attenuation = _phantom()
    sm = _sinogram_system_matrix(tof, attenuation)
    data = shared.listmode_to_sinogram(_events(60000, tof, seed=11), INFO, tof_meta=_tof_meta(tof))
    additive = torch.rand(data.shape, generator=torch.Generator().manual_seed(12)) * 0.05
    likelihood = PoissonLogLikelihood(sm, data, additive_term=additive)
    sm.set_n_subsets(4)
    likelihood._set_n_subsets(4)
    obj = torch.ones(OBJECT_META.shape, device=DEV)
    expected_FP = sm.forward(obj, 1) + sm.get_projection_subset(additive, 1)
    expected = sm.backward(sm.get_projection_subset(data, 1) / (expected_FP + pytomography.delta), 1) - likelihood._get_normBP(1)
    assert _close(likelihood.compute_gradient(obj, 1), expected)
    assert torch.equal(likelihood.projections_predicted, expected_FP)      # forward projections add up in a fixed order
    assert torch.equal(additive, torch.rand(data.shape, generator=torch.Generator().manual_seed(12)) * 0.05)   # not modified


@needs_kernels
@pytest.mark.parametrize("tof", [False, True])
def test_lazy_scatter_estimate_equals_dense(tof, small_chunks):
    """The scatter estimate cell of the sinogram tutorials, with the data and the estimate lazy: same estimate."""
    activity, attenuation = _phantom()
    sm = _sinogram_system_matrix(tof, attenuation)
    proj_meta = sm.proj_meta
    data_lazy = shared.listmode_to_sinogram(_events(60000, tof, seed=13), INFO, tof_meta=_tof_meta(tof), lazy=True)
    randoms = torch.rand(SHAPE, generator=torch.Generator().manual_seed(14)) * 0.01
    estimates = []
    for data, lazy in ((data_lazy.to_dense(), False), (data_lazy, True)):
        torch.manual_seed(0)
        estimates.append(sss.get_sss_scatter_estimate(OBJECT_META, proj_meta, activity, attenuation, sm, proj_data=data,
                                                      image_stepsize=2, sinogram_interring_stepsize=4, sinogram_intraring_stepsize=4,
                                                      sinogram_random=randoms, tof_meta=_tof_meta(tof), lazy=lazy))
    assert isinstance(estimates[1], LazySinogram)
    assert _close(estimates[1].to_dense(), estimates[0])      # the scale factor is fitted to back projections
    assert torch.isfinite(estimates[0]).all() and estimates[0].sum() > 0


@needs_kernels
@pytest.mark.parametrize("tof", [False, True])
def test_lazy_list_mode_scatter_estimate_equals_dense(tof, small_chunks):
    """The scatter estimate cell of the list mode tutorials: the events, looked up in the lazy estimate, get the same values."""
    activity, attenuation = _phantom()
    proj_meta = PETLMProjMeta(_events(20000, tof, seed=15), INFO, tof_meta=_tof_meta(tof))
    system_matrix = PETLMSystemMatrix(OBJECT_META, proj_meta, attenuation_map=attenuation, N_splits=2)
    per_event = []
    for lazy in (False, True):
        torch.manual_seed(0)
        scatter = sss.get_sss_scatter_estimate(OBJECT_META, proj_meta, activity, attenuation, system_matrix,
                                               image_stepsize=2, sinogram_interring_stepsize=4, sinogram_intraring_stepsize=4,
                                               tof_meta=_tof_meta(tof), lazy=lazy)
        per_event.append(shared.sinogram_to_listmode(proj_meta.detector_ids, scatter, proj_meta.info).cpu())
    assert _close(per_event[1], per_event[0])                 # the scale factor is fitted to back projections
    assert torch.isfinite(per_event[0]).all() and per_event[0].sum() > 0
