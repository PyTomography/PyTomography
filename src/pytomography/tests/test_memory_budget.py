"""pytomography.set_memory_budget sets how much host memory the memory-heavy steps may use, and they split their work to
fit: steps over every crystal pair go a block at a time, sinograms larger than a quarter of the budget are computed one
subset at a time, and a reconstruction whose subsets would not fit stops before it starts. These tests check that each
split gives the result of doing the work at once, and that the budget switches as documented."""
from __future__ import annotations

import sys
import types

import pytest
import torch

try:
    import parallelproj_core  # noqa: F401
except ImportError:                                   # only the tests marked needs_kernels call the projector kernels
    sys.modules["parallelproj_core"] = types.ModuleType("parallelproj_core")

import pytomography
from pytomography.io.PET import gate, shared
from pytomography.io.PET.shared import LazySinogram
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETSinogramPolygonProjMeta, PETTOFMeta
from pytomography.projectors.PET import PETLMSystemMatrix, PETSinogramSystemMatrix
from pytomography.utils import memory, sss

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
N_CRYSTALS = INFO['NrCrystalsPerRing'] * INFO['NrRings']
N_PAIRS = N_CRYSTALS * (N_CRYSTALS - 1) // 2
SHAPE = (INFO['NrCrystalsPerRing'] // 2, INFO['NrCrystalsPerRing'] + 1, (INFO['moduleAxialNr'] * INFO['crystalAxialNr'])**2)
TOF_META = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3)
TOF_BYTES = 4 * SHAPE[0] * SHAPE[1] * SHAPE[2] * 5      # 2.7 MB
OBJECT_META = ObjectMeta(dr=(4, 4, 4), shape=(32, 32, 6))


@pytest.fixture
def budget(monkeypatch):
    """Sets the memory budget for one test (in GB); it is None again afterwards."""
    def set_budget(gb):
        monkeypatch.setattr(pytomography, "memory_budget", None if gb is None else gb * 1e9)
    return set_budget


def _events(n: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    d0 = torch.randint(0, N_CRYSTALS, (n,), generator=gen)
    d1 = (d0 + N_CRYSTALS // 2 + torch.randint(-24, 24, (n,), generator=gen)) % N_CRYSTALS
    return torch.stack([d0, d1, torch.randint(0, 5, (n,), generator=gen)], dim=1)


def test_set_memory_budget():
    try:
        pytomography.set_memory_budget(25)
        assert pytomography.memory_budget == 25e9
        with pytest.raises(ValueError):
            pytomography.set_memory_budget(0)
    finally:
        pytomography.set_memory_budget(None)
    assert pytomography.memory_budget is None


def test_block_and_subset_sizes(budget):
    budget(None)
    assert memory.block_size(100, default=7) == 7 and not memory.prefer_lazy(1e12) and memory.subsets_for_budget(1e12, minimum=3) == 3
    budget(8)                                         # an eighth is 1 GB; a quarter is 2 GB; half is 4 GB
    assert memory.block_size(100, default=7) == 10**7
    assert memory.prefer_lazy(2.1e9) and not memory.prefer_lazy(1.9e9)
    assert memory.subsets_for_budget(34.6e9) == 26    # three 34.6 GB / 26 arrays fit in 4 GB, 25 do not
    assert memory.subsets_for_budget(34.6e9, held_bytes=2e9) == 18
    assert memory.subsets_for_budget(1e9, minimum=14) == 14
    with pytest.raises(ValueError):
        memory.subsets_for_budget(1e9, held_bytes=9e9)


def test_sinograms_follow_the_budget(budget):
    events = _events(20000)
    dense = shared.listmode_to_sinogram(events, INFO, tof_meta=TOF_META)
    assert isinstance(dense, torch.Tensor)            # no budget: whole, as before
    budget(4 * TOF_BYTES / 1e9 * 0.99)                # the sinogram takes just over a quarter
    lazy = shared.listmode_to_sinogram(events, INFO, tof_meta=TOF_META)
    assert isinstance(lazy, LazySinogram) and torch.equal(lazy.to_dense(), dense)
    assert isinstance(shared.listmode_to_sinogram(events, INFO, tof_meta=TOF_META, lazy=False), torch.Tensor)
    assert lazy._angles_per_chunk() < SHAPE[0]        # read a sixteenth of the budget at a time
    budget(4 * TOF_BYTES / 1e9 * 1.01)
    assert isinstance(shared.listmode_to_sinogram(events, INFO, tof_meta=TOF_META), torch.Tensor)


def test_scatter_interpolation_follows_the_budget(budget):
    proj_meta = PETSinogramPolygonProjMeta(INFO, TOF_META)
    idx_intraring, idx_ring, detector_ids = sss.get_sample_detector_ids(proj_meta, 4, 4)
    sparse = sss.SparseSinogram(detector_ids, torch.rand((5, detector_ids.shape[0]), generator=torch.Generator().manual_seed(1)).to(DEV), INFO, tof_meta=TOF_META)
    dense = sss.interpolate_sparse_sinogram(sparse, proj_meta, idx_intraring, idx_ring, tof_bins=range(5))
    budget(4 * TOF_BYTES / 1e9 * 0.99)
    lazy = sss.interpolate_sparse_sinogram(sparse, proj_meta, idx_intraring, idx_ring, tof_bins=range(5))
    assert isinstance(lazy, LazySinogram) and torch.equal(lazy.to_dense(), dense)


@pytest.mark.parametrize("normalization", [False, True])
def test_pair_sinogram_blocks_follow_the_budget(budget, normalization):
    weights = torch.rand(N_PAIRS, generator=torch.Generator().manual_seed(2))
    expected = shared.listmode_to_sinogram(torch.combinations(torch.arange(N_CRYSTALS), 2), INFO, weights=weights, normalization=normalization)
    budget(130 * 3000 * 8 / 1e9)                      # blocks of about 3000 pairs: 44 blocks
    assert torch.equal(shared.all_pairs_to_sinogram(weights, INFO, normalization=normalization), expected)


def test_normalization_weights_in_blocks_equal_all_pairs_at_once(budget):
    """The weights of every crystal pair, from a calibration scan's symmetry histogram, as the previous code computed them
    from all pairs at once (about 75 GB for the mMR). The cylinder is smaller than the scanner, so some weights are NaN,
    in the same places."""
    shape = [INFO['crystalAxialNr'], INFO['crystalAxialNr'], INFO['crystalTransNr'], INFO['crystalTransNr'],
             INFO['submoduleAxialNr'] * 2 - 1, INFO['moduleAxialNr'] * 2 - 1, INFO['rsectorTransNr']]
    histo = torch.rand(shape, generator=torch.Generator().manual_seed(3)) * 100
    cylinder_radius = 100.0
    # the previous implementation
    vals_all_pairs = gate.get_symmetry_histogram_all_combos(INFO)
    bin_edges = [torch.arange(n + 1).to(torch.float32) - 0.5 for n in shape]
    N_bins = torch.histogramdd(vals_all_pairs.to(torch.float32), bin_edges)[0]
    scanner_LUT = shared.get_scanner_LUT(INFO)
    all_LOR_ids = torch.combinations(torch.arange(scanner_LUT.shape[0]).to(torch.int32), 2)
    geometric_correction_factor = 1/(torch.sqrt(1-(torch.abs(gate.get_radius(all_LOR_ids, scanner_LUT)) / cylinder_radius )**2) + pytomography.delta)
    expected = (histo/N_bins)[tuple(vals_all_pairs.T)] * geometric_correction_factor
    assert expected.isnan().any() and not expected.isnan().all()
    for pairs_per_block in (5000, None):
        got = gate._weights_from_symmetry_histogram(histo, INFO, cylinder_radius, pairs_per_block=pairs_per_block)
        assert torch.allclose(got, expected, rtol=0, atol=0, equal_nan=True)
    budget(200 * 3000 * 8 / 1e9)                      # blocks of about 3000 pairs
    assert torch.allclose(gate._weights_from_symmetry_histogram(histo, INFO, cylinder_radius), expected, rtol=0, atol=0, equal_nan=True)


def test_reconstruction_stops_when_its_subsets_do_not_fit(budget):
    sm = PETSinogramSystemMatrix(OBJECT_META, PETSinogramPolygonProjMeta(INFO, TOF_META), device='cpu')
    likelihood = PoissonLogLikelihood(sm, shared.listmode_to_sinogram(_events(1000), INFO, tof_meta=TOF_META, lazy=True))
    budget(4 * TOF_BYTES / 1e9)                       # half the budget holds two whole sinograms: three subset arrays need 2 subsets
    with pytest.raises(ValueError, match="use at least 2 subsets"):
        likelihood._set_n_subsets(1)
    likelihood._check_memory_budget(2)
    budget(None)
    likelihood._check_memory_budget(1)


def _close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5) -> bool:
    """Equal up to the order in which parallelproj adds up back projections (atomic adds)."""
    return torch.allclose(a, b, rtol=rtol, atol=rtol * float(b.abs().max()))


@needs_kernels
def test_list_mode_sensitivity_in_blocks_equals_all_pairs_at_once(budget):
    """The list mode sensitivity image goes through every crystal pair a block at a time: the same weight for every pair
    as building all pairs at once, and the same image to the projector's rounding."""
    x, y, z = torch.meshgrid(*[(torch.arange(n) - n / 2 + 0.5) * d for n, d in zip(OBJECT_META.shape, OBJECT_META.dr)], indexing="ij")
    attenuation = (0.0096 * (x**2 + y**2 < 40**2).float()).to(DEV)
    weights = torch.rand(N_PAIRS, generator=torch.Generator().manual_seed(4)) + 0.5
    proj_meta = PETLMProjMeta(_events(1000)[:, :2], INFO, weights_sensitivity=weights)
    budget(48 * 3000 * 8 / 1e9)                       # blocks of about 3000 pairs
    sm = PETLMSystemMatrix(OBJECT_META, proj_meta, attenuation_map=attenuation, N_splits=2)
    pairs = torch.combinations(torch.arange(N_CRYSTALS), 2)
    expected = torch.ones(N_PAIRS) * weights * sm._compute_attenuation_probability_projection(pairs).cpu()
    assert torch.equal(sm._compute_sensitivity_projection(all_ids=True), expected)
    budget(None)
    one_block = PETLMSystemMatrix(OBJECT_META, proj_meta, attenuation_map=attenuation, N_splits=1)
    assert _close(sm.norm_BP, one_block.norm_BP)
