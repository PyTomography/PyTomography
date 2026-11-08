"""Proton CT list mode: the standard form (PCTProjMeta) and its file, bands of paths, the projector (PyTorch walk, and
the CUDA kernels against it), SART with its fused terms and its support, and FBP."""
import math

import numpy as np
import pytest
import torch

import pytomography
from pytomography.algorithms import SART, FilteredBackProjection
from pytomography.io.PCT import load_listmode, save_listmode
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PCT import PCTProjMeta
from pytomography.metadata.PCT.pct_metadata import FWHM_PER_SD
from pytomography.projectors.PCT import PCTSystemMatrix, band_paths, hull_from_image
from pytomography.projectors.PCT import _pct_cuda

OBJECT = ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(24, 22, 6))


def _scan(angles=(0.0, 37.0, 180.0, 270.0), per_view=300, slope=0.05, entry_fwhm=0.0, seed=0):
    """Protons across a small grid: entries over x in [-20, 20] mm and z in [-4, 4] mm, exits bent by up to ``slope``
    in x (planes 60 mm either side of the axis)."""
    g = torch.Generator().manual_seed(seed)
    n = len(angles) * per_view
    x = (torch.rand(n, generator=g) * 2 - 1) * 20
    z = (torch.rand(n, generator=g) * 2 - 1) * 4
    sx = (torch.rand(n, generator=g) * 2 - 1) * slope
    entry = torch.stack([x, z, torch.zeros(n), torch.zeros(n)], 1)
    exit = torch.stack([x + sx * 120, z, sx, torch.zeros(n)], 1)
    return PCTProjMeta(angles, -60.0, 60.0, entry, exit, [per_view] * len(angles), 330e3, entry_fwhm=entry_fwhm,
                       beam_divergence=0.0437)


def _cylinder(radius=16.0):
    x = (np.arange(OBJECT.shape[0]) - (OBJECT.shape[0] - 1) / 2) * 2
    y = (np.arange(OBJECT.shape[1]) - (OBJECT.shape[1] - 1) / 2) * 2
    disk = x[:, None] ** 2 + y[None, :] ** 2 < radius ** 2
    return np.repeat(disk[:, :, None], OBJECT.shape[2], axis=2)


def test_load_turns_each_view_into_the_object_frame():
    pm = _scan(angles=(90.0,), per_view=5)
    d = pm.load(0, 'cpu')
    # a view turned by 90 degrees: its (x, y) is the object's (y, -x)
    assert torch.allclose(d['entry_points'][:, 0], torch.full((5,), -60.0), atol=1e-4)
    assert torch.allclose(d['entry_points'][:, 1], -pm.entry[:, 0], atol=1e-4)
    assert torch.allclose(d['entry_directions'], torch.tensor([[1.0, 0.0, 0.0]]).expand(5, 3), atol=1e-6)
    assert d['index'].tolist() == list(range(5))
    assert pm.nbytes() == 5 * 32


def test_straight_paths_through_ones_have_the_grid_length():
    pm = _scan(angles=(0.0, 90.0), slope=0.0)
    sm = PCTSystemMatrix(OBJECT, pm, path='straight')
    wepl = sm.forward(torch.ones(OBJECT.shape))
    # along y (view 0) the grid is 44 mm deep, along x (view 90) 48 mm
    assert torch.allclose(wepl[:300], torch.full((300,), 44.0, device=wepl.device), atol=1e-3)
    assert torch.allclose(wepl[300:], torch.full((300,), 48.0, device=wepl.device), atol=1e-3)


@pytest.mark.parametrize('band', [1, 3])
def test_back_projection_is_the_transpose(band):
    pm = _scan(entry_fwhm=3.0)
    sm = PCTSystemMatrix(OBJECT, pm, band=band, hull=_cylinder())
    g = torch.Generator().manual_seed(1)
    f = torch.rand(OBJECT.shape, generator=g)
    y = torch.rand(pm.N_protons, generator=g).to(pytomography.device)
    lhs = float((sm.forward(f) * y).sum())
    rhs = float((f.to(pytomography.device) * sm.backward(y)).sum())
    assert lhs == pytest.approx(rhs, rel=1e-5)
    norm = sm.compute_normalization_factor()
    assert float(norm.sum()) == pytest.approx(float(sm.forward(torch.ones(OBJECT.shape)).sum()), rel=1e-5)


def test_band_weights_and_spread():
    pm = _scan(per_view=2000, entry_fwhm=3.69 * FWHM_PER_SD)
    d = pm.load(0, 'cpu')
    b = band_paths(d, 5, pm.beam_divergence)
    w = b['weights'].view(-1, 25)
    assert torch.allclose(w.sum(1), torch.ones(2000), atol=1e-5)
    off = (b['entry_points'] - d['entry_points'].repeat_interleave(25, 0)).view(-1, 25, 3)
    mean = (w[:, :, None] * off).sum(1)
    sd = torch.sqrt((w * (off[:, :, 0] - mean[:, None, 0]) ** 2).sum(1))
    # the weighted spread is the entry's on average over the band's shifts
    assert float(sd.mean()) == pytest.approx(3.69, rel=0.01)
    # each path turns with its offset by the beam's divergence
    turn = (b['entry_directions'][:, 0] - d['entry_directions'].repeat_interleave(25, 0)[:, 0]).view(-1, 25)
    assert torch.allclose(turn, math.radians(pm.beam_divergence) * off[:, :, 0], atol=1e-5)


@pytest.mark.parametrize('per_proton', [False, True])
def test_listmode_file_round_trip(tmp_path, per_proton):
    pm = _scan(per_view=50, entry_fwhm=3.0)
    if per_proton:
        pm = pm.replace(entry_fwhm=torch.linspace(1.0, 4.0, pm.N_protons))
    wepl = torch.linspace(0.0, 300.0, pm.N_protons)
    save_listmode(tmp_path / 'scan.h5', wepl, pm)
    w2, om, pm2 = load_listmode(tmp_path / 'scan.h5', dr=2.0)
    assert torch.equal(w2, wepl) and torch.equal(pm2.entry, pm.entry) and torch.equal(pm2.exit, pm.exit)
    assert pm2.angles.tolist() == pm.angles.tolist() and pm2.view_sizes.tolist() == pm.view_sizes.tolist()
    assert (torch.equal(pm2.entry_fwhm, pm.entry_fwhm) if per_proton else pm2.entry_fwhm == 3.0)
    assert pm2.beam_divergence == pm.beam_divergence and om.shape[2] >= 4
    # some views, a fraction of their protons: read a view at a time, the same protons whatever else is read
    w3, _, pm3 = load_listmode(tmp_path / 'scan.h5', dr=2.0, views=[1, 3], fraction=0.5, seed=4)
    w4, _, pm4 = load_listmode(tmp_path / 'scan.h5', dr=2.0, views=[3], fraction=0.5, seed=4)
    assert pm3.angles.tolist() == [37.0, 270.0] and 10 < pm3.N_protons < 90
    assert torch.equal(w3[int(pm3.view_sizes[0]):], w4)


def test_sart_fused_terms_equal_the_three_projections_and_hold_the_support():
    pm = _scan(per_view=200, entry_fwhm=3.0)
    truth = torch.as_tensor(_cylinder(), dtype=torch.float32)
    hull = _cylinder()
    wepl = PCTSystemMatrix(OBJECT, pm, band=3, hull=hull).forward(truth)

    class Unfused(PCTSystemMatrix):
        _sart_terms = None

    support = _cylinder(radius=18.0)
    images = []
    for cls in (PCTSystemMatrix, Unfused):
        algorithm = SART(cls(OBJECT, pm, band=3, hull=hull), wepl, object_initial=torch.full(OBJECT.shape, 0.5),
                         relaxation_sequence=lambda _: 0.5, support=support)
        images.append(algorithm(n_iters=2, n_subsets=2))
    assert torch.allclose(images[0], images[1], atol=1e-4)
    outside = torch.as_tensor(~support).to(images[0].device)
    assert float(images[0][outside].abs().max()) == 0.0


def test_fbp_of_a_water_cylinder():
    angles = tuple(np.arange(0, 360, 10.0))
    pm = _scan(angles=angles, per_view=1500, slope=0.0)
    truth = torch.as_tensor(_cylinder(), dtype=torch.float32)
    wepl = PCTSystemMatrix(OBJECT, pm, path='straight').forward(truth)
    sm = PCTSystemMatrix(OBJECT, pm, path='straight')
    image = FilteredBackProjection(wepl, sm, filter='hann', n_angles=36, line='points')()
    covered = np.zeros(OBJECT.shape, bool)
    covered[:, :, 1:-1] = True                      # the protons cover z in [-4, 4] mm: slices 1 to 4
    inner = torch.as_tensor(_cylinder(radius=10.0) & covered).to(image.device)
    assert float(image[inner].mean()) == pytest.approx(1.0, abs=0.05)
    hull = hull_from_image(image, OBJECT, fwhm=2.0, opening=2.0)
    assert (hull.numpy() != _cylinder())[covered].mean() < 0.05
    second = FilteredBackProjection(wepl, PCTSystemMatrix(OBJECT, pm, path='straight', hull=hull), filter='hann',
                                    n_angles=36, line='hull')()
    assert float(second[inner].mean()) == pytest.approx(1.0, abs=0.05)


@pytest.mark.gpu
def test_cuda_kernels_match_the_pytorch_walk():
    if not _pct_cuda.available('cuda'):
        pytest.skip('needs CuPy')
    pm = _scan(per_view=2000, entry_fwhm=3.0)
    g = torch.Generator().manual_seed(2)
    f = torch.rand(OBJECT.shape, generator=g).cuda()
    y = torch.rand(pm.N_protons, generator=g).cuda()
    out = []
    for cuda in (True, False):
        sm = PCTSystemMatrix(OBJECT, pm, band=3, hull=_cylinder())
        sm._cuda = cuda
        paths = sm._paths(1, 0, 500)
        out.append((sm.forward(f), *sm._back(y, None), *sm.hull_points(paths)))
    (fc, bc, nc, h0c, h1c, hitc), (ft, bt, nt, h0t, h1t, hitt) = out
    assert torch.allclose(fc, ft, rtol=1e-4, atol=1e-4)
    assert torch.allclose(bc, bt, rtol=1e-4, atol=1e-3 * float(bt.abs().max()))
    assert torch.allclose(nc, nt, rtol=1e-4, atol=1e-3 * float(nt.abs().max()))
    assert torch.equal(hitc, hitt)
    assert torch.allclose(h0c[hitc], h0t[hitt], atol=1e-4) and torch.allclose(h1c[hitc], h1t[hitt], atol=1e-4)
