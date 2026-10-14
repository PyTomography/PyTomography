"""SPECTKernelPSFTransform: arbitrary 2D kernels per plane distance (ARF decision 1, 14 Oct 2026: the PSF transform
lives in PyTomography, spectarf makes the kernels). Back projection is the transpose for kernels with no symmetry, the
FFT convolution is a direct one, the kernel axes run along r and z, the distances are the planes', a distance grid
interpolates them, and Gaussian kernels give SPECTPSFTransform's projections."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pytomography
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTKernelPSFTransform, SPECTPSFTransform
from pytomography.transforms.SPECT.psf_kernels import DEFAULT_CENTRING_MM, DEFAULT_DISTANCE_RATIO
from pytomography.utils import get_distance

CPU = torch.device('cpu')


@pytest.fixture(autouse=True)
def _cpu_device():
    """Transforms take pytomography.device when made, the system matrix at every call: the CPU for both here."""
    old = pytomography.device
    pytomography.device = CPU
    yield
    pytomography.device = old


def metas(shape=(15, 15, 9), dr_mm=4.0, angles=(0.0,), radii_mm=(200.0,)):
    obj = SPECTObjectMeta(dr=(dr_mm, dr_mm, dr_mm), shape=shape)
    proj = SPECTProjMeta((shape[1], shape[2]), (dr_mm, dr_mm), angles=np.asarray(angles, float),
                         radii=np.asarray(radii_mm, float))
    return obj, proj


def random_kernels(seed=0, dtype=torch.float64):
    """Random positive kernels with no symmetry at all."""
    gen = torch.Generator().manual_seed(seed)

    def kernels(distances_mm, pixel_mm, shape):
        k = torch.rand((len(distances_mm), *shape), generator=gen, dtype=dtype)
        k[:, 0, :] *= 3.0          # break every reflection and 180-degree symmetry
        k[:, :, -1] += 1.0
        return k

    return kernels


def shifted_delta(shift_rz):
    """A unit delta displaced by (dr, dz) pixels from the kernel's centre."""
    def kernels(distances_mm, pixel_mm, shape):
        k = torch.zeros((len(distances_mm), *shape))
        k[:, shape[0] // 2 + shift_rz[0], shape[1] // 2 + shift_rz[1]] = 1.0
        return k

    return kernels


def linear_in_distance(seed=2):
    """Random kernels whose every element is linear in the distance: a grid's interpolation is then exact."""
    gen = torch.Generator().manual_seed(seed)
    state = {}

    def kernels(distances_mm, pixel_mm, shape):
        if shape not in state:
            state[shape] = (torch.rand(shape, generator=gen, dtype=torch.float64),
                            torch.rand(shape, generator=gen, dtype=torch.float64) / 100.0)
        a, b = state[shape]
        d = torch.as_tensor(np.asarray(distances_mm), dtype=torch.float64)[:, None, None]
        return a[None] + b[None] * d

    return kernels


def configured(transform, obj, proj):
    transform.configure(obj, proj)
    return transform


def test_backward_is_the_transpose_of_forward_for_kernels_with_no_symmetry():
    obj, proj = metas()
    t = configured(SPECTKernelPSFTransform(random_kernels(), kernel_shape=(7, 5)), obj, proj)
    gen = torch.Generator().manual_seed(3)
    x = torch.rand(obj.padded_shape, generator=gen, dtype=torch.float64)
    y = torch.rand(obj.padded_shape, generator=gen, dtype=torch.float64)
    assert float((t.forward(x, 0) * y).sum()) == pytest.approx(float((x * t.backward(y, 0)).sum()), rel=1e-10)
    # the transpose of an asymmetric kernel is not the kernel (which SPECTPSFTransform's Gaussian may assume)
    assert not torch.allclose(t.forward(y, 0), t.backward(y, 0), rtol=1e-3)


def test_fft_convolution_is_the_direct_convolution_and_its_transpose_the_correlation():
    obj, proj = metas()
    t = configured(SPECTKernelPSFTransform(random_kernels(seed=1), kernel_shape=(7, 5)), obj, proj)
    x = torch.rand(obj.padded_shape, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    k = t.kernels[t._planes[200.0]]
    n = x.shape[0]
    convolution = torch.nn.functional.conv2d(x[None], k.flip((1, 2))[:, None], padding=(3, 2), groups=n)[0]
    correlation = torch.nn.functional.conv2d(x[None], k[:, None], padding=(3, 2), groups=n)[0]
    np.testing.assert_allclose(t.forward(x, 0).numpy(), convolution.numpy(), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(t.backward(x, 0).numpy(), correlation.numpy(), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize('shift_rz', [(2, 0), (0, 1), (2, -1)])
def test_a_kernels_first_axis_runs_along_r_and_its_second_along_z(shift_rz):
    obj, proj = metas()
    source = torch.zeros(obj.shape)
    source[7, 4, 3] = 1.0                          # off centre in every direction
    plain = SPECTSystemMatrix([], [], obj, proj).forward(source)
    blurred = SPECTSystemMatrix([SPECTKernelPSFTransform(shifted_delta(shift_rz), kernel_shape=(9, 9))], [], obj, proj).forward(source)
    assert plain.shape == blurred.shape == (1, 15, 9) and float(plain.sum()) > 0.5
    np.testing.assert_allclose(blurred.numpy(), torch.roll(plain, shifts=shift_rz, dims=(1, 2)).numpy(), atol=1e-6)


def test_the_kernels_are_made_once_at_every_planes_distance_plus_the_front_cover():
    obj, proj = metas(angles=(0.0, 90.0, 180.0), radii_mm=(200.0, 250.0, 200.0))
    calls = []

    def kernels(distances_mm, pixel_mm, shape):
        calls.append((np.array(distances_mm), pixel_mm, shape))
        return torch.ones((len(distances_mm), *shape))

    kernels.front_cover_mm = 1.5
    t = configured(SPECTKernelPSFTransform(kernels, kernel_shape=(3, 3)), obj, proj)
    assert len(calls) == 1 and t.front_cover_mm == 1.5     # one call, over every view's planes
    n_x = obj.padded_shape[0]
    want = {r: get_distance(n_x, r, obj.dx) + 1.5 for r in (200.0, 250.0)}
    np.testing.assert_allclose(calls[0][0], np.unique(np.concatenate(list(want.values()))))
    assert calls[0][1] == pytest.approx((4.0, 4.0)) and calls[0][2] == (3, 3)
    for r, d in want.items():
        np.testing.assert_allclose(t.distances_mm[t._planes[r].numpy()], d)


def test_the_default_kernel_shape_couples_every_pair_of_voxels_in_a_plane():
    obj, proj = metas()
    shapes = []

    def kernels(distances_mm, pixel_mm, shape):
        shapes.append(shape)
        return torch.ones((len(distances_mm), *shape))

    configured(SPECTKernelPSFTransform(kernels), obj, proj)
    _, n_y, n_z = obj.padded_shape
    assert shapes == [(2 * n_y - 1, 2 * n_z - 1)]


def test_wrong_kernels_are_refused():
    obj, proj = metas()
    with pytest.raises(ValueError, match='expected'):
        configured(SPECTKernelPSFTransform(lambda d, p, s: torch.ones((len(d), 5, 5)), kernel_shape=(3, 3)), obj, proj)
    with pytest.raises(ValueError, match='odd'):
        configured(SPECTKernelPSFTransform(lambda d, p, s: torch.ones((len(d), *s)), kernel_shape=(4, 3)), obj, proj)
    with pytest.raises(TypeError):
        SPECTKernelPSFTransform(torch.ones((3, 3)))


@pytest.mark.parametrize('clip', [True, False])
def test_negative_tails_are_clipped_and_their_share_recorded(clip):
    obj, proj = metas()

    def kernels(distances_mm, pixel_mm, shape):
        k = torch.full((len(distances_mm), *shape), -0.01)
        k[:, 2, 2] = 1.0
        return k

    t = configured(SPECTKernelPSFTransform(kernels, kernel_shape=(5, 5), clip_negative=clip), obj, proj)
    np.testing.assert_allclose(t.clipped_fraction, 0.24 / 1.24, rtol=1e-6)
    assert (float(t.kernels.min()) >= 0.0) == clip


def test_normalize_scales_every_kernel_to_sum_to_one_and_the_default_keeps_its_scale():
    obj, proj = metas(angles=(0.0, 90.0), radii_mm=(150.0, 220.0))
    absolute = configured(SPECTKernelPSFTransform(linear_in_distance(), kernel_shape=(5, 3)), obj, proj)
    unit = configured(SPECTKernelPSFTransform(linear_in_distance(), kernel_shape=(5, 3), normalize=True), obj, proj)
    np.testing.assert_allclose(unit.kernels.sum(dim=(1, 2)).numpy(), 1.0, rtol=1e-12)
    assert not np.allclose(absolute.kernels.sum(dim=(1, 2)).numpy(), 1.0)
    np.testing.assert_allclose((absolute.kernels / absolute.kernels.sum(dim=(1, 2), keepdim=True)).numpy(),
                               unit.kernels.numpy(), rtol=1e-12)


def test_a_distance_grid_interpolates_every_planes_kernel():
    obj, proj = metas(angles=(0.0, 90.0, 180.0), radii_mm=(200.0, 231.7, 254.3))
    exact = configured(SPECTKernelPSFTransform(linear_in_distance(), kernel_shape=(5, 3)), obj, proj)
    grid = configured(SPECTKernelPSFTransform(linear_in_distance(), kernel_shape=(5, 3), distance_step_mm=7.0), obj, proj)
    assert not exact._grid and grid._grid and np.allclose(np.diff(grid.distances_mm), 7.0)
    x = torch.rand(obj.padded_shape, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    for view in range(3):
        np.testing.assert_allclose(grid.forward(x, view).numpy(), exact.forward(x, view).numpy(), rtol=1e-9, atol=1e-12)
        y = torch.rand(obj.padded_shape, generator=torch.Generator().manual_seed(view), dtype=torch.float64)
        assert float((grid.forward(x, view) * y).sum()) == pytest.approx(float((x * grid.backward(y, view)).sum()), rel=1e-10)


def test_many_orbit_radii_put_the_kernels_on_a_geometric_grid():
    radii = 200.0 + 3.1 * np.arange(12)
    obj, proj = metas(angles=tuple(30.0 * np.arange(12)), radii_mm=tuple(radii))
    t = configured(SPECTKernelPSFTransform(linear_in_distance(), kernel_shape=(3, 3)), obj, proj)
    assert t._grid
    from_source = t.distances_mm + DEFAULT_CENTRING_MM          # the distances to the centring plane grow geometrically
    np.testing.assert_allclose(from_source[1:] / from_source[:-1], 1.0 + DEFAULT_DISTANCE_RATIO, rtol=1e-12)
    planes = np.concatenate([get_distance(obj.padded_shape[0], r, obj.dx) for r in radii])
    assert t.distances_mm[0] <= planes.min() + 1e-9 and t.distances_mm[-1] >= planes.max() - 1e-9


def test_the_grids_centring_depth_comes_from_the_kernel_function():
    radii = 200.0 + 3.1 * np.arange(12)
    obj, proj = metas(angles=tuple(30.0 * np.arange(12)), radii_mm=tuple(radii))
    kernels = linear_in_distance()
    kernels.centring_mm = 30.0
    t = configured(SPECTKernelPSFTransform(kernels, kernel_shape=(3, 3)), obj, proj)
    from_source = t.distances_mm + 30.0
    np.testing.assert_allclose(from_source[1:] / from_source[:-1], 1.0 + DEFAULT_DISTANCE_RATIO, rtol=1e-12)


def gaussian_kernels(a, b, shape_from):
    """The kernels SPECTPSFTransform's Gaussian layers apply, as 2D kernels: the outer product of the same normalised
    1D Gaussians, of the sizes ``shape_from`` gives."""
    def kernels(distances_mm, pixel_mm, shape):
        sigma = a * np.asarray(distances_mm) + b
        out = []
        for s in sigma:
            g = []
            for axis, k in enumerate(shape):
                x = torch.arange(-(k // 2), k // 2 + 1, dtype=torch.float64)
                w = torch.exp(-x ** 2 / (2 * (s / pixel_mm[axis]) ** 2 + pytomography.delta))
                g.append(w / w.sum())
            out.append(g[0][:, None] * g[1][None, :])
        return torch.stack(out).to(torch.float32)

    return kernels


def test_gaussian_kernels_give_the_projections_of_the_gaussian_psf():
    obj, proj = metas(shape=(16, 16, 10), dr_mm=4.8, angles=(0.0, 45.0, 90.0, 135.0), radii_mm=(180.0,) * 4)
    a, b = 0.03, 1.2
    gaussian = SPECTPSFTransform(SPECTPSFMeta((a, b)))
    reference = SPECTSystemMatrix([gaussian], [], obj, proj)
    layer = gaussian.layers[180.0]
    shape = (int(layer.layer_r.kernel_size[0]), int(layer.layer_z.kernel_size[0]))
    kernel = SPECTKernelPSFTransform(gaussian_kernels(a, b, shape), kernel_shape=shape)
    system = SPECTSystemMatrix([kernel], [], obj, proj)
    source = torch.rand(obj.shape, generator=torch.Generator().manual_seed(7))
    want, got = reference.forward(source), system.forward(source)
    np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-4, atol=1e-5 * float(want.max()))
    views = torch.rand(want.shape, generator=torch.Generator().manual_seed(8))
    np.testing.assert_allclose(system.backward(views).numpy(), reference.backward(views).numpy(), rtol=1e-4,
                               atol=1e-5 * float(reference.backward(views).max()))


def test_memory_reports_what_the_transform_keeps():
    obj, proj = metas(angles=(0.0, 90.0), radii_mm=(160.0, 230.0))
    t = configured(SPECTKernelPSFTransform(random_kernels(dtype=torch.float32), kernel_shape=(7, 5)), obj, proj)
    for view in range(2):
        t.forward(torch.rand(obj.padded_shape), view)
    ram, gpu, call = t.memory(obj.padded_shape)
    held = t.kernels.numel() * 4 + sum(s.numel() * 8 for s in t._spectra.values())
    assert gpu == 0.0 and ram == pytest.approx(held, rel=1e-9) and call > 0
    estimate = SPECTSystemMatrix([t], [], obj, proj).estimate_memory(n_subsets=1)
    assert any('object transforms keep' in p.name for p in estimate.parts)


def test_osem_runs_with_the_kernels():
    from pytomography.algorithms import OSEM
    from pytomography.likelihoods import PoissonLogLikelihood
    obj, proj = metas(shape=(9, 9, 5), dr_mm=4.8, angles=(0.0, 90.0, 180.0, 270.0), radii_mm=(150.0,) * 4)
    system = SPECTSystemMatrix([SPECTKernelPSFTransform(gaussian_kernels(0.03, 1.0, None), kernel_shape=(9, 9))], [],
                               obj, proj)
    truth = torch.zeros(obj.shape)
    truth[3:6, 3:6, 1:4] = 1.0
    recon = OSEM(PoissonLogLikelihood(system, system.forward(truth)))(n_iters=2, n_subsets=2)
    assert torch.isfinite(recon).all() and float(recon.sum()) > 0.0
