"""Least-squares solutions checked independently with SciPy and the singular-value decomposition."""
from __future__ import annotations

import numpy as np
import pytest
import pytomography
import torch
from pytomography.algorithms import CGLS
from pytomography.likelihoods import NegativeMSELikelihood, poisson_weights
from pytomography.metadata import ObjectMeta, ProjMeta
from pytomography.projectors import SystemMatrix
from scipy.sparse.linalg import cg


class MatrixSystem(SystemMatrix):
    def __init__(self, matrix, projection_device=None):
        self.matrix = torch.as_tensor(matrix, dtype=pytomography.dtype, device=pytomography.device)
        self.projection_device = projection_device or pytomography.device
        super().__init__(ObjectMeta((1, 1, 1), (self.matrix.shape[1], 1, 1)),
                         ProjMeta(torch.zeros(self.matrix.shape[0])))

    def forward(self, object, subset_idx=None):
        return (self.matrix @ object.reshape(-1)).to(self.projection_device)

    def backward(self, projections, subset_idx=None):
        return (self.matrix.T @ projections.to(self.matrix)).reshape(self.object_meta.shape)


@pytest.fixture(autouse=True)
def double_precision():
    dtype = pytomography.dtype
    pytomography.set_dtype(torch.float64)
    yield
    pytomography.set_dtype(dtype)


def problem(matrix, data, **kwargs):
    system = MatrixSystem(matrix)
    likelihood = NegativeMSELikelihood(system, torch.as_tensor(data, dtype=pytomography.dtype), **kwargs)
    return system, likelihood


@pytest.mark.parametrize("preconditioned", [False, True])
def test_inconsistent_problem_matches_scipy_cg_and_lstsq(preconditioned):
    rng = np.random.default_rng(41)
    matrix = rng.normal(size=(53, 17)) * np.geomspace(0.2, 5, 17)
    data = rng.normal(size=53)
    background = rng.uniform(0, 0.3, 53)
    system, likelihood = problem(matrix, data, additive_term=torch.tensor(background))
    diagonal = torch.tensor(1 / np.sum(matrix * matrix, axis=0)).reshape(system.object_meta.shape)
    algorithm = CGLS(likelihood, preconditioner=diagonal if preconditioned else None)
    actual = algorithm(200, rtol=1e-11).cpu().numpy().reshape(-1)
    expected = np.linalg.lstsq(matrix, data - background, rcond=None)[0]
    reference, info = cg(matrix.T @ matrix, matrix.T @ (data - background), rtol=1e-12, maxiter=200)
    assert info == 0 and algorithm.converged
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(actual, reference, rtol=1e-8, atol=1e-9)
    assert actual.min() < 0
    assert np.linalg.norm(matrix @ actual - (data - background)) > 1
    actual_normal = np.linalg.norm(matrix.T @ (data - background - matrix @ actual))
    assert actual_normal <= 1e-11 * np.linalg.norm(matrix.T @ (data - background))
    assert algorithm.normal_residual_norms[-1] == pytest.approx(actual_normal, abs=1e-12)


def test_diagonal_preconditioning_solves_a_scaled_diagonal_system_in_one_step():
    diagonal = np.geomspace(0.01, 100, 12)
    truth = np.linspace(-2, 3, len(diagonal))
    system, likelihood = problem(np.diag(diagonal), diagonal * truth)
    preconditioner = torch.tensor(1 / diagonal**2).reshape(system.object_meta.shape)
    algorithm = CGLS(likelihood, preconditioner=preconditioner)
    actual = algorithm(1, rtol=1e-12).cpu().numpy().reshape(-1)
    np.testing.assert_allclose(actual, truth, rtol=1e-12, atol=1e-12)
    assert algorithm.iterations == 1 and algorithm.converged


@pytest.mark.parametrize("nonzero_initial", [False, True])
def test_rank_deficiency_and_initial_null_space_component(nonzero_initial):
    matrix = np.array([[1, 2, 0, 0], [0, 0, 3, 6], [2, 4, 0, 0]], dtype=float)
    data = np.array([1, -2, 3], dtype=float)
    initial = np.array([2, -1, 4, -2], dtype=float) if nonzero_initial else np.zeros(4)
    _, likelihood = problem(matrix, data)
    algorithm = CGLS(likelihood, object_initial=torch.tensor(initial).reshape(4, 1, 1))
    actual = algorithm(20, rtol=1e-12).cpu().numpy().reshape(-1)
    expected = np.linalg.lstsq(matrix, data, rcond=None)[0] + initial
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert algorithm.converged


def test_zero_data_and_exact_initial_solution_do_not_divide_by_zero():
    _, likelihood = problem(np.eye(4), np.zeros(4))
    algorithm = CGLS(likelihood)
    assert torch.count_nonzero(algorithm(10)) == 0
    assert algorithm.iterations == 0 and algorithm.converged
    _, likelihood = problem(np.eye(4), np.arange(4.0))
    initial = torch.arange(4.0).reshape(4, 1, 1)
    algorithm = CGLS(likelihood, object_initial=initial)
    torch.testing.assert_close(algorithm(10).cpu(), initial)
    assert algorithm.iterations == 0 and algorithm.converged


def test_restarting_preserves_data_and_converges_from_current_image():
    rng = np.random.default_rng(12)
    matrix, data = rng.normal(size=(30, 9)), rng.normal(size=30)
    _, likelihood = problem(matrix, data, scaling_constant=7)
    initial = torch.ones(9, 1, 1)
    algorithm = CGLS(likelihood, object_initial=initial)
    first = algorithm(2, rtol=1e-12).clone()
    assert algorithm.iterations == 2 and not algorithm.converged
    expected = np.linalg.lstsq(matrix, data, rcond=None)[0]
    actual = algorithm(100, rtol=1e-12).cpu().numpy().reshape(-1)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert algorithm.converged and not torch.equal(algorithm.object_prediction, first)
    torch.testing.assert_close(initial, torch.ones_like(initial))
    np.testing.assert_array_equal(likelihood.projections.cpu().numpy(), data)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 2e-5), (torch.float64, 1e-11)])
def test_precision_and_absolute_stopping(dtype, tolerance):
    pytomography.set_dtype(dtype)
    rng = np.random.default_rng(61)
    matrix, data = rng.normal(size=(25, 8)), rng.normal(size=25)
    _, likelihood = problem(matrix, data)
    algorithm = CGLS(likelihood)
    image = algorithm(100, rtol=0, atol=tolerance)
    normal = likelihood.system_matrix.backward(likelihood.projections.to(image) - likelihood.system_matrix.forward(image))
    assert algorithm.converged and float(torch.linalg.vector_norm(normal)) <= tolerance
    assert image.dtype == dtype


@pytest.mark.parametrize("preconditioned", [False, True])
def test_weighted_problem_matches_lstsq_on_the_whitened_system(preconditioned):
    rng = np.random.default_rng(7)
    matrix = rng.uniform(0, 1, size=(61, 19)) * (rng.uniform(size=(61, 19)) < 0.4) + 0.01
    mean = matrix @ rng.uniform(1, 20, 19) + 2.0
    data = rng.poisson(mean).astype(float)
    background = np.full(61, 2.0)
    weights = 1 / np.maximum(data, 1)
    system, likelihood = problem(matrix, data, additive_term=torch.tensor(background), weights=torch.tensor(weights))
    diagonal = torch.tensor(1 / (matrix.T @ (weights * (matrix @ np.ones(19))))).reshape(system.object_meta.shape)
    algorithm = CGLS(likelihood, preconditioner=diagonal if preconditioned else None)
    actual = algorithm(200, rtol=1e-11).cpu().numpy().reshape(-1)
    root = np.sqrt(weights)
    expected = np.linalg.lstsq(root[:, None] * matrix, root * (data - background), rcond=None)[0]
    assert algorithm.converged
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)
    unweighted = np.linalg.lstsq(matrix, data - background, rcond=None)[0]
    assert np.abs(actual - unweighted).max() > 1e-3   # the weights change the solution
    normal = np.linalg.norm(matrix.T @ (weights * (data - background - matrix @ actual)))
    assert algorithm.normal_residual_norms[-1] == pytest.approx(normal, abs=1e-10)


def test_unit_weights_give_the_unweighted_iterates():
    rng = np.random.default_rng(3)
    matrix, data = rng.normal(size=(30, 12)), rng.normal(size=30)
    _, plain = problem(matrix, data)
    _, ones = problem(matrix, data, weights=torch.ones(30, dtype=torch.float64))
    for n in (1, 4, 9):
        torch.testing.assert_close(CGLS(ones)(n, rtol=0), CGLS(plain)(n, rtol=0), rtol=1e-12, atol=1e-12)


def test_zero_weight_ignores_a_bin():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    data = np.array([1.0, 2.0, 3.0, 50.0])   # the last bin is an outlier
    _, likelihood = problem(matrix, data, weights=torch.tensor([1.0, 1.0, 1.0, 0.0]))
    image = CGLS(likelihood)(20, rtol=1e-12)
    torch.testing.assert_close(image.cpu().reshape(-1), torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_weighted_gradient_of_the_likelihood():
    rng = np.random.default_rng(5)
    matrix, data, weights = rng.normal(size=(9, 4)), rng.normal(size=9), rng.uniform(0.1, 2, 9)
    on_device = lambda x: torch.tensor(x, dtype=pytomography.dtype, device=pytomography.device)
    likelihood = NegativeMSELikelihood(MatrixSystem(matrix), on_device(data), weights=on_device(weights),
                                       scaling_constant=0.5)
    image = torch.tensor(rng.normal(size=4), device=pytomography.device).reshape(4, 1, 1)
    expected = 0.5 * matrix.T @ (weights * (data - matrix @ image.cpu().numpy().reshape(-1)))
    np.testing.assert_allclose(likelihood.compute_gradient(image).cpu().numpy().reshape(-1), expected, rtol=1e-12)


@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan"), "shape"])
def test_weights_must_be_finite_non_negative_and_shaped_like_the_projections(bad):
    weights = torch.ones(4 if bad == "shape" else 3, dtype=torch.float64)
    if bad != "shape":
        weights[1] = bad
    with pytest.raises(ValueError, match="weights"):
        problem(np.eye(3), np.ones(3), weights=weights)


def smoothed(counts, sigma):
    """The reference: a Gaussian to 3 sigma along each axis, normalised, with the edge bins repeated."""
    for axis, sig in enumerate(sigma):
        if sig > 0:
            r = int(np.ceil(3 * sig))
            k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sig ** 2))
            pad = [(0, 0)] * counts.ndim
            pad[axis] = (r, r)
            padded = np.pad(counts, pad, mode="edge")
            counts = np.apply_along_axis(lambda line: np.convolve(line, k / k.sum(), mode="valid"), axis, padded)
    return counts


def test_poisson_weights_smooth_within_each_view_and_keep_a_floor():
    rng = np.random.default_rng(11)
    counts = rng.poisson(rng.uniform(0, 6, size=(5, 16, 12))).astype(float)
    counts[:, :3] = 0   # bins with no counts
    g = torch.tensor(counts, device=pytomography.device)
    weights = poisson_weights(g, smoothing=1.5, min_counts=0.8)
    expected = 1 / np.maximum(smoothed(counts, (0, 1.5, 1.5)), 0.8)
    assert weights.shape == g.shape and weights.device == g.device and weights.dtype == g.dtype
    np.testing.assert_allclose(weights.cpu().numpy(), expected, rtol=1e-10)
    assert float(weights.max()) <= 1 / 0.8 + 1e-12
    # a sequence gives one sigma per axis; 0 everywhere uses the counts as they are
    np.testing.assert_allclose(poisson_weights(g, smoothing=(1, 0, 2)).cpu().numpy(),
                               1 / np.maximum(smoothed(counts, (1, 0, 2)), 0.5), rtol=1e-10)
    np.testing.assert_allclose(poisson_weights(g, smoothing=0, min_counts=1).cpu().numpy(),
                               1 / np.maximum(counts, 1), rtol=1e-12)


@pytest.mark.parametrize("kwargs", [{"min_counts": 0}, {"smoothing": -1}, {"smoothing": (1, 2)}])
def test_poisson_weights_reject_bad_settings(kwargs):
    with pytest.raises(ValueError):
        poisson_weights(torch.ones(2, 3, 4, dtype=torch.float64), **kwargs)


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan")])
def test_preconditioner_must_be_positive_and_finite(bad):
    _, likelihood = problem(np.eye(3), np.ones(3))
    with pytest.raises(ValueError, match="preconditioner"):
        CGLS(likelihood, preconditioner=torch.full((3, 1, 1), bad, dtype=torch.float64))


@pytest.mark.gpu
def test_gpu_images_with_cpu_projection_storage():
    old_device = pytomography.device
    pytomography.set_device(torch.device("cuda"))
    matrix = np.array([[1, 0], [0, 2], [1, 1]], dtype=float)
    system = MatrixSystem(matrix, projection_device="cpu")
    likelihood = NegativeMSELikelihood(system, torch.tensor([2.0, -2.0, 1.0], device="cpu"))
    algorithm = CGLS(likelihood)
    image = algorithm(10, rtol=1e-12)
    assert image.device.type == "cuda" and algorithm.converged
    torch.testing.assert_close(image.cpu().reshape(-1), torch.tensor([2.0, -1.0], device="cpu"))
    pytomography.set_device(old_device)
