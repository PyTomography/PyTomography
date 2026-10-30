"""Least-squares solutions checked independently with SciPy and the singular-value decomposition."""
from __future__ import annotations

import numpy as np
import pytest
import pytomography
import torch
from pytomography.algorithms import CGLS
from pytomography.likelihoods import NegativeMSELikelihood
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
