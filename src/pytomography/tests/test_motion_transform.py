"""DVFMotionTransform (#229): the forward transform multiplies the deformed object by the volume change of the
deformation, |det(I + grad u_f)|, so the total activity is conserved, and the backward transform (sampling with the
inverse deformation, no Jacobian) is its adjoint.

Two deformations with known inverses: a uniform expansion about the centre (volume ratio 1/1.25^3 everywhere) and an
anisotropic bulge whose volume ratio varies from 0.85 to 1.38 across the grid (inverse by fixed-point iteration). The
transform is trilinear interpolation, so conservation and adjointness hold to interpolation accuracy, not exactly:
measured, the totals are kept within 0.5% and the two sides of the dot-product test agree within 0.45%, where without
the Jacobian the totals change by 17% to 95% and the dot products differ by 22% to 95%."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.transforms.shared import DVFMotionTransform

DEV = pytomography.device
SHAPE = (36, 32, 28)                              # not cubic, so that a swap of axes shows
CENTRE = (np.array(SHAPE) - 1) / 2
SCALE = 1.25                                      # the uniform expansion: lengths x 1.25, volumes x 1.953


def _points():
    return np.stack(np.meshgrid(*[np.arange(n, dtype=np.float64) for n in SHAPE], indexing='ij'), -1)


def _expansion():
    """Forward: the object at x is sampled at c + (x - c) / SCALE; backward: at c + SCALE (y - c)."""
    d = _points() - CENTRE
    return d * (1 / SCALE - 1), d * (SCALE - 1)


def _bulge():
    """u_f(x) = g(x) a (x - c) with an anisotropic Gaussian g; u_b(y) = psi(y) - y, psi solving psi + u_f(psi) = y."""
    a, sigma = np.array([0.35, 0.2, -0.15]), np.array([7.0, 6.0, 5.0])
    def u_f(p):
        d = p - CENTRE
        return np.exp(-0.5 * ((d / sigma) ** 2).sum(-1, keepdims=True)) * a * d
    y = _points()
    psi = y.copy()
    for _ in range(200):
        psi = y - u_f(psi)
    assert np.abs(psi + u_f(psi) - y).max() < 1e-10
    return u_f(y), psi - y


def _transform(dvfs):
    u_f, u_b = dvfs
    transform = DVFMotionTransform(torch.tensor(u_f), torch.tensor(u_b))
    transform.configure(ObjectMeta(dr=(4.0, 4.0, 4.0), shape=SHAPE), None)
    return transform


def _ball(radius):
    d = _points() - CENTRE
    return torch.tensor(np.sqrt((d ** 2).sum(-1)) <= radius, dtype=pytomography.dtype).to(DEV)


def _blob(offset, sigma):
    d = _points() - CENTRE - offset
    return torch.tensor(np.exp(-0.5 * ((d / sigma) ** 2).sum(-1)), dtype=pytomography.dtype).to(DEV)


def _without_jacobian(transform, x):
    return transform._apply_dvf(transform.dvf_forward, None, x)


@pytest.mark.parametrize('dvfs', [_expansion, _bulge], ids=['expansion', 'bulge'])
def test_volume_ratio_is_the_jacobian_determinant(dvfs):
    """The volume ratio is |det(I + grad u_f)| (checked against numpy's gradient and determinant), with u_f's
    components along x, y and z."""
    u_f, u_b = dvfs()
    ratio = _transform((u_f, u_b)).dvf_forward_vol_ratio.cpu().double().numpy()
    jacobian = np.stack([np.stack(np.gradient(u_f[..., i]), -1) for i in range(3)], -2) + np.eye(3)
    assert ratio.shape == SHAPE
    assert np.allclose(ratio, np.abs(np.linalg.det(jacobian)), atol=1e-5)
    if dvfs is _expansion:
        assert np.allclose(ratio, SCALE ** -3, atol=1e-6)


def test_uniform_object_under_expansion_conserves_activity():
    """A uniform ball expanded 1.25 times keeps its total activity; its concentration drops by 1.25^3."""
    transform = _transform(_expansion())
    ball = _ball(8.0)
    expanded = transform.forward(ball)
    assert expanded.shape == ball.shape
    assert expanded.sum().item() == pytest.approx(ball.sum().item(), rel=0.01)          # measured 0.2%
    centre = tuple(int(c) for c in CENTRE)
    assert expanded[centre].item() == pytest.approx(SCALE ** -3, rel=1e-5)
    inside = (expanded > 0.5 * SCALE ** -3).sum().item()                                      # half the new concentration
    assert inside == pytest.approx(SCALE ** 3 * ball.sum().item(), rel=0.05)
    # without the Jacobian the total grows by the volume ratio
    assert _without_jacobian(transform, ball).sum().item() == pytest.approx(SCALE ** 3 * ball.sum().item(), rel=0.01)


@pytest.mark.parametrize('obj', [lambda: _ball(8.0), lambda: _blob([2, -1, 1], [3.5, 3, 2.5])], ids=['ball', 'blob'])
def test_non_uniform_deformation_conserves_activity(obj):
    transform = _transform(_bulge())
    x = obj()
    assert transform.forward(x).sum().item() == pytest.approx(x.sum().item(), rel=0.01)          # measured < 0.5%
    assert abs(_without_jacobian(transform, x).sum().item() / x.sum().item() - 1) > 0.15


@pytest.mark.parametrize('dvfs', [_expansion, _bulge], ids=['expansion', 'bulge'])
def test_backward_is_the_adjoint_of_forward(dvfs):
    """<M a, b> = <a, M^T b> for smooth a and b (interpolation accuracy: measured within 0.45%)."""
    transform = _transform(dvfs())
    a, b = _blob([2, -1, 1], [3.5, 3, 2.5]), _blob([-1, 2, 0], [4, 3, 3.5])
    lhs = (transform.forward(a).double() * b.double()).sum().item()
    rhs = (a.double() * transform.backward(b).double()).sum().item()
    assert lhs == pytest.approx(rhs, rel=0.01)
    without = (_without_jacobian(transform, a).double() * b.double()).sum().item()
    assert abs(without / rhs - 1) > 0.2


def test_none_is_the_identity():
    transform = DVFMotionTransform(None, None)
    x = _ball(5.0)
    assert transform.forward(x) is x and transform.backward(x) is x
