"""PSF2D convolves an object with the 2D kernel its operator provides. When the operator exposes that kernel,
the convolution is done as a depthwise FFT convolution instead of going through the operator's own grouped
convolution. These tests check that the result matches the operator, that the fast path is only taken when it
agrees, that caching the kernel changes nothing, and that a changed object shape is handled."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn.functional import conv2d, pad

import pytomography
from pytomography.transforms.SPECT.psf import PSF2D

DEV = pytomography.device
LX, LY, LZ, NK = 24, 20, 16, 11
DR = (0.4, 0.4, 0.4)


class _OpaqueOperator:
    """Stand-in for an operator that convolves internally and does not expose its kernel."""
    def __init__(self, kernels, scale=1.0):
        self.kernels = kernels                      # [LX, NK, NK]
        self.scale = scale

    def __call__(self, input, xv, yv, a, normalize=False):
        return self.scale * conv2d(input.unsqueeze(0), self.kernels.unsqueeze(1), padding='same',
                                   groups=input.shape[0]).squeeze()


class _KernelOperator(_OpaqueOperator):
    """Stand-in for a SPECTPSFToolbox 2D operator that exposes the kernel of every plane."""
    def _get_kernel(self, xv, yv, a):
        return self.kernels


def _kernels(seed=0):
    gen = torch.Generator().manual_seed(seed)
    k = torch.rand(LX, NK, NK, generator=gen)
    k[:, NK // 2, NK // 2] += 5.0
    return (k / k.sum(dim=(1, 2), keepdim=True)).to(DEV)


def _layer(op, cache_kernel=False):
    distances = np.linspace(0.0, 10.0, LX)
    return PSF2D(op, distances, NK, DR, cache_kernel=cache_kernel)


@pytest.mark.parametrize("cache_kernel", [False, True])
def test_matches_operator_and_uses_fast_path(cache_kernel):
    gen = torch.Generator().manual_seed(1)
    obj = torch.rand(LX, LY, LZ, generator=gen).to(DEV)
    op = _KernelOperator(_kernels())
    layer = _layer(op, cache_kernel)
    out = layer(obj)
    assert layer._use_fast_conv is True
    assert (layer._kernel_spectrum is not None) == cache_kernel
    reference = op(obj, layer.xv, layer.yv, layer.distances, normalize=True)
    assert out.shape == reference.shape
    assert (out - reference).abs().max() < 1e-4 * reference.abs().max()


def test_cached_and_uncached_give_the_same_result():
    gen = torch.Generator().manual_seed(2)
    obj = torch.rand(LX, LY, LZ, generator=gen).to(DEV)
    kernels = _kernels()
    a = _layer(_KernelOperator(kernels), cache_kernel=False)(obj)
    layer = _layer(_KernelOperator(kernels), cache_kernel=True)
    b = layer(obj)
    c = layer(obj)                                   # second call reuses the cached spectrum
    assert torch.equal(a, b) and torch.equal(b, c)


def test_falls_back_when_operator_does_not_expose_a_kernel():
    gen = torch.Generator().manual_seed(3)
    obj = torch.rand(LX, LY, LZ, generator=gen).to(DEV)
    op = _OpaqueOperator(_kernels())
    layer = _layer(op)
    out = layer(obj)
    assert layer._use_fast_conv is False
    assert torch.equal(out, op(obj, layer.xv, layer.yv, layer.distances, normalize=True))


def test_falls_back_when_the_kernel_does_not_describe_the_operator():
    """An operator whose output is not the plain convolution of its advertised kernel must not take the fast path."""
    gen = torch.Generator().manual_seed(4)
    obj = torch.rand(LX, LY, LZ, generator=gen).to(DEV)
    op = _KernelOperator(_kernels(), scale=2.7)       # applies an extra factor the kernel does not account for
    layer = _layer(op)
    out = layer(obj)
    assert layer._use_fast_conv is False
    assert torch.equal(out, op(obj, layer.xv, layer.yv, layer.distances, normalize=True))


@pytest.mark.parametrize("cache_kernel", [False, True])
def test_handles_a_change_of_object_shape(cache_kernel):
    gen = torch.Generator().manual_seed(5)
    op = _KernelOperator(_kernels())
    layer = _layer(op, cache_kernel)
    for shape in ((LX, LY, LZ), (LX, LY + 6, LZ + 4), (LX, LY, LZ)):
        obj = torch.rand(*shape, generator=gen).to(DEV)
        out = layer(obj)
        reference = op(obj, layer.xv, layer.yv, layer.distances, normalize=True)
        assert out.shape == reference.shape
        assert (out - reference).abs().max() < 1e-4 * reference.abs().max()


def test_even_sized_last_axis_is_handled():
    """The one sided FFT needs an even final dimension; odd sizes are padded and cropped back."""
    gen = torch.Generator().manual_seed(6)
    op = _KernelOperator(_kernels())
    layer = _layer(op)
    for lz in (15, 16):
        obj = torch.rand(LX, LY, lz, generator=gen).to(DEV)
        out = layer(obj)
        reference = op(obj, layer.xv, layer.yv, layer.distances, normalize=True)
        assert out.shape == reference.shape
        assert (out - reference).abs().max() < 1e-4 * reference.abs().max()
