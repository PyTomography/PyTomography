"""SPECTPSFTransform takes a SPECTPSFMeta, and only that: v3's psf_operator (SPECTPSFToolbox's operators) was removed
in v4 (ARF decision 3, 14 Oct 2026), and anything else is refused with a pointer to SPECTKernelPSFTransform."""
from __future__ import annotations

import pytest

from pytomography.metadata.SPECT import SPECTPSFMeta
from pytomography.transforms.SPECT import SPECTPSFTransform


def test_the_psf_is_a_psf_meta():
    psf_meta = SPECTPSFMeta((0.03, 1.0))
    assert SPECTPSFTransform(psf_meta).psf_meta is psf_meta


def test_anything_else_points_to_the_kernel_transform():
    with pytest.raises(TypeError, match='SPECTKernelPSFTransform'):
        SPECTPSFTransform(None)
    with pytest.raises(TypeError, match='SPECTKernelPSFTransform'):
        SPECTPSFTransform(lambda input, xv, yv, distances, normalize=False: input)


def test_psf_operator_is_gone():
    with pytest.raises(TypeError):
        SPECTPSFTransform(psf_operator=lambda input, xv, yv, distances, normalize=False: input)
