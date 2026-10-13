"""SPECTPSFTransform takes exactly one PSF model, psf_meta or psf_operator: it built the Exception for two or none
without raising it, so psf_meta silently won over psf_operator, and with neither the transform failed later, when
configured."""
from __future__ import annotations

import pytest

from pytomography.metadata.SPECT import SPECTPSFMeta
from pytomography.transforms.SPECT import SPECTPSFTransform


def _operator(input, xv, yv, a, normalize=False):
    return input


def test_exactly_one_psf_model():
    psf_meta = SPECTPSFMeta((0.03, 1.0))
    assert SPECTPSFTransform(psf_meta).psf_meta is psf_meta
    assert SPECTPSFTransform(psf_operator=_operator).psf_operator is _operator
    with pytest.raises(ValueError, match='exactly one'):
        SPECTPSFTransform(psf_meta, psf_operator=_operator)
    with pytest.raises(ValueError, match='exactly one'):
        SPECTPSFTransform()
