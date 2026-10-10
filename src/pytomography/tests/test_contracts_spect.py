"""The system-matrix contract (see contracts.py) applied to the SPECT system matrix."""
from __future__ import annotations

import numpy as np
import torch

import pytomography
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform

from contracts import SystemMatrixContract

N, NA = 16, 12


def tiny_spect_system(attenuation=True, psf=True):
    dx = 0.4
    object_meta = SPECTObjectMeta([dx] * 3, (N, N, N))
    angles = list(np.linspace(0, 360, NA, endpoint=False) + 0.37)
    proj_meta = SPECTProjMeta((N, N), [dx, dx], angles, radii=[15.0 + (i % 3) for i in range(NA)])
    gen = torch.Generator().manual_seed(0)
    transforms = []
    if attenuation:
        mu = (0.15 * torch.rand(N, N, N, generator=gen)).to(pytomography.device)
        transforms.append(SPECTAttenuationTransform(attenuation_map=mu))
    if psf:
        transforms.append(SPECTPSFTransform(psf_meta=SPECTPSFMeta((0.03, 0.1))))
    H = SPECTSystemMatrix(obj2obj_transforms=transforms, proj2proj_transforms=[], object_meta=object_meta, proj_meta=proj_meta)
    f = torch.rand(N, N, N, generator=gen).to(pytomography.device)
    g = torch.rand(NA, N, N, generator=gen).to(pytomography.device)
    return H, f, g


class TestSPECTSystemMatrix(SystemMatrixContract):
    # Back projection rotates by the opposite angle with bilinear interpolation, which is the transpose of the
    # forward rotation only to interpolation accuracy: the mismatch measured on this system is 1.5e-4 to 2.5e-4
    adjoint_rtol = 1e-3

    def make(self):
        return tiny_spect_system()


class TestSPECTSystemMatrixNoTransforms(SystemMatrixContract):
    adjoint_rtol = 1e-3

    def make(self):
        return tiny_spect_system(attenuation=False, psf=False)
