"""get_smoothed_scatter takes the angle between neighbouring views whichever way the views turn: with decreasing
angles the first difference was negative, which made the kernel size along the views negative (an error), and
across 0/360 degrees it was -354 for 6 degree steps."""
from __future__ import annotations

import numpy as np
import torch

import pytomography
from pytomography.metadata.SPECT import SPECTProjMeta
from pytomography.utils.scatter import get_smoothed_scatter

SIGMAS = dict(sigma_theta=12.0, sigma_r=8.0, sigma_z=8.0)


def _smooth(scatter, angles):
    proj_meta = SPECTProjMeta((16, 12), (4.0, 4.0), list(angles), radii=[200.0] * len(angles))
    return get_smoothed_scatter(scatter.to(pytomography.device), proj_meta, **SIGMAS).cpu()


def test_smoothing_does_not_depend_on_which_way_the_views_turn():
    angles = np.arange(0, 360, 6.0)
    scatter = torch.rand(len(angles), 16, 12, generator=torch.Generator().manual_seed(0))
    increasing = _smooth(scatter, angles)
    assert (increasing - scatter).abs().max() > 0.1                       # it smooths
    decreasing = _smooth(scatter.flip(0), angles[::-1])
    assert torch.allclose(decreasing.flip(0), increasing, rtol=1e-5, atol=1e-6)
    across_zero = _smooth(scatter.roll(1, 0), np.roll(angles, 1))         # 354, 0, 6, ...
    assert torch.allclose(across_zero.roll(-1, 0), increasing, rtol=1e-5, atol=1e-6)
