"""PyTomography 3's argument names that changed in 4 (Gaussians as FWHM, no unit suffixes) still work until 5.0:
with a DeprecationWarning, and a sigma becomes the FWHM of the same Gaussian."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pytomography
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.transforms.shared import GaussianFilter
from pytomography.utils._renamed import FWHM_PER_SIGMA
from pytomography.utils.scatter import compute_EW_scatter, get_smoothed_scatter


def _projections():
    proj_meta = SPECTProjMeta((16, 12), (4.0, 4.0), np.linspace(0, 360, 24, endpoint=False), radii=np.full(24, 200.0))
    g = torch.Generator().manual_seed(0)
    return torch.rand((24, 16, 12), generator=g).to(pytomography.device), proj_meta


def test_scatter_smoothing_takes_fwhm_and_the_old_sigmas_with_a_warning():
    scatter, proj_meta = _projections()
    new = get_smoothed_scatter(scatter, proj_meta, fwhm_theta=2 * FWHM_PER_SIGMA, fwhm_r=3 * FWHM_PER_SIGMA,
                               fwhm_z=1.5 * FWHM_PER_SIGMA)
    with pytest.warns(DeprecationWarning, match="'sigma_r' is now 'fwhm_r'.*FWHM in mm"):
        old = get_smoothed_scatter(scatter, proj_meta, sigma_theta=2, sigma_r=3, sigma_z=1.5)
    torch.testing.assert_close(old, new)
    with pytest.warns(DeprecationWarning, match="'sigma_theta' is now 'fwhm_theta'"):
        old = compute_EW_scatter(scatter, scatter, 10.0, 10.0, 20.0, proj_meta=proj_meta, sigma_theta=2)
    torch.testing.assert_close(old, compute_EW_scatter(scatter, scatter, 10.0, 10.0, 20.0, proj_meta=proj_meta,
                                                       fwhm_theta=2 * FWHM_PER_SIGMA))


def test_the_fwhm_is_the_gaussians_full_width_at_half_maximum():
    # A single bright view, smoothed along the views: half its peak height is reached FWHM / 2 away
    scatter, proj_meta = _projections()
    spike = torch.zeros_like(scatter)
    spike[12] = 1.0
    smoothed = get_smoothed_scatter(spike, proj_meta, fwhm_theta=60.0)[:, 8, 6]   # 60 degrees = 4 views of 15
    profile = smoothed / smoothed.max()
    assert abs(profile[12 + 2].item() - 0.5) < 0.02 and abs(profile[12 - 2].item() - 0.5) < 0.02


def test_old_and_new_names_together_are_refused():
    scatter, proj_meta = _projections()
    with pytest.raises(TypeError, match="both 'sigma_r' and its new name 'fwhm_r'"):
        get_smoothed_scatter(scatter, proj_meta, sigma_r=1, fwhm_r=2)


def test_gaussian_filter_takes_fwhm_and_the_old_FWHM_with_a_warning():
    object_meta = SPECTObjectMeta((4.0, 4.0, 4.0), (16, 16, 16))
    proj_meta = SPECTProjMeta((16, 16), (4.0, 4.0), np.array([0.0]), radii=np.array([200.0]))
    x = torch.rand(16, 16, 16, generator=torch.Generator().manual_seed(1))
    new = GaussianFilter(fwhm=10.0)
    with pytest.warns(DeprecationWarning, match="'FWHM' is now 'fwhm'"):
        old = GaussianFilter(FWHM=10.0)
    for f in (new, old):
        f.configure(object_meta, proj_meta)
    x = x.to(pytomography.device)
    torch.testing.assert_close(old.forward(x), new.forward(x))


def test_randoms_smoothing_takes_fwhm_and_the_old_sigmas_with_a_warning():
    from pytomography.io.PET import shared
    pytest.importorskip("parallelproj_core")
    info = dict(crystalTransNr=4, crystalAxialNr=2, moduleTransNr=1, moduleAxialNr=1, rsectorTransNr=8,
                crystalTransSpacing=4.0, crystalAxialSpacing=4.0, radius=100.0)
    from pytomography.io.PET import scanner_info
    info = scanner_info(info)
    from pytomography.io.PET.shared import sinogram_coordinates
    _, index = sinogram_coordinates(info)
    randoms = torch.rand((16, 9, int(index.numel())), generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    new = shared.smooth_randoms_sinogram(randoms, info, fwhm_r=4 * FWHM_PER_SIGMA, fwhm_theta=4 * FWHM_PER_SIGMA,
                                         fwhm_z=4 * FWHM_PER_SIGMA)
    with pytest.warns(DeprecationWarning, match="'sigma_z' is now 'fwhm_z'.*FWHM in rings"):
        old = shared.smooth_randoms_sinogram(randoms, info, sigma_r=4, sigma_theta=4, sigma_z=4)
    torch.testing.assert_close(old, new)


def test_cardiac_angles_take_the_old_names_with_a_warning():
    from pytomography.utils import cardiac_spect
    image = torch.zeros(32, 32)
    image[10:22, 14:18] = 1
    angles = torch.arange(0, 180, 5.0)
    with pytest.warns(DeprecationWarning, match="'angle_range_degrees' is now 'angle_range'"):
        old = cardiac_spect.get_angle(image, angle_range_degrees=angles)
    assert old == cardiac_spect.get_angle(image, angle_range=angles)
