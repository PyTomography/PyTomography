"""The single scatter simulation weights each scattered photon by the probability that the detector registers it: that
its measured energy, Gaussian about its true energy, falls in the energy window. The window and the resolution come
from the scanner's ``info`` (``energy_window_low``, ``energy_window_high`` in keV, ``energy_resolution`` as the FWHM
at 511 keV over 511 keV), as the PETSIRD reader gives them, and each falls back on its own to the values the
simulation always used (430 keV, no upper limit, 0.15).

These tests check that without the keys the estimate is the one from before, bit for bit; that a narrower window
lowers it; that the window's probability is the integral of the Gaussian over the window; and that the keys fall back
one by one. The estimates here compute their line integrals with a small PyTorch sampler instead of parallelproj, so
they run on any device: they test the energy factor, which multiplies whatever the integrals are."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad

import pytomography

pytest.importorskip("parallelproj_core", reason="the scatter simulation requires parallelproj 2")
from pytomography.metadata import ObjectMeta  # noqa: E402
from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta  # noqa: E402
from pytomography.utils import sss  # noqa: E402
from pytomography.io.PET.scanner import scanner_info  # noqa: E402

DEV = pytomography.device
INFO = scanner_info(min_rsector_difference=0, crystal_length=20.0, radius=120.0, rsector0_angle=90.0, transaxial_direction=-1, axial_direction=-1,
            rsectorTransNr=16, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=2, moduleTransSpacing=0.0,
            moduleAxialSpacing=17.0, submoduleTransNr=1, submoduleAxialNr=1, submoduleTransSpacing=0.0,
            submoduleAxialSpacing=0.0, crystalTransNr=4, crystalAxialNr=4, crystalTransSpacing=4.0,
            crystalAxialSpacing=4.0, NrCrystalsPerRing=64, NrRings=8)
OBJECT_META = ObjectMeta(dr=(4, 4, 4), shape=(32, 32, 6))
STEPS = (2, 0.004, 4, 4)     # image step, attenuation cutoff, inter-ring step, intra-ring step
TOF_META = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3)


def _efficiency_before(scatter_energy: torch.Tensor) -> torch.Tensor:
    """``detector_efficiency`` as it was before the energy window came from ``info`` (development da18fde), with the
    defaults the scatter simulation called it with."""
    energy_resolution, energy_threshhold = 0.15, 430
    sigma = 511 * energy_resolution / (2*np.sqrt(2*np.log(2)))
    return 0.5 * (1 - torch.erf((energy_threshhold-scatter_energy) / (np.sqrt(2) * sigma)))


def _sampled_line_integrals(lor_start, lor_end, image, object_origin, voxel_size, samples=64):
    """Integrals of ``image`` along the segments from ``lor_start`` to ``lor_end``, from the voxel nearest to each of
    ``samples`` evenly spaced points: a stand-in for parallelproj's projector, on the image's device."""
    device = image.device
    start, end = lor_start.to(device, torch.float32), lor_end.to(device, torch.float32)
    t = (torch.arange(samples, device=device, dtype=torch.float32) + 0.5) / samples
    points = start.unsqueeze(1) + t.reshape(1, -1, 1) * (end - start).unsqueeze(1)
    index = torch.round((points - object_origin) / voxel_size).long()
    inside = ((index >= 0) & (index < torch.tensor(image.shape, device=device))).all(dim=-1)
    index = index * inside.unsqueeze(-1)
    values = image[index[..., 0], index[..., 1], index[..., 2]] * inside
    return values.sum(dim=1) * (end - start).norm(dim=1) / samples


@pytest.fixture
def sampled_integrals(monkeypatch):
    monkeypatch.setattr(sss, "_line_integrals", _sampled_line_integrals)


def _phantom():
    """Water cylinder of radius 40 mm with uniform activity and a hot sphere: (activity, attenuation) on the device."""
    x, y, z = torch.meshgrid(*[(torch.arange(n) - n / 2 + 0.5) * d for n, d in zip(OBJECT_META.shape, OBJECT_META.dr)], indexing="ij")
    cylinder = (x**2 + y**2 < 40**2).float()
    sphere = (((x - 15)**2 + (y + 10)**2 + z**2) < 12**2).float()
    return (cylinder + 3 * sphere).to(DEV), (0.0096 * cylinder).to(DEV)


def _estimate(tof: bool, **energy_keys) -> torch.Tensor:
    """Weights of the sparse scatter estimate of the phantom, with ``energy_keys`` added to the scanner's info."""
    activity, attenuation = _phantom()
    info = dict(INFO, **energy_keys)
    proj_meta = PETLMProjMeta(torch.zeros((1, 3 if tof else 2), dtype=torch.long), info, tof_meta=TOF_META if tof else None)
    torch.manual_seed(0)      # the scatter points' random offsets in their voxels
    if tof:
        return sss.compute_sss_sparse_sinogram_TOF(OBJECT_META, proj_meta, activity, attenuation, TOF_META, *STEPS).weights
    return sss.compute_sss_sparse_sinogram(OBJECT_META, proj_meta, activity, attenuation, *STEPS).weights


def test_default_efficiency_is_the_formula_before_bit_for_bit():
    """Scattered 511 keV photons have 170 to 511 keV; the defaults (and an upper limit of None or infinity) give the
    efficiency of before, bit for bit, in float32 and float64."""
    for dtype in (torch.float32, torch.float64):
        energy = torch.linspace(150, 560, 10001, dtype=dtype)
        before = _efficiency_before(energy)
        assert torch.equal(sss.detector_efficiency(energy), before)
        assert torch.equal(sss.detector_efficiency(energy, 0.15, 430.0, None), before)
        assert torch.equal(sss.detector_efficiency(energy, 0.15, 430.0, math.inf), before)


@pytest.mark.parametrize("tof", [False, True])
def test_without_energy_keys_the_estimate_is_unchanged_bit_for_bit(tof, sampled_integrals, monkeypatch):
    """GATE and GE data have no energy keys in their info: their estimate is the one from before, bit for bit; so is
    the estimate with the defaults written into info."""
    without_keys = _estimate(tof)
    with_defaults = _estimate(tof, energy_window_low=430, energy_window_high=None, energy_resolution=0.15)
    monkeypatch.setattr(sss, "detector_efficiency", lambda energy, *args, **kwargs: _efficiency_before(energy))
    before = _estimate(tof)
    assert before.abs().sum() > 0
    assert torch.equal(without_keys, before)
    assert torch.equal(with_defaults, before)


@pytest.mark.parametrize("tof", [False, True])
def test_a_narrower_window_lowers_the_estimate(tof, sampled_integrals):
    """Every weight is a product of non-negative factors and the detection probability, which falls as the lower
    limit rises or as an upper limit comes down; so each weight does not rise and their sum falls."""
    totals = []
    previous = None
    for low in (410, 430, 450, 480):
        estimate = _estimate(tof, energy_window_low=low)
        if previous is not None:
            assert (estimate <= previous).all()
        previous = estimate
        totals.append(estimate.sum().item())
    assert totals == sorted(totals, reverse=True) and len(set(totals)) == len(totals)
    without_upper = _estimate(tof)
    with_upper = _estimate(tof, energy_window_high=560)
    assert (with_upper <= without_upper).all() and with_upper.sum() < without_upper.sum()


def test_window_probability_is_the_integral_of_the_gaussian():
    """The probability of a measured energy in (low, high): the Gaussian about the photon's energy, with the FWHM
    ``energy_resolution`` x 511 keV at every energy, integrated numerically over the window."""
    resolution, low, high = 0.12, 425.0, 650.0
    sigma = resolution * 511 / (2 * math.sqrt(2 * math.log(2)))
    def gaussian_integral(energy, a, b):
        pdf = lambda x: math.exp(-(x - energy)**2 / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
        # quad's adaptive steps can miss a narrow peak on a long or open interval: split there at the peak
        points = [p for p in (energy,) if a < p < b]
        edges = [a, *points, b]
        return sum(quad(pdf, lo, hi, epsabs=1e-14, epsrel=1e-12)[0] for lo, hi in zip(edges[:-1], edges[1:]))
    energies = [200.0, 350.0, 400.0, 425.0, 450.0, 480.0, 511.0, 600.0, 650.0, 700.0]
    computed = sss.detector_efficiency(torch.tensor(energies, dtype=torch.float64), resolution, low, high)
    expected = torch.tensor([gaussian_integral(e, low, high) for e in energies], dtype=torch.float64)
    assert torch.allclose(computed, expected, rtol=0, atol=1e-11)
    # without an upper limit, the integral up to infinity
    computed = sss.detector_efficiency(torch.tensor(energies, dtype=torch.float64), resolution, low)
    expected = torch.tensor([gaussian_integral(e, low, math.inf) for e in energies], dtype=torch.float64)
    assert torch.allclose(computed, expected, rtol=0, atol=1e-11)


def test_energy_keys_fall_back_one_by_one(sampled_integrals):
    window = sss.detector_energy_window
    assert window({}) == (430.0, None, 0.15)
    assert window(dict(INFO)) == (430.0, None, 0.15)
    assert window({"energy_window_low": 450}) == (450.0, None, 0.15)
    assert window({"energy_window_high": 650}) == (430.0, 650.0, 0.15)
    assert window({"energy_resolution": 0.12}) == (430.0, None, 0.12)
    assert window({"energy_window_low": None, "energy_window_high": 600.0}) == (430.0, 600.0, 0.15)
    assert window({"energy_window_low": 425, "energy_window_high": 650, "energy_resolution": np.float32(0.1)}) == \
        (425.0, 650.0, float(np.float32(0.1)))
    # the estimate reads each key on its own
    partial = _estimate(False, energy_window_low=450)
    assert torch.equal(partial, _estimate(False, energy_window_low=450, energy_window_high=None, energy_resolution=0.15))
    assert not torch.equal(partial, _estimate(False))
    # a resolution in percent, or a window upside down, is an error rather than a wrong estimate
    for bad in ({"energy_resolution": 15}, {"energy_resolution": 0}, {"energy_window_low": 500, "energy_window_high": 450}):
        with pytest.raises(ValueError):
            window(bad)


def test_the_petsird_standard_form_gives_the_window_the_scatter_simulation_reads():
    """The PETSIRD reader puts the file's energy window and resolution in info under the keys SSS reads."""
    pytest.importorskip("petsird")
    from petsird.helpers import generator
    from pytomography.io.PET import petsird as reader
    module = generator.CylindricalBlocksInfo(
        radius=200, crystal_length=(20, 4, 5), num_crystals_per_module=(1, 4, 3), num_modules_along_ring=12,
        num_modules_along_axis=2, arc=2 * math.pi, start_angle=0, module_spacing_along_axis=20., number_of_tof_bins=5,
        tof_resolution=60., LLD=425, ULD=650, number_of_event_energy_bins=1, energy_resolution=0.12, material_id=1)
    info = reader._detector_info(generator.get_scanner_info([module]))
    low, high, resolution = sss.detector_energy_window(info)
    assert (low, high) == (425.0, 650.0) and math.isclose(resolution, 0.12, rel_tol=1e-6)
