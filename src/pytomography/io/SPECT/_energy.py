"""Energy windows, scatter estimates from them, and collimator PSFs: the parts of SPECT reading shared by every
source."""
from __future__ import annotations

import os
import re
from typing import Sequence

import numpy as np
import torch

import pytomography
from pytomography.metadata.SPECT import EnergyWindow, SPECTPSFMeta

#: Gamma (and the main X-ray) lines of common SPECT nuclides, keV: intensity (photons per decay). Used to choose the
#: photon energy of an energy window: the strongest line inside it. Values rounded from NNDC/ENSDF.
NUCLIDE_LINES = {
    'Tc-99m': {140.5: 0.89},
    'I-123': {159.0: 0.83},
    'I-125': {35.5: 0.067, 27.5: 1.13, 31.0: 0.26},
    'I-131': {364.5: 0.815, 284.3: 0.061, 637.0: 0.072, 80.2: 0.026},
    'In-111': {171.3: 0.906, 245.4: 0.941, 23.0: 0.70},
    'Lu-177': {112.9: 0.062, 208.4: 0.104, 55.8: 0.045},
    'Ga-67': {93.3: 0.39, 184.6: 0.21, 300.2: 0.17, 393.5: 0.046},
    'Tl-201': {70.8: 0.46, 80.2: 0.16, 135.3: 0.026, 167.4: 0.10},
    'Xe-133': {81.0: 0.37},
    'Ho-166': {80.6: 0.067, 1379.4: 0.009},
    'Sm-153': {103.2: 0.29, 69.7: 0.047, 41.5: 0.49},
    'Re-186': {137.2: 0.094},
    'Re-188': {155.0: 0.15},
    'Co-57': {122.1: 0.856, 136.5: 0.107},
    'Cr-51': {320.1: 0.099},
    'Tb-161': {74.6: 0.102, 48.9: 0.17},
    'Cu-67': {184.6: 0.487, 93.3: 0.161, 91.3: 0.07},
    'Pb-203': {279.2: 0.81},
    'Pb-212': {238.6: 0.436, 300.1: 0.033},
    'Ra-223': {81.1: 0.15, 83.8: 0.25, 154.2: 0.057, 269.5: 0.139},
    'Ac-225': {218.0: 0.114, 440.5: 0.259, 99.6: 0.01},        # with its daughters Fr-221 and Bi-213
    'Bi-213': {440.5: 0.259},
    'Y-90': {},
}
_ELEMENTS = {'technetium': 'Tc', 'iodine': 'I', 'indium': 'In', 'lutetium': 'Lu', 'gallium': 'Ga', 'thallium': 'Tl',
             'xenon': 'Xe', 'holmium': 'Ho', 'samarium': 'Sm', 'rhenium': 'Re', 'cobalt': 'Co', 'chromium': 'Cr',
             'terbium': 'Tb', 'copper': 'Cu', 'lead': 'Pb', 'radium': 'Ra', 'actinium': 'Ac', 'bismuth': 'Bi',
             'yttrium': 'Y'}


def nuclide_key(text: str | None) -> str | None:
    """The nuclide named in ``text`` ('99m Technetium', 'Technetium Tc^99m^', 'Lu177_EM', 'lu177', 'In-111'...), as a
    key of :data:`NUCLIDE_LINES`, or None when it names none of them."""
    if not text:
        return None
    t = str(text).lower().replace('^', ' ')
    masses = re.findall(r'(\d{1,3})\s*(m?)\b', t) or re.findall(r'(\d{1,3})(m?)', t)
    symbols = set()
    for name, symbol in _ELEMENTS.items():
        if name in t:
            symbols.add(symbol)
    for symbol in _ELEMENTS.values():
        if re.search(rf'(?<![a-z]){symbol.lower()}(?![a-z])', t) or re.search(rf'(?<![a-z]){symbol.lower()}(?=[-_ ]?\d)', t):
            symbols.add(symbol)
    for mass, meta in masses:
        for symbol in symbols:
            key = f'{symbol}-{int(mass)}{meta}'
            if key in NUCLIDE_LINES:
                return key
    return None


def photon_energy(ranges_keV: Sequence, nuclide: str | None = None, default: float | None = None) -> float:
    """The photon energy an energy window stands for: the strongest line of ``nuclide`` inside one of its ranges;
    else ``default`` when it is inside; else the centre of the window."""
    lines = NUCLIDE_LINES.get(nuclide_key(nuclide) or '', {})
    inside = {e: i for e, i in lines.items() if any(lo <= e <= hi for lo, hi in ranges_keV)}
    if inside:
        return max(inside, key=inside.get)
    if default is not None and any(lo <= default <= hi for lo, hi in ranges_keV):
        return float(default)
    lo, hi = min(r[0] for r in ranges_keV), max(r[1] for r in ranges_keV)
    return (lo + hi) / 2


def energy_window_scatter(
    projections: torch.Tensor,
    proj_meta,
    peak: int,
    lower: int,
    upper: int | None = None,
    *,
    weighting_lower: float = 0.5,
    weighting_upper: float = 0.5,
    sigma_theta: float = 0,
    sigma_r_mm: float = 0,
    sigma_z_mm: float = 0,
    N_sigmas: int = 3,
    fov_mask: torch.Tensor | None = None,
    return_scatter_variance_estimate: bool = False,
):
    r"""Estimates the scatter in the photopeak window from the windows beside it (dual or triple energy window).

    The estimate is :math:`w_\text{peak} (\alpha_l P_l / w_l + \alpha_u P_u / w_u)`, with :math:`P` the projections
    of the lower and upper windows, :math:`w` the windows' widths in keV (from ``proj_meta.energy_windows``) and
    :math:`\alpha` the weightings; without ``upper`` (dual energy window) the upper term is left out. It can then be
    smoothed with Gaussians along θ, r and z.

    Args:
        projections (torch.Tensor): [W, θ, r, z], every energy window (as the readers give them).
        proj_meta (SPECTProjMeta): The projections' metadata, with ``energy_windows``.
        peak (int): Index of the photopeak window.
        lower (int): Index of the lower scatter window.
        upper (int, optional): Index of the upper scatter window. Defaults to None (dual energy window).
        weighting_lower (float, optional): Weight of the lower window. Defaults to 0.5.
        weighting_upper (float, optional): Weight of the upper window. Defaults to 0.5.
        sigma_theta (float, optional): Gaussian sigma of the smoothing along the views, in degrees. Defaults to 0 (no
            smoothing).
        sigma_r_mm (float, optional): Gaussian sigma of the smoothing along r, in mm. Defaults to 0.
        sigma_z_mm (float, optional): Gaussian sigma of the smoothing along z, in mm. Defaults to 0.
        N_sigmas (int, optional): Half-width of the smoothing kernels, in sigmas. Defaults to 3.
        fov_mask (torch.Tensor, optional): [r, z] mask of the detector's field of view (see :func:`fov_mask`), so that
            smoothing does not blur across its edge; the smoothed estimate is zero outside it. Defaults to None.
        return_scatter_variance_estimate (bool, optional): Also return the estimate's variance as an operator, for
            uncertainty estimates. Defaults to False.

    Returns:
        torch.Tensor | tuple[torch.Tensor, Callable]: The estimate [θ, r, z], and, with
        ``return_scatter_variance_estimate``, a function that applies its variance to a tensor.

    Raises:
        ValueError: If ``proj_meta`` has no energy windows, or ``projections`` is not [W, θ, r, z] with every window.
    """
    from pytomography.utils.scatter import compute_EW_scatter
    windows = proj_meta.energy_windows
    if not windows:
        raise ValueError('proj_meta has no energy windows')
    if projections.ndim != 4 or projections.shape[0] != len(windows):
        raise ValueError(f'projections must be [W, θ, r, z] with every one of the {len(windows)} energy windows, got '
                         f'shape {tuple(projections.shape)}')
    return compute_EW_scatter(
        projections[lower], projections[upper] if upper is not None else None,
        windows[lower].width_keV, windows[upper].width_keV if upper is not None else None, windows[peak].width_keV,
        weighting_lower, weighting_upper, proj_meta, sigma_theta, sigma_r_mm, sigma_z_mm, N_sigmas,
        return_scatter_variance_estimate, fov_mask)


def fov_mask(projections: torch.Tensor, contraction: int = 1) -> torch.Tensor:
    """The detector's field of view: the [r, z] rectangle where any projection has counts.

    Args:
        projections (torch.Tensor): Projections [..., r, z], e.g. [θ, r, z] or [W, θ, r, z]; all leading axes are
            summed.
        contraction (int, optional): Shrink the rectangle by this many pixels on each side (edge pixels can be less
            sensitive). Defaults to 1.

    Returns:
        torch.Tensor: [r, z], 1 inside the field of view and 0 outside, of the projections' type and device.
    """
    dims = len(projections.shape)
    x = projections.sum(dim=tuple(range(dims - 2)))
    r_valid = (x.sum(dim=1) > 0).to(torch.int)
    z_valid = (x.sum(dim=0) > 0).to(torch.int)
    r_min = int(r_valid.argmax()) + contraction
    r_max = int(r_valid.shape[0] - r_valid.flip(dims=(0,)).argmax() - 1) - contraction
    z_min = int(z_valid.argmax()) + contraction
    z_max = int(z_valid.shape[0] - z_valid.flip(dims=(0,)).argmax() - 1) - contraction
    mask = torch.zeros_like(x)
    mask[r_min:r_max + 1, z_min:z_max + 1] = 1
    return mask


_FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


def _mu_per_mm(material: str, photon_keV: float) -> float:
    from pytomography.utils import get_mu_from_spectrum_interp
    module_path = os.path.dirname(os.path.abspath(__file__))
    return float(get_mu_from_spectrum_interp(os.path.join(module_path, f'../../data/NIST_attenuation_data/{material}.csv'),
                                             photon_keV)) / 10


def _psf_meta(hole_diameter_mm: float, hole_length_mm: float, mu_per_mm: float, intrinsic_fwhm_mm: float,
              shape: str, min_sigmas: float) -> SPECTPSFMeta:
    """The collimator's geometric resolution D (L_eff + d) / L_eff, with L_eff = L - 2 / mu, as a Gaussian sigma (with
    the intrinsic resolution in quadrature) or as the half-width of a triangle (square holes)."""
    slope = hole_diameter_mm / (hole_length_mm - 2 / mu_per_mm)
    intercept = hole_diameter_mm
    if shape == 'gaussian':
        params = [slope * _FWHM_TO_SIGMA, intercept * _FWHM_TO_SIGMA, intrinsic_fwhm_mm * _FWHM_TO_SIGMA]
    elif shape == 'square':
        # a square hole's PSF along r (and z) is a triangle reaching zero at the geometric resolution, which is also
        # its FWHM; the intrinsic resolution is not modelled for it
        params = [slope, intercept, 0.0]
    else:
        raise ValueError(f"shape must be 'gaussian' or 'square', not {shape!r}")
    return SPECTPSFMeta(sigma_fit_params=params, sigma_fit=lambda r, a, b, c: np.sqrt((a * r + b) ** 2 + c ** 2),
                        min_sigmas=min_sigmas, shape=shape)


def collimator_psf(collimator: str, photon_keV: float, intrinsic_fwhm_mm: float = 0.0, *,
                   intrinsic_fwhm_140keV_mm: float | None = None, material: str = 'lead', shape: str = 'gaussian',
                   min_sigmas: float = 3) -> SPECTPSFMeta:
    r"""The PSF of a parallel-hole collimator, from its code in PyTomography's collimator table, at a photon energy.

    The collimator's geometric resolution (FWHM) at distance :math:`d` from it is :math:`D (L_\text{eff} + d) /
    L_\text{eff}`, with :math:`D` the hole diameter and :math:`L_\text{eff} = L - 2/\mu` the hole length shortened by
    the septal penetration of the material at ``photon_keV``. With ``shape='gaussian'`` it becomes a Gaussian sigma,
    added in quadrature to the intrinsic resolution; with ``shape='square'`` it is the half-width of a triangle (the
    intrinsic resolution is then not modelled). List the table's collimators with
    :func:`pytomography.utils.print_collimator_parameters`.

    Args:
        collimator (str): The collimator's code, e.g. ``'SY-ME'`` or ``'G8-LEHR'``.
        photon_keV (float): The photon energy, in keV (``proj_meta.energy_windows[i].photon_keV``).
        intrinsic_fwhm_mm (float, optional): The detector's intrinsic resolution (FWHM, mm) at ``photon_keV``.
            Defaults to 0.
        intrinsic_fwhm_140keV_mm (float, optional): The intrinsic resolution at 140 keV instead (most data sheets give
            that), scaled as :math:`E^{-1/2}`. Defaults to None.
        material (str, optional): The collimator's material (``'lead'``, ``'tungsten'``). Defaults to ``'lead'``.
        shape (str, optional): ``'gaussian'``, or ``'square'`` for square holes (CZT cameras such as the StarGuide),
            whose PSF along r and z is a triangle. Defaults to ``'gaussian'``.
        min_sigmas (float, optional): Kernel half-width in sigmas (Gaussian). Defaults to 3.

    Returns:
        SPECTPSFMeta: The PSF, its width as a function of the distance in mm; give it to
        :class:`~pytomography.transforms.SPECT.SPECTPSFTransform`.

    Raises:
        ValueError: If the table has no such collimator, or ``shape`` is not ``'gaussian'`` or ``'square'``.
    """
    module_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(module_path, '../../data/collim.col')) as f:
        lines = [line for line in f if line.startswith('*')]
    found = [line for line in lines if line[1:].split()[0] == collimator]
    if not found:
        raise ValueError(f"no collimator '{collimator}' in the table; see pytomography.utils.print_collimator_parameters()")
    fields = found[0][1:].split()
    hole_diameter_mm, hole_length_mm = 10 * float(fields[1]), 10 * float(fields[3])     # the table is in cm
    if intrinsic_fwhm_140keV_mm is not None:
        intrinsic_fwhm_mm = intrinsic_fwhm_140keV_mm * (photon_keV / 140) ** -0.5
    return _psf_meta(hole_diameter_mm, hole_length_mm, _mu_per_mm(material, photon_keV), intrinsic_fwhm_mm, shape,
                     min_sigmas)
