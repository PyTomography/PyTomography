"""Hounsfield units to linear attenuation coefficients (mu, 1/mm) at a photon energy, and the water check of mu maps.

One module for every modality (before PyTomography 4 there were two copies of the HU to mu conversion, in
``io.SPECT.dicom`` and ``io.SPECT.attenuation_map``). A CT's HU become mu at the photon energy along a bilinear curve,
one line below 0 HU (air to water) and one above (water to bone). Its coefficients come from

* ``'table'``: PyTomography's table of measured coefficients (``data/ct_table.csv``), for the CT's model and kVp and
  the photon energy nearest ``energy_keV``. When the table has no entry within 5 % of ``energy_keV`` (it stops at
  440 keV), the ``'nist'`` curve is used, with a warning.
* ``'cortical_bone_fit'``: the CT's effective energy found from its cortical bone peak (1200 to 1600 HU), and the
  curve through air, water and cortical bone at that energy (NIST), taken to ``energy_keV``. Without a peak, the
  effective energy is 75 % of the kVp, with a warning.
* ``'nist'``: that curve with the effective energy 75 % of the kVp.

A mu map read from a file is checked against water at ``energy_keV``: :func:`check_water` warns when the map's water
peak differs from NIST's water by more than 5 %, which catches a map made for another photon energy or in other
units.
"""
from __future__ import annotations

import functools
import os
import warnings

import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_NIST = os.path.join(_DATA, "NIST_attenuation_data")
TECHNIQUES = ("table", "cortical_bone_fit", "nist")
#: units of linear attenuation coefficients, and their factor to 1/mm
MU_UNITS = {"1/mm": 1.0, "mm^-1": 1.0, "mm-1": 1.0, "/mm": 1.0, "1/cm": 0.1, "cm^-1": 0.1, "cm-1": 0.1, "/cm": 0.1,
            "1/m": 0.001, "m^-1": 0.001}
#: how far a mu map's water may be from NIST's at the photon energy before :func:`check_water` warns
WATER_TOLERANCE = 0.05


def mu_unit_factor(units: str | None) -> float:
    """The factor that takes linear attenuation coefficients in ``units`` to 1/mm."""
    key = None if units is None else str(units).strip().lower().replace(" ", "")
    if key not in MU_UNITS:
        raise ValueError(f"{units!r} are not units of linear attenuation; give units= as one of {sorted(MU_UNITS)}")
    return MU_UNITS[key]


@functools.lru_cache(maxsize=None)
def _nist_fit(material: str) -> np.ndarray:
    """The fit of NIST's linear attenuation (1/cm) between 50 and 550 keV that PyTomography has always used."""
    from scipy.optimize import curve_fit
    from pytomography.utils.nist_data import dual_sqrt_exponential, get_E_mu_data_from_datasheet
    energy, mu = get_E_mu_data_from_datasheet(os.path.join(_NIST, f"{material}.csv"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return curve_fit(dual_sqrt_exponential, energy, mu)[0]


def nist_mu(material: str, energy_keV) -> np.ndarray | float:
    """NIST's linear attenuation coefficient of a material (``'water'``, ``'air'``, ``'bonecortical'``...; a file of
    ``data/NIST_attenuation_data``) at photon energies between 50 and 550 keV, in 1/mm."""
    from pytomography.utils.nist_data import dual_sqrt_exponential
    return dual_sqrt_exponential(np.asarray(energy_keV, dtype=np.float64), *_nist_fit(material)) / 10


def nist_hu(material: str, energy_keV: float) -> float:
    """The Hounsfield units of a material at a (CT effective) energy, from NIST."""
    water, air = nist_mu("water", energy_keV), nist_mu("air", energy_keV)
    return float((nist_mu(material, energy_keV) - water) / (water - air) * 1000)


def bilinear(hu: np.ndarray, a1: float, b1: float, a2: float, b2: float) -> np.ndarray:
    """mu = a1 HU + b1 below 0 HU and a2 HU + b2 from 0 HU up, and never below 0 (float32)."""
    hu = np.asarray(hu, dtype=np.float32)
    mu = np.where(hu < 0, np.float32(a1) * hu + np.float32(b1), np.float32(a2) * hu + np.float32(b2))
    return np.maximum(mu, np.float32(0)).astype(np.float32)


def _normalised_model(name) -> str:
    return str(name or "").replace(" ", "").replace("-", "").lower()


def table_coefficients(energy_keV: float, kvp: float | None, model: str | None) -> tuple | None:
    """Bilinear coefficients (a1, b1, a2, b2), mu in 1/mm, from PyTomography's table: the scanner model's rows (all rows,
    with a warning, when the model is not in the table), the photon energy nearest ``energy_keV`` and then the kVp
    nearest ``kvp``. None when no photon energy of the table is within 5 % of ``energy_keV``."""
    import csv
    with open(os.path.join(_DATA, "ct_table.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    candidates = [r for r in rows if _normalised_model(r["Model Name"]) == _normalised_model(model)]
    if not candidates:
        warnings.warn(f"the CT's model {model!r} is not in PyTomography's table of HU to mu coefficients; the closest "
                      "configuration of another model is used", stacklevel=3)
        candidates = rows
    kvp = float(kvp) if kvp is not None else 120.0
    best = min(candidates, key=lambda r: (abs(float(r["Energy"]) - energy_keV), abs(float(r["CT Energy"]) - kvp)))
    if abs(float(best["Energy"]) - energy_keV) > 0.05 * energy_keV:
        return None
    a1, b1, a2, b2 = (float(best[k]) / 10 for k in ("Lower Coeff", "Lower Constant", "Upper Coeff", "Upper Constant"))
    return a1, b1, a2, b2


def cortical_bone_peak(hu: np.ndarray) -> float | None:
    """The HU of cortical bone in a CT, from the peak of the histogram's Laplacian between 1200 and 1600 HU; None when
    there is none (a phantom without bone)."""
    from scipy.signal import find_peaks
    x = np.asarray(hu).ravel()
    x = x[(x > 1200) & (x < 1600)]
    if x.size < 10:
        return None
    counts, edges = np.histogram(x, bins=10, density=True)
    centres = edges[:-1] + np.diff(edges)[0] / 2
    peaks, _ = find_peaks(np.gradient(np.gradient(counts)), prominence=8e-5)
    return float(centres[peaks[-1]]) if len(peaks) else None


def effective_ct_energy(bone_hu: float) -> float:
    """The CT energy (keV) at which NIST's cortical bone has ``bone_hu`` Hounsfield units."""
    from scipy.optimize import minimize
    f = lambda e: 100 * (nist_mu("bonecortical", e[0]) * 10 - (bone_hu / 1000 * (nist_mu("water", e[0]) - nist_mu("air", e[0])) + nist_mu("water", e[0])) * 10) ** 2
    return float(minimize(f, x0=(115,), method="SLSQP").x[0])


def nist_coefficients(energy_keV: float, ct_energy_keV: float) -> tuple:
    """Bilinear coefficients (a1, b1, a2, b2), mu in 1/mm at ``energy_keV``, through air, water and cortical bone,
    whose HU are NIST's at the CT's effective energy."""
    hu = [nist_hu(m, ct_energy_keV) for m in ("air", "water", "bonecortical")]
    mu = [float(nist_mu(m, energy_keV)) for m in ("air", "water", "bonecortical")]
    a1 = (mu[1] - mu[0]) / (hu[1] - hu[0])
    a2 = (mu[2] - mu[1]) / (hu[2] - hu[1])
    return a1, mu[1], a2, mu[1]


def hu_to_mu(hu: np.ndarray, energy_keV: float, *, technique="table", kvp: float | None = None,
             model: str | None = None) -> np.ndarray:
    """Linear attenuation coefficients (1/mm, float32) at ``energy_keV`` of a CT in HU.

    Args:
        hu (np.ndarray): the CT, in HU.
        energy_keV (float): the photon energy.
        technique (str | Callable): ``'table'``, ``'cortical_bone_fit'`` or ``'nist'`` (see the module), or a function
            that takes the HU array and returns mu in 1/mm.
        kvp (float, optional): the CT's kVp (DICOM KVP).
        model (str, optional): the CT's model (DICOM ManufacturerModelName), for the table.

    Returns:
        np.ndarray: mu in 1/mm.
    """
    if callable(technique):
        return np.asarray(technique(hu), dtype=np.float32)
    if technique not in TECHNIQUES:
        raise ValueError(f"technique must be one of {TECHNIQUES} or a function of the HU, not {technique!r}")
    energy_keV = float(energy_keV)
    if technique == "table":
        coefficients = table_coefficients(energy_keV, kvp, model)
        if coefficients is not None:
            return bilinear(hu, *coefficients)
        warnings.warn(f"PyTomography's table of HU to mu coefficients has no photon energy near {energy_keV} keV: the "
                      "curve through air, water and cortical bone from NIST is used (technique='nist')", stacklevel=3)
        technique = "nist"
    if kvp is None:
        raise ValueError("the CT's kVp is needed (the file has no KVP)")
    ct_energy = 0.75 * float(kvp)
    if technique == "cortical_bone_fit":
        peak = cortical_bone_peak(hu)
        if peak is None:
            warnings.warn("no cortical bone peak in the CT: its effective energy is taken as 75 % of its kVp",
                          stacklevel=3)
        else:
            ct_energy = effective_ct_energy(peak)
    return bilinear(hu, *nist_coefficients(energy_keV, ct_energy))


def water_peak(mu_mm: np.ndarray, energy_keV: float) -> float | None:
    """The most common value of a mu map (1/mm) between a fifth and three times NIST's water at ``energy_keV``: its
    water (or soft tissue); None when it has almost no voxels there."""
    water = float(nist_mu("water", energy_keV))
    x = np.asarray(mu_mm, dtype=np.float32).ravel()
    x = x[(x > 0.2 * water) & (x < 3 * water)]
    if x.size < 100:
        return None
    counts, edges = np.histogram(x, bins=np.geomspace(0.2 * water, 3 * water, 271))   # 1 % bins
    k = int(np.argmax(counts))
    return float(np.sqrt(edges[k] * edges[k + 1]))


def check_water(mu_mm: np.ndarray, energy_keV: float, source: str = "") -> None:
    """Warns when a mu map's water peak is more than 5 % from NIST's water at ``energy_keV``, or when the map has
    almost no voxels near water's value (other units than stated?)."""
    water = float(nist_mu("water", energy_keV))
    peak = water_peak(mu_mm, energy_keV)
    where = f" ({source})" if source else ""
    if peak is None:
        warnings.warn(f"the mu map{where} has almost no voxels near water's {water:.5f}/mm at {energy_keV} keV: are its "
                      "units right (units=), and is it for this photon energy?", stacklevel=3)
    elif abs(peak / water - 1) > WATER_TOLERANCE:
        warnings.warn(f"the mu map{where} has water at {peak:.5f}/mm, {100 * (peak / water - 1):+.0f} % from NIST's "
                      f"{water:.5f}/mm at {energy_keV} keV: is it made for another photon energy, or in other units?",
                      stacklevel=3)
