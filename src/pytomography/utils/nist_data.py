"""NIST's attenuation data: reading its tables of mass attenuation coefficients, and the fit PyTomography uses.

The tables are the files of ``pytomography/data/NIST_attenuation_data`` (one per material, e.g. ``water.csv``): the
density in g/cm³ on the first line, then rows of photon energy (MeV), mass attenuation coefficient (cm²/g) and mass
energy absorption coefficient (cm²/g). The functions here keep NIST's unit, 1/cm, for the linear attenuation
coefficient; the rest of PyTomography divides it by 10 where it uses it, for 1/mm.
"""
from __future__ import annotations
import warnings
import numpy as np
import re
from scipy.optimize import curve_fit
np.seterr(all="ignore")

def dual_sqrt_exponential(
    energy: float,
    c1: float,
    c2: float,
    d1: float,
    d2: float
    ) -> float:
    r"""The function fitted to NIST's linear attenuation coefficients against photon energy.

    It is :math:`f(E) = c_1e^{-d_1\sqrt{E}} + c_2e^{-d_2\sqrt{E}}`, chosen only because it fits the data well between 50
    and 550 keV.

    Args:
        energy (float): Photon energy, in keV (a NumPy array also works).
        c1 (float): Fit parameter :math:`c_1`.
        c2 (float): Fit parameter :math:`c_2`.
        d1 (float): Fit parameter :math:`d_1`.
        d2 (float): Fit parameter :math:`d_2`.

    Returns:
        float: :math:`f(E)`, in the units of the data fitted (1/cm for :func:`get_mu_from_spectrum_interp`).
    """
    return c1*np.exp(-d1*np.sqrt(energy)) + c2*np.exp(-d2*np.sqrt(energy))

def get_E_mu_data_from_datasheet(file: str) -> tuple[np.array, np.array]:
    """Reads the photon energies and linear attenuation coefficients between 50 and 550 keV from a NIST table.

    The linear attenuation coefficient is the mass attenuation coefficient times the density on the table's first line.

    Args:
        file (str): Path of the table of one material (see the module's description).

    Returns:
        tuple[np.ndarray, np.ndarray]: The energies strictly between 50 and 550 keV, in keV, and the linear attenuation
        coefficients at them, in 1/cm.
    """
    with open(file) as f:
        lines = f.readlines()
        rho = float(lines[0])
        lines = lines[1:]
    for i in range(len(lines)):
        lines[i] = re.split(r'\s+', lines[i])[-4:-1]
    E, mu_rho, _ = np.array(lines).astype(float).T
    E*=1000
    mu = mu_rho * rho
    idx = (E>50)*(E<550)
    return E[idx], mu[idx]

def get_mu_from_spectrum_interp(
    file: str,
    energy: float
    ) -> np.array:
    """Linear attenuation coefficient of a material at a photon energy, from the fit to its NIST table.

    :func:`dual_sqrt_exponential` is fitted to the table's values between 50 and 550 keV
    (:func:`get_E_mu_data_from_datasheet`) and evaluated at ``energy``.

    Args:
        file (str): Path of the NIST table of the material.
        energy (float): Photon energy, in keV (a NumPy array also works).

    Returns:
        np.ndarray: The linear attenuation coefficient at ``energy``, in 1/cm (divide by 10 for 1/mm).
    """
    Edata, mudata = get_E_mu_data_from_datasheet(file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_f2_opt = curve_fit(dual_sqrt_exponential, Edata, mudata)[0]
    return dual_sqrt_exponential(energy, *p_f2_opt)