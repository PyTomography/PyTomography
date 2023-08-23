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
    """Function used for curve fitting of linear attenuation coefficient vs. photon energy curves from NIST. It's given by the functional form :math:`f(x) = c_1e^{-d_1\sqrt{x}} + c_2e^{-d_2\sqrt{x}}`. It was chosen purely because it gave good fit results.

    Args:
        energy (float): Energy of photon
        c1 (float): Fit parameter 1
        c2 (float): Fit parameter 2
        d1 (float): Fit parameter 3
        d2 (float): Fit parameter 4

    Returns:
        float: _description_
    """
    return c1*np.exp(-d1*np.sqrt(energy)) + c2*np.exp(-d2*np.sqrt(energy))

def get_E_mu_data_from_datasheet(file: str) -> tuple[np.array, np.array]:
    """Return energy and linear attenuation data from NIST datafiles of mass attenuation coefficients between 50keV and 511keV.

    Args:
        file (str): Location of NIST data file. Corresponds to a particular element/material.

    Returns:
        tuple[np.array, np.array]: Energy and linear attenuation values.
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
    """Gets linear attenuation corresponding to a given energy in in the data from ``file``.

    Args:
        file (str): Filepath of the mu-energy data
        energy (float): Energy at which mu is computed

    Returns:
        np.array: Linear attenuation coefficient (in 1/cm) at the desired energies.
    """
    Edata, mudata = get_E_mu_data_from_datasheet(file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_f2_opt = curve_fit(dual_sqrt_exponential, Edata, mudata)[0]
    return dual_sqrt_exponential(energy, *p_f2_opt)