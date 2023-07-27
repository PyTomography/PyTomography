"""This module contains helper function for inputting/outputting/interpretting SPECT/CT data in the SIMIND and DICOM file formats. A considerable amount of these functions have to do with creating attenuation maps from CT data files for attenuation correction in SPECT imaging.
"""
from __future__ import annotations
from typing import Sequence
import warnings
import numpy as np
import re
import os
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks
import pydicom
from functools import partial

# Set filepaths of the module
module_path = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(module_path, '../../data/NIST_attenuation_data')
FILE_WATER = os.path.join(DATADIR, 'water.csv')
FILE_AIR = os.path.join(DATADIR, 'air.csv')
FILE_CBONE = os.path.join(DATADIR, 'bonecortical.csv')

def compute_TEW(projection_lower, projection_upper, width_lower, width_upper, width_peak):
    return (projection_lower/width_lower + projection_upper/width_upper)*width_peak / 2

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

def bilinear_transform(
    HU: float,
    a1: float,
    a2: float,
    b1: float,
    b2: float
    ) -> float:
    """Function used to convert between Hounsfield Units at an effective CT energy and linear attenuation coefficient at a given SPECT radionuclide energy. It consists of two distinct linear curves in regions :math:`HU<0` and :math:`HU \geq 0`.

    Args:
        HU (float): Hounsfield units at CT energy
        a1 (float): Fit parameter 1
        a2 (float): Fit parameter 2
        b1 (float): Fit parameter 3
        b2 (float): Fit parameter 4

    Returns:
        float: Linear attenuation coefficient at SPECT energy
    """
    output =  np.piecewise(
        HU,
        [HU < 0, HU >= 0],
        [lambda x: a1*x + b1,
        lambda x: a2*x + b2])
    output[output<0] = 0
    return output

def open_CT_file(
    files_CT: Sequence[str],
    return_max_slice_loc: bool = False
    ) -> np.array:
    """Given a list of seperate DICOM files, opens them up and stacks them together into a single CT image. 

    Args:
        files_CT (Sequence[str]): List of DICOM files corresponding to a particular scan
        return_max_slice_loc (bool, optional): Whether or not to return the max slice location of all the CT files. Needed when applying affine transformation to align CT and SPECT images. Defaults to False.

    Returns:
        np.array: CT scan in units of Hounsfield Units at the effective CT energy.
    """
    CT = []
    slice_locs = []
    for file in files_CT:
        ds = pydicom.read_file(file)
        CT.append(ds.pixel_array)
        slice_locs.append(float(ds.SliceLocation))
    CT = np.transpose(np.array(CT)[np.argsort(slice_locs)], (2,1,0)).astype(np.float32)
    CT = ds.RescaleSlope * CT + ds.RescaleIntercept
    if return_max_slice_loc:
        return CT, np.max(np.abs(slice_locs))
    else:
        return CT

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
    p_f2_opt = curve_fit(dual_sqrt_exponential, Edata, mudata)[0]
    return dual_sqrt_exponential(energy, *p_f2_opt)

def get_HU_from_spectrum_interp(
    file: str,
    energy: float
    ) -> np.array:
    """Obtains the Hounsfield Units of some material at a given energy

    Args:
        file (str): Filepath of material
        energy (float): Energy at which HU is desired

    Returns:
        np.array: HU at the desired energies.
    """
    mu_water = get_mu_from_spectrum_interp(FILE_WATER, energy)
    mu_air = get_mu_from_spectrum_interp(FILE_AIR, energy)
    mu_material = get_mu_from_spectrum_interp(file, energy)
    return (mu_material - mu_water)/(mu_water-mu_air) * 1000

def get_HU_mu_curve(
    files_CT: Sequence[str],
    CT_kvp: float,
    E_SPECT: float
    ) ->tuple[np.array, np.array]:
    """Gets Housnfield Unit vs. linear attenuation coefficient for air, water, and cortical bone data points

    Args:
        files_CT (Sequence[str]): Filepaths of all CT slices
        CT_kvp (float): Value of kVp for the CT scan
        E_SPECT (float): Photopeak energy of the SPECT scan

    Returns:
        tuple[np.array, np.array]: 
    """
    # try to get HU corresponding to cortical bone
    HU_cortical_bone = get_HU_corticalbone(files_CT)
    if HU_cortical_bone is not None:
        # compute effective CT energy from CBone HU
        E_CT = get_ECT_from_corticalbone_HU(HU_cortical_bone)
        print(f'Cortical Bone Peak: {HU_cortical_bone} HU')
        print(f'Effective CT Energy Determined: {E_CT} keV')
    else:
        # If can't get cortical bone peak, default to 75% KVP value
        warnings.warn("Could not find cortical bone peak: defaulting to 75% kVp value", category=Warning)
        E_CT = 0.75 * CT_kvp
    HU_CT = []
    mu_SPECT = []
    for file in [FILE_AIR, FILE_WATER, FILE_CBONE]:
        HU_CT.append(get_HU_from_spectrum_interp(file, E_CT))
        mu_SPECT.append(get_mu_from_spectrum_interp(file, E_SPECT))
    return np.array(HU_CT), np.array(mu_SPECT)

def HU_to_mu(
    HU: float,
    E: float,
    p_water_opt: Sequence[float],
    p_air_opt: Sequence[float]
    ):
    """Converts hounsfield units to linear attenuation coefficient

    Args:
        HU (float): Hounsfield Unit value
        E (float): Effective CT energy
        p_water_opt (Sequence[float]): Optimal fit parameters for mu vs. E data for water
        p_air_opt (Sequence[float]): Optimal fit parameters for mu vs. E data for air

    Returns:
        _type_: _description_
    """
    
    mu_water = dual_sqrt_exponential(E, *p_water_opt)
    mu_air = dual_sqrt_exponential(E, *p_air_opt)
    return 1/1000 * HU * (mu_water - mu_air) + mu_water

def get_HU_corticalbone(
    files_CT: Sequence[str]
    ) -> float | None:
    """Obtains the Hounsfield Unit corresponding to cortical bone from a CT scan.

    Args:
        files_CT (Sequence[str]): CT data files

    Returns:
        float | None: Hounsfield unit of bone. If not found, then returns ``None``.
    """
    HU_from_CT_slices = open_CT_file(files_CT)
    x = HU_from_CT_slices.ravel()
    N, bin_edges = np.histogram(x[(x>1200)*(x<1600)], bins=10, density=True)
    bins = bin_edges[:-1] + np.diff(bin_edges)[0]/2
    # Compute laplacian of histogram
    N_laplace = np.gradient(np.gradient(N))
    peaks, _ = find_peaks(N_laplace, prominence=8e-5)
    if len(peaks)>0:
        return bins[peaks[-1]]
    else:
        return None
    
def get_ECT_from_corticalbone_HU(HU: float) -> float:
    """Finds the effective CT energy that gives a cortical bone Hounsfield Unit value corresponding to ``HU``.

    Args:
        HU (float): Hounsfield Unit of Cortical bone at effective CT energy

    Returns:
        float: Effective CT energy
    """
    Edata, mudata_CB = get_E_mu_data_from_datasheet(FILE_CBONE)
    Edata, mudata_water = get_E_mu_data_from_datasheet(FILE_WATER)
    Edata, mudata_air = get_E_mu_data_from_datasheet(FILE_AIR)
    p_CB_opt = curve_fit(dual_sqrt_exponential, Edata, mudata_CB)[0]
    p_water_opt = curve_fit(dual_sqrt_exponential, Edata, mudata_water)[0]
    p_air_opt = curve_fit(dual_sqrt_exponential, Edata, mudata_air)[0]
    f = lambda E: 100*(dual_sqrt_exponential(E, *p_CB_opt) - HU_to_mu(HU, E, p_water_opt, p_air_opt))**2
    return minimize(f, x0=(115), method='SLSQP').x[0]

def get_HU2mu_conversion(
    files_CT: Sequence[str],
    CT_kvp: float,
    E_SPECT: float
    ) -> function:
    """Obtains the HU to mu conversion function that converts CT data to the required linear attenuation value in units of 1/cm required for attenuation correction in SPECT/PET imaging.

    Args:
        files_CT (Sequence[str]): CT data files
        CT_kvp (float): kVp value for CT scan
        E_SPECT (float): Energy of photopeak in SPECT scan

    Returns:
        function: Conversion function from HU to mu.
    """
    HU_CT, mu_SPECT = get_HU_mu_curve(files_CT, CT_kvp, E_SPECT)
    b1opt = b2opt = mu_SPECT[1] #water atten value
    a1opt = (mu_SPECT[1] - mu_SPECT[0]) / (HU_CT[1] - HU_CT[0])
    a2opt = (mu_SPECT[2] - mu_SPECT[1]) / (HU_CT[2] - HU_CT[1])
    return partial(bilinear_transform, a1=a1opt, a2=a2opt, b1=b1opt, b2=b2opt)

