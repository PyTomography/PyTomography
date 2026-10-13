"""Ac-225 PSF modelling (SIMIND)

Alpha-emitter imaging with PSFs from Monte Carlo ARFs (spectarf), penetration and collimator scatter included.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_ac225_simind_recon.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% Ac-225 Advanced PSF Modeling (SIMIND)
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/SIMIND-Jaszak")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/SIMIND-Jaszak")

import os
import numpy as np
import torch
import pytomography
from pytomography import io
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT  import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from spectarf import Collimator, DetectorStack, DetectorResponse, EnergyResolution
from spectarf.pytomography import generate_arfs, psf_at, psf_transform
from spectarf.response import FWHM_TO_SIGMA
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# change to where you saved the data
PATH = DATA / 'SPECT' / 'SIMIND-Jaszak'

dT = 2.5 * 60 # seconds per projection
activity_conc = 10 # kBq/L, a very high activity conc for Ac225
CPSpMBq = 17 # an approximate calibration factor

# Specify the isotopes and equilibirum ratios based on Bateman equations
isotopes = ['ac225', 'bi213', 'fr221', 'tl209']
isotopes_ratios = [1,1,1,0.0209]
# look at the .h00 files; there are many E windows simulated. These are 5% scat win for 440peak
i_peak, i_lower, i_upper = 10, 11, 14
files_NM = [[os.path.join(PATH, f'{isotope}', f'tot_w{i}.h00') for isotope in isotopes] for i in [i_lower, i_peak, i_upper]]
activity_concs = [activity_conc * ratio for ratio in isotopes_ratios]
projections, object_meta, proj_meta = io.SPECT.read_simind(files_NM, weights=activity_concs)
projections *= dT

# Based on how they are loaded, this is idx of peak, lower upper
idx_peak, idx_lower, idx_upper = 1, 0, 2
photopeak = projections[idx_peak]
# each window's width from its header (the first isotope's; the isotopes share the windows)
ww_lower, ww_peak, ww_upper = [window.width_keV for window in proj_meta.energy_windows]
print(f'window widths: lower {ww_lower:.0f} keV, photopeak {ww_peak:.0f} keV, upper {ww_upper:.0f} keV')
scatter_estimate_TEW = io.SPECT.energy_window_scatter(
    projections, proj_meta, peak=idx_peak, lower=idx_lower, upper=idx_upper,
    sigma_r_mm=5,
    sigma_z_mm=5,
)

path_amap = os.path.join(PATH, 'attenuation_maps', 'amap440.hct')
amap = io.attenuation_map(path_amap, object_meta, energy_keV=proj_meta.energy_windows[idx_peak].photon_keV, layout='mu')

def perform_reconstruction(psf_transform):
    att_transform = SPECTAttenuationTransform(attenuation_map=amap)
    system_matrix = SPECTSystemMatrix(
            obj2obj_transforms = [att_transform,psf_transform],
            proj2proj_transforms = [],
            object_meta = object_meta,
            proj_meta = proj_meta)
    likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter_estimate_TEW)
    algorithm = OSEM(likelihood)
    print(algorithm.estimate_memory(n_subsets=4))
    return algorithm(n_iters=50, n_subsets=4)

# %% The camera
collimator = Collimator(hole_diameter_mm=4.0, septal_thickness_mm=2.0, hole_length_mm=59.7,
                        hole_shape='hex', rotated=True, material='lead')
stack = DetectorStack(collimator, crystal_mm=9.525, crystal_material='nai',
                      cover_mm=1.0, cover_material='aluminium',
                      backscatter_mm=100.0, backscatter_material='pmt')
energy_resolution = EnergyResolution.geb(a_keV=-0.534, b=0.946, c_per_keV=0.006)   # SIMIND's siemens.erf
response = DetectorResponse(energy_resolution, intrinsic_fwhm_mm_140=3.6,
                            residual_sigma_mm_140=3.6 * FWHM_TO_SIGMA)

# %% The ARFs
arfs = generate_arfs(stack, response, proj_meta, peak=idx_peak, lower=idx_lower, upper=idx_upper)
for name, arf in arfs.items():
    print(f"{name}: {arf.window[1]:.0f}-{arf.window[2]:.0f} keV, {arf.energy_keV:.1f} keV photons, "
          f"grid (theta, phi, dx, dy) {arf.data.shape}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, gridspec_kw={'wspace': 0.05})
positive = lambda p: np.where(p > 0, p, np.nan)    # no zeros on a log axis
for name, arf in arfs.items():
    ax[0].semilogy(arf.theta_deg, positive(arf.in_window_probability().mean(axis=1)),
                   label=f"{name}: {arf.window[1]:.0f}-{arf.window[2]:.0f} keV")
P = arfs['P']
for j, phi in enumerate(P.phi_deg):
    ax[1].semilogy(P.theta_deg, positive(P.in_window_probability()[:, j]), color=plt.cm.viridis(j / max(len(P.phi_deg) - 1, 1)),
                   label=f"phi = {phi:.0f} deg" if j in (0, len(P.phi_deg) - 1) else None)
ax[0].set(xlabel='incidence angle theta (degrees)', ylabel='detection probability per photon',
          title='every window')
ax[1].set(xlabel='incidence angle theta (degrees)', title='photopeak window, by azimuth phi')
for a in ax:
    a.legend(frameon=False)
    a.grid(alpha=0.3)

# %% PSFs
distances = [50.0, 150.0, 300.0]
kernels = psf_at(arfs, distances, pixel_mm=1.0, size_mm=160.0, normalize=True)   # (3, 161, 161)

w = kernels.shape[-1] * 1.0 / 2          # half width, mm
c = kernels.shape[-1] // 2
u = (np.arange(kernels.shape[-1]) - c) * 1.0
fig, ax = plt.subplots(1, 4, figsize=(16, 3.7), gridspec_kw={'width_ratios': [1, 1, 1, 1.25], 'wspace': 0.3})
for a, d, k in zip(ax, distances, kernels):
    norm = LogNorm(vmin=k.max() * 1e-3, vmax=k.max())          # three decades below each PSF's peak
    a.imshow(np.clip(k, norm.vmin, None).T, origin='lower', extent=(-w, w, -w, w), cmap='hot', norm=norm)
    a.set(title=f'{d:.0f} mm from the collimator', xlabel='r (mm)')
ax[0].set_ylabel('z (mm)')
for d, k in zip(distances, kernels):
    ax[3].semilogy(u, np.clip(k[:, c], 1e-12, None), label=f'{d:.0f} mm')
ax[3].set(xlabel='r (mm)', ylabel='fraction per mm$^2$', title='through the centre',
          ylim=(0.5 * np.clip(kernels[:, :, c], 1e-12, None).min(), 2 * kernels.max()))
ax[3].legend(frameon=False)
ax[3].grid(alpha=0.3)

# %% Reconstruct
psf_transform_arf = psf_transform(arfs, memory_budget_bytes=2 * 2**30)
recon_arf = perform_reconstruction(psf_transform_arf)

path_bi213_prim = files_NM[1][1]
psf_meta = io.SPECT.simind_psf(path_bi213_prim)
psf_transform = SPECTPSFTransform(psf_meta)
reconbad = perform_reconstruction(psf_transform)

recons = [reconbad, recon_arf]
titles = ['Gaussian PSF', 'ARF PSF']

# %% Save the images
for name, image in [('arf_psf', recon_arf), ('gaussian_psf', reconbad)]:
    io.save_dicom(image, OUTPUT / 't_ac225_simind_recon' / name, object_meta, units='counts',
                  series_description=f'OSEM 50x4, {name}', overwrite=True)
