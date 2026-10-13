"""Ac-225 PSF modelling (DICOM)

The same ARF PSFs on a measured Symbia T2 scan with a non-circular orbit.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_ac225_dicom_recon.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% Ac-225 Advanced PSF Modeling (DICOM)
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Ac225-NEMA-SymT2")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Ac225-NEMA-SymT2")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pytomography
from pytomography import io
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.SPECT import SPECTAttenuationTransform
from pytomography.transforms.shared import GaussianFilter
import os
from spectarf import Collimator, DetectorStack, DetectorResponse, EnergyResolution
from spectarf.pytomography import generate_arfs, psf_at, psf_transform
from pytomography.utils import plot_utils

path = DATA / 'SPECT'

pathCT = os.path.join(path, 'Ac225-NEMA-SymT2', 'CT')
files_CT = [os.path.join(pathCT, file) for file in os.listdir(pathCT)]
file_NM = os.path.join(path, 'Ac225-NEMA-SymT2', 'projection_data.IMA')

# %% The camera
collimator = Collimator(hole_diameter_mm=4.0, septal_thickness_mm=2.0, hole_length_mm=59.7,
                        hole_shape='hex', rotated=True, material='lead')
stack = DetectorStack(collimator, crystal_mm=9.525, crystal_material='nai',
                      cover_mm=1.0, cover_material='aluminium',
                      backscatter_mm=100.0, backscatter_material='pmt')
energy_resolution = EnergyResolution.geb(a_keV=-0.534, b=0.946, c_per_keV=0.006)   # Morphis et al. 2021
response = DetectorResponse(energy_resolution, intrinsic_fwhm_mm_140=3.8)

index_peak = 3
index_lower = 4
index_upper = 5
projections, object_meta, proj_meta = io.SPECT.read_dicom(file_NM)
attenuation_map = io.attenuation_map(files_CT, object_meta, energy_keV=proj_meta.energy_windows[index_peak].photon_keV)

photopeak = projections[index_peak]
scatter = io.SPECT.energy_window_scatter(projections, proj_meta, peak=index_peak, lower=index_lower, upper=index_upper,
                                         sigma_theta=2, sigma_r_mm=4.8, sigma_z_mm=4.8)

# %% The ARFs
arfs = generate_arfs(stack, response, proj_meta, peak=index_peak, lower=index_lower, upper=index_upper)
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
att_transform = SPECTAttenuationTransform(attenuation_map=attenuation_map)
psf = psf_transform(arfs, memory_budget_bytes=2 * 2**30)
system_matrix = SPECTSystemMatrix(
        obj2obj_transforms = [att_transform, psf],
        proj2proj_transforms = [],
        object_meta = object_meta,
        proj_meta = proj_meta)

likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter)
algorithm = OSEM(likelihood)
print(algorithm.estimate_memory(n_subsets=3))

recon = algorithm(n_iters=100, n_subsets=3)

filter = GaussianFilter(20) # FWHM in mm
filter.configure(object_meta, proj_meta)
recon_smoothed = filter(recon)

ct = io.load_image(files_CT)
SPECT_imshow_kwargs = {
    'cmap': plot_utils.pet_cmap,
    'interpolation': 'Gaussian',
    'alpha': 0.6,
    'vmin': 0,
    'vmax': 0.1,
    'zorder': 1, # this will ensure SPECT is on top of CT
    'origin': 'lower'
}
CT_imshow_kwargs = {
    'cmap': 'Greys_r',
    'interpolation': 'Gaussian',
    'vmin': -25, # HU
    'vmax': 300, # HU
    'zorder': 0,
    'origin': 'lower'
}

plt.figure(figsize=(5,5), facecolor='k')
plt.title('440keV Peak: OSEM100it3ss', color='white')
plot_utils.dual_imshow_axial(
    im1 = recon_smoothed,
    im2 = ct.data,
    im1_idx = 70,
    affine1 = object_meta.affine,
    affine2 = ct.affine,
    imshow1_kwargs=SPECT_imshow_kwargs,
    imshow2_kwargs=CT_imshow_kwargs,
)
plt.axis('off')
plt.xlim(-120,130)
plt.ylim(-275,-25)
plt.tight_layout()

# %% Save the images
for name, image in [('osem', recon), ('osem_smoothed', recon_smoothed)]:
    io.save_dicom(image, OUTPUT / 't_ac225_dicom_recon' / name, object_meta, reference=file_NM,
                  series_description=f'OSEM 100x3, {name}', overwrite=True)
