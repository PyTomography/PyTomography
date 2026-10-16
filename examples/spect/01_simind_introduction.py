"""SIMIND introduction

A complete OSEM pipeline on Monte Carlo data, from projections to a reconstructed image.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_siminddata.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/SIMIND-Jaszak")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/SIMIND-Jaszak")

import torch
from pytomography import io
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
import os

# Paths are set in the data cell at the top of this tutorial
PATH = DATA / 'SPECT'
data_path = os.path.join(PATH, 'SIMIND-Jaszak', 'lu177_SYME_jaszak')

# %% 1. Opening Data
photopeak_path = os.path.join(data_path,'tot_w1.h00')  # the 208 keV photopeak
lower_path = os.path.join(data_path, 'tot_w2.h00')
upper_path = os.path.join(data_path, 'tot_w3.h00')

projections, object_meta, proj_meta = io.SPECT.read_simind([photopeak_path, lower_path, upper_path])
for name, window in zip(['photopeak', 'lower', 'upper'], proj_meta.energy_windows):
    print(f'{name:9s} window: {window.lower:.1f}-{window.upper:.1f} keV')

photopeak = projections[0]

activity = 1000 # MBq 
time_per_proj = 15 # s
photopeak_realization = torch.poisson(photopeak * activity * time_per_proj)

lower_realization = torch.poisson(projections[1] * activity * time_per_proj)
upper_realization = torch.poisson(projections[2] * activity * time_per_proj)

realizations = torch.stack([photopeak_realization, lower_realization, upper_realization])
scatter_estimate_TEW = io.SPECT.energy_window_scatter(realizations, proj_meta, peak=0, lower=1, upper=2)

# %% 2.1 Attenuation Modeling
path_amap = os.path.join(data_path, 'amap.hct')
amap = io.attenuation_map(path_amap, object_meta, energy=proj_meta.energy_windows[0].photon_energy, layout='mu')

att_transform = SPECTAttenuationTransform(amap)

# %% 2.2 Collimator Detector Response (or Point Spread Function) Modeling
psf_meta = io.SPECT.simind_psf(photopeak_path)
# Below is optional to print details about psf_meta
print(psf_meta)

psf_transform = SPECTPSFTransform(psf_meta)

# %% 2.3 Creating the system matrix
system_matrix = SPECTSystemMatrix(
        obj2obj_transforms = [att_transform,psf_transform],
        proj2proj_transforms = [],
        object_meta = object_meta,
        proj_meta = proj_meta
    )

# %% 3 Likelihood / Reconstruction
likelihood = PoissonLogLikelihood(
    system_matrix = system_matrix,
    projections = photopeak_realization,
    additive_term = scatter_estimate_TEW
)

recon_algorithm = OSEM(likelihood)
print(recon_algorithm.estimate_memory(n_subsets=8))

reconstructed_image = recon_algorithm(
    n_iters=4,
    n_subsets=8,
)

# %% Save the images
saved = io.save_dicom(reconstructed_image, OUTPUT / 't_siminddata' / 'osem', object_meta, units='counts',
                      series_description='OSEM 4x8', overwrite=True)
