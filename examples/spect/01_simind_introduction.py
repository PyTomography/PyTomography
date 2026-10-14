"""SIMIND introduction

A complete OSEM pipeline on Monte Carlo data, from projections to a reconstructed image.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_siminddata.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/SIMIND-Jaszak")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/SIMIND-Jaszak")

import inspect
import matplotlib.pyplot as plt
import torch
from pytomography.io.SPECT import simind
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
import os

# Paths are set in the data cell at the top of this tutorial
PATH = DATA / 'SPECT'
data_path = os.path.join(PATH, 'SIMIND-Jaszak', 'lu177_SYME_jaszak')

# %% 1. Opening Data
photopeak_path = os.path.join(data_path,'tot_w4.h00')
lower_path = os.path.join(data_path, 'tot_w5.h00')
upper_path = os.path.join(data_path, 'tot_w6.h00')

object_meta, proj_meta = simind.get_metadata(photopeak_path)

object_meta

proj_meta

photopeak = simind.get_projections(photopeak_path)
photopeak.shape

activity = 1000 # MBq 
time_per_proj = 15 # s
photopeak_realization = torch.poisson(photopeak * activity * time_per_proj)

lower = simind.get_projections(lower_path)
upper = simind.get_projections(upper_path)
lower_realization = torch.poisson(lower * activity * time_per_proj)
upper_realization = torch.poisson(upper * activity * time_per_proj)

ww_peak, ww_lower, ww_upper = [simind.get_energy_window_width(path) for path in [photopeak_path, lower_path, upper_path]]
scatter_estimate_TEW = simind.compute_EW_scatter(lower_realization, upper_realization , ww_lower, ww_upper, ww_peak)

# %% 2.1 Attenuation Modeling
path_amap = os.path.join(data_path, 'amap.hct')
amap = simind.get_attenuation_map(path_amap)

att_transform = SPECTAttenuationTransform(amap)

# %% 2.2 Collimator Detector Response (or Point Spread Function) Modeling
psf_meta = simind.get_psfmeta_from_header(photopeak_path)
# Below is optional to print details about psf_meta
sigma_fit_func_code = inspect.getsource(psf_meta.sigma_fit)
print(psf_meta)
print(sigma_fit_func_code)

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

reconstructed_image = recon_algorithm(
    n_iters=4,
    n_subsets=8,
)
