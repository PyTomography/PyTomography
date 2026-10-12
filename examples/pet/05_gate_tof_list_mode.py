"""GATE TOF list mode

Time-of-flight list-mode reconstruction with TOF scatter.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_PETGATE_scat_lmTOF.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% GATE (Listmode Reconstruction; With Time of Flight)
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("PET/GATE-mMR-BrainInWater")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("PET/GATE-mMR-BrainInWater")

import torch
import numpy as np
import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.algorithms import OSEM
from pytomography import io
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.shared import GaussianFilter
import matplotlib.pyplot as plt
from pytomography.utils import sss
import gc

# The first run reads the ROOT files and caches each result in OUTPUT; later runs load the cache.
# Set this to True to recompute everything.
LOAD_FROM_ROOT = False

path = DATA / 'PET' / 'GATE-mMR-BrainInWater'
macro = path / 'mMR_Geometry.mac'  # the scanner, as the simulation's GATE macro describes it
paths = [path / 'simple_phantom' / f'mMR_voxBrain_withSimplePhantom_{i}.root' for i in range(1, 55)]  # its 54 runs


def read_gate(events, tof_meta=None, normalization=None):
    """io.PET.read_gate for one kind of coincidence: (events, object_meta, proj_meta). Reading the 54 ROOT files takes
    a few minutes, so the first run keeps the result in OUTPUT and later runs load it from there."""
    name = f'gate_{events}' + ('' if tof_meta is None else f'_tof{tof_meta.num_bins}') + ('' if normalization is None else '_norm')
    cache = OUTPUT / f'{name}.pt'
    if LOAD_FROM_ROOT or not cache.exists():
        # doi_mm: the lines of response end 9 mm inside the crystals, the mean depth of interaction
        torch.save(io.PET.read_gate(paths, macro, events=events, tof_meta=tof_meta, normalization=normalization,
                                    doi_mm=9), cache)
    return torch.load(cache, weights_only=False)

speed_of_light = 0.3 #mm/ps
fwhm_tof_resolution = 550 * speed_of_light / 2 #ps to position along LOR
TOF_range = 1000 * speed_of_light #ps to position along LOR (full range)
num_tof_bins = 21
tof_meta = PETTOFMeta(num_tof_bins, TOF_range, fwhm_tof_resolution, n_sigmas=3)

# %% How much memory this needs
RAM_GB = 25 # memory (RAM) this tutorial may use, in GB; it needs about 23 GB however low this is
pytomography.set_memory_budget(RAM_GB)

# %% Normalization Correction
# The true coincidences in their TOF bins (used from "Primary-Only Reconstruction" on), with the normalization weights
# of every crystal pair. proj_meta.info describes the scanner
detector_ids, object_meta, proj_meta = read_gate('trues', tof_meta=tof_meta,
                                                 normalization=path / 'normalization_weights.npy')
info = proj_meta.info
# The weights come with the dataset. They were computed from the calibration scan (36 ROOT files, 46 GB, in
# PET/GATE-mMR-Brain/normalization_scan), randoms left out (PyTomography PR #290):
# normalization_weights = io.PET.gate_normalization(normalization_paths, info, cylinder_radius_mm=318)
normalization_weights = proj_meta.weights_sensitivity  # read_gate's: one per crystal pair, in torch.combinations order

# %% Primary-Only Reconstruction
# The reconstruction grid: 128 x 128 x 96 voxels of 2 mm, centred where read_gate placed the scanner in the patient
object_meta = object_meta.with_grid(dr=(2, 2, 2), shape=(128, 128, 96))  # mm, voxels
# The phantom's attenuation map (1/cm), an image in GATE's world: gate_image places it in the patient as read_gate
# placed the scanner, and attenuation_map puts it on the grid, in 1/mm
atten_map = io.attenuation_map(io.PET.gate_image(path / 'umap_mMR_brainSimplePhantom.hv', proj_meta, units='1/cm'),
                               object_meta)
psf_transform = GaussianFilter(3.) # 3mm gaussian blurring
# Create system matrix: read_gate's proj_meta holds the events, the scanner's info, the TOF bins and the normalization
# weights
system_matrix = PETLMSystemMatrix(
    object_meta,
    proj_meta,
    obj2obj_transforms = [psf_transform],
    attenuation_map = atten_map,
    N_splits=8,
)
# Create likelihood. For listmode reconstruction, projections don't need to be provided, since all detection events are stored in proj_meta
likelihood = PoissonLogLikelihood(
    system_matrix,
)
# Initialize reconstruction algorithm
recon_algorithm = OSEM(likelihood)
print(recon_algorithm.estimate_memory(n_subsets=14)) # its peak RAM and GPU memory, before it runs
# Reconstruct
recon_primaryonly = recon_algorithm(n_iters=4, n_subsets=14)

# %% Reconstruction With Random/Scatter Estimation
# The primary-only reconstruction is done: free its events and system matrix (and PyTorch's GPU cache) before loading all the events
del detector_ids, proj_meta, system_matrix, likelihood, recon_algorithm
gc.collect()
torch.cuda.empty_cache()
detector_ids, _, _ = read_gate('prompts', tof_meta=tof_meta)

# %% Randoms
detector_ids_delays, _, _ = read_gate('delayeds')

sinogram_randoms_estimate = io.PET.listmode_to_sinogram(
    detector_ids_delays,
    info
)
del detector_ids_delays # binned: the events are not needed again
sinogram_randoms_estimate = io.PET.smooth_randoms_sinogram(
    sinogram_randoms_estimate,
    info,
    sigma_r=4,
    sigma_theta=4,
    sigma_z=4
)
sinogram_randoms_estimate = io.PET.randoms_sinogram_to_sinogramTOF(
    sinogram_randoms_estimate,
    tof_meta = tof_meta,
    coincidence_timing_width = 4300
) # coinicidence timing window for this GATE simulation was set to 4300ps
lm_randoms_estimate = io.PET.sinogram_to_listmode(
    detector_ids,
    sinogram_randoms_estimate,
    info,
)

# %% Scatters
# The same attenuation map as above, and all the events in their TOF bins with the normalization weights
proj_meta = PETLMProjMeta(
    detector_ids,
    info,
    weights_sensitivity=normalization_weights,
    tof_meta=tof_meta
    )
psf_transform = GaussianFilter(3.)
system_matrix = PETLMSystemMatrix(
       object_meta,
       proj_meta,
       obj2obj_transforms = [psf_transform],
       N_splits=10,
       attenuation_map=atten_map,
)
lm_norm = system_matrix._compute_sensitivity_projection(all_ids=False)
additive_term = lm_randoms_estimate / lm_norm
additive_term[additive_term.isnan()] = 0 # remove NaN values
# Provide the random-only
likelihood = PoissonLogLikelihood(
        system_matrix,
        additive_term = additive_term
    )
recon_algorithm = OSEM(likelihood)
recon_without_scatter_estimation = recon_algorithm(4,14)

# its peak RAM and GPU memory, before it runs
print(sss.estimate_sss_memory(object_meta, proj_meta, system_matrix, sinogram_random=sinogram_randoms_estimate,
                              tof_meta=tof_meta, sinogram_interring_stepsize=6, sinogram_intraring_stepsize=6,
                              num_dense_tof_bins=25))
scatter_sinogram = sss.get_sss_scatter_estimate(
        object_meta,
        proj_meta,
        recon_without_scatter_estimation,
        atten_map,
        system_matrix,
        sinogram_random=sinogram_randoms_estimate,
        tof_meta=tof_meta,
        num_dense_tof_bins=25,
        image_stepsize=6,
        sinogram_interring_stepsize=6,
        sinogram_intraring_stepsize=6,
        N_splits=1)
lm_scatter_estimate = io.PET.sinogram_to_listmode(proj_meta.detector_ids, scatter_sinogram, proj_meta.info)
# Save memory, these are not needed anymore
del(scatter_sinogram)
del(sinogram_randoms_estimate)
gc.collect()

additive_term = (lm_scatter_estimate + lm_randoms_estimate) / lm_norm
additive_term[additive_term.isnan()] = 0
likelihood = PoissonLogLikelihood(
        system_matrix,
        additive_term = additive_term
    )
recon_algorithm = OSEM(likelihood)
print(recon_algorithm.estimate_memory(n_subsets=14))
recon_lm_tof = recon_algorithm(4,14)

# %% Save the images
saved = io.save_dicom(recon_lm_tof, OUTPUT / 't_PETGATE_scat_lmTOF' / 'osem', object_meta,
                      series_description='TOF list-mode OSEM, scatter and randoms corrected', overwrite=True)
