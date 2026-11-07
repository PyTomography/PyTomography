"""GATE TOF sinogram

Time-of-flight sinogram reconstruction with TOF scatter estimation.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_PETGATE_scat_sinoTOF.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
from __future__ import annotations
# %% GATE (Sinogram Reconstruction; With Time of Flight)
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
from pytomography.metadata.PET import PETTOFMeta, PETSinogramPolygonProjMeta
from pytomography.projectors.PET import PETSinogramSystemMatrix
from pytomography.algorithms import OSEM
from pytomography import io
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.shared import GaussianFilter
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
        # doi: the lines of response end 9 mm inside the crystals, the mean depth of interaction
        torch.save(io.PET.read_gate(paths, macro, events=events, tof_meta=tof_meta, normalization=normalization,
                                    doi=9), cache)
    return torch.load(cache, weights_only=False)

speed_of_light = 0.3 #mm/ps
fwhm_tof_resolution = 550 * speed_of_light / 2 #ps to position along LOR
TOF_range = 1000 * speed_of_light #ps to position along LOR (full range)
num_tof_bins = 21
tof_meta = PETTOFMeta(num_tof_bins, TOF_range, fwhm_tof_resolution, n_sigmas=3)

# %% How much memory this needs
RAM_GB = 25 # memory (RAM) this tutorial may use, in GB (at least about 21)
pytomography.set_memory_budget(RAM_GB)

# Measured on Windows: besides one subset's data, expected counts and their ratio (about four arrays of a subset's
# size, on the GPU), the final reconstruction holds about 15.5 GB: the normalization, randoms and sensitivity
# sinograms of 1.65 GB each, the events, the scatter estimate, Python and CUDA, and memory PyTorch keeps for reuse.
# At least 28 subsets keep a subset's arrays on the GPU small (1.2 GB each).
from pytomography.utils.memory import subsets_for_budget
sinogram_bytes = 224 * 449 * 4096 * num_tof_bins * 4 # one whole TOF sinogram (34.6 GB)
n_subsets = subsets_for_budget(sinogram_bytes, arrays=4, held_bytes=15.5e9, minimum=28, maximum=224)  # at least one of the 224 angles each
n_iters = max(1, round(2 * 28 / n_subsets))
print(f'{n_iters} iterations of {n_subsets} subsets')

# %% Normalization Correction
# The true coincidences in their TOF bins (used from "Primary-Only Reconstruction" on), with the normalization weights
# of every crystal pair. proj_meta.info describes the scanner
detector_ids, object_meta, proj_meta = read_gate('trues', tof_meta=tof_meta,
                                                 normalization=path / 'normalization_weights.npy')
info = proj_meta.info
if LOAD_FROM_ROOT or not (OUTPUT / 'normalization_sinogram.pt').exists():
    # The weights come with the dataset. They were computed from the calibration scan (36 ROOT files, 46 GB, in
    # PET/GATE-mMR-Brain/normalization_scan), randoms left out (PyTomography PR #290):
    # normalization_weights = io.PET.gate_normalization(normalization_paths, info, cylinder_radius=318)
    torch.save(io.PET.normalization_sinogram(proj_meta.weights_sensitivity, info), OUTPUT / 'normalization_sinogram.pt')
normalization_sinogram = torch.load(OUTPUT / 'normalization_sinogram.pt')
proj_meta.weights_sensitivity = None  # the sinogram holds what the reconstructions need

# %% Primary-Only Reconstruction
sinogram = io.PET.listmode_to_sinogram(detector_ids, info, tof_meta=tof_meta)

# The reconstruction grid: 128 x 128 x 96 voxels of 2 mm, centred where read_gate placed the scanner in the patient
object_meta = object_meta.with_grid(dr=(2, 2, 2), shape=(128, 128, 96))  # mm, voxels
# The phantom's attenuation map (1/cm), an image in GATE's world: gate_image places it in the patient as read_gate
# placed the scanner, and attenuation_map puts it on the grid, in 1/mm
atten_map = io.attenuation_map(io.PET.gate_image(path / 'umap_mMR_brainSimplePhantom.hv', proj_meta, units='1/cm'),
                               object_meta)
# Get projection space metadata from PET geometry information dictionary and TOF metadata
proj_meta = PETSinogramPolygonProjMeta(info, tof_meta=tof_meta)
psf_transform = GaussianFilter(3.) # 3mm gaussian blurring
# Create system matrix.
system_matrix = PETSinogramSystemMatrix(
       object_meta,
       proj_meta,
       obj2obj_transforms = [psf_transform],
       sinogram_sensitivity = normalization_sinogram,
       N_splits=10,
       attenuation_map=atten_map,
       device=pytomography.device # the projections of each subset stay on the GPU (device='cpu' keeps them in RAM: slower)
)
# Create likelihood
likelihood = PoissonLogLikelihood(
    system_matrix,
    sinogram,
)
# Reconstruct
recon_algorithm = OSEM(likelihood)
print(recon_algorithm.estimate_memory(n_subsets=n_subsets)) # its peak RAM and GPU memory, before it runs
# delete to save memory
del(sinogram)
del(system_matrix)
del(likelihood)
del(recon_algorithm)
gc.collect()

# %% Reconstruction Correcting For Randoms + Scatters
detector_ids, _, _ = read_gate('prompts', tof_meta=tof_meta)

sinogram = io.PET.listmode_to_sinogram(detector_ids, info, tof_meta=tof_meta)
del detector_ids # the sinogram keeps its own compact copy of the events (0.3 GB instead of 1.2 GB)

# %% Randoms
detector_ids_delays, _, _ = read_gate('delayeds')

# Load random events accross all TOF bins
sinogram_randoms_estimate = io.PET.listmode_to_sinogram(detector_ids_delays, info)
sinogram_randoms_estimate = io.PET.smooth_randoms_sinogram(sinogram_randoms_estimate, info, fwhm_r=20, fwhm_theta=7, fwhm_z=40)
sinogram_randoms_estimate = io.PET.randoms_sinogram_to_sinogramTOF(sinogram_randoms_estimate, tof_meta, coincidence_timing_width = 4300) # coinicidence timing window for this GATE simulation was set to 4300ps

# %% Scatters
# The same attenuation map and normalization sinogram as above
proj_meta = PETSinogramPolygonProjMeta(info, tof_meta)
psf_transform = GaussianFilter(3.)
system_matrix = PETSinogramSystemMatrix(
       object_meta,
       proj_meta,
       obj2obj_transforms = [psf_transform],
       sinogram_sensitivity = normalization_sinogram,
       N_splits=10,
       attenuation_map=atten_map,
       device=pytomography.device # the projections of each subset stay on the GPU
)
additive_term = sinogram_randoms_estimate.unsqueeze(-1) / system_matrix._compute_sensitivity_sinogram().cpu()
likelihood = PoissonLogLikelihood(
        system_matrix,
        sinogram,
        additive_term = additive_term
    )
recon_algorithm = OSEM(likelihood)
recon_without_scatter_estimation = recon_algorithm(n_iters, n_subsets)
# The likelihood keeps its last expected counts (one subset, 1.2 GB) and the additive term; the scatter estimate below
# needs only the system matrix
del likelihood, recon_algorithm, additive_term
gc.collect()

# its peak RAM and GPU memory, before it runs
print(sss.estimate_sss_memory(object_meta, proj_meta, system_matrix, proj_data=sinogram,
                              sinogram_random=sinogram_randoms_estimate, tof_meta=tof_meta,
                              sinogram_interring_stepsize=6, sinogram_intraring_stepsize=6, num_dense_tof_bins=25))
sinogram_scatter_estimate = sss.get_sss_scatter_estimate(
    object_meta = object_meta,
    proj_meta = proj_meta,
    pet_image = recon_without_scatter_estimation,
    attenuation_image = atten_map,
    system_matrix = system_matrix,
    proj_data = sinogram,
    image_stepsize = 6,
    attenuation_cutoff = 0.004,
    sinogram_interring_stepsize = 6,
    sinogram_intraring_stepsize = 6,
    sinogram_random = sinogram_randoms_estimate,
    tof_meta = tof_meta,
    num_dense_tof_bins = 25
    )

additive_term = (sinogram_randoms_estimate.unsqueeze(-1) + sinogram_scatter_estimate) / system_matrix._compute_sensitivity_sinogram().cpu()
likelihood = PoissonLogLikelihood(
        system_matrix,
        sinogram,
        additive_term = additive_term
    )
recon_algorithm = OSEM(likelihood)
print(recon_algorithm.estimate_memory(n_subsets=n_subsets))
recon_sinogram_TOF = recon_algorithm(n_iters, n_subsets)

# %% Save the images
saved = io.save_dicom(recon_sinogram_TOF, OUTPUT / 't_PETGATE_scat_sinoTOF' / 'osem', object_meta,
                      series_description='TOF OSEM, scatter and randoms corrected', overwrite=True)
