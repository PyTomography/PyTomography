"""GATE sinogram

Non-TOF sinogram reconstruction of a simulated brain phantom with scatter and randoms.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_PETGATE_scat_sino.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
from __future__ import annotations
# %% GATE (Sinogram Reconstruction; No Time of Flight)
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
from pytomography.metadata.PET import PETSinogramPolygonProjMeta
from pytomography.projectors.PET import PETSinogramSystemMatrix
from pytomography.algorithms import OSEM
from pytomography import io
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.utils import sss
from pytomography.transforms.shared import GaussianFilter

# %% How much memory this needs
RAM_GB = 25 # memory (RAM) this tutorial may use, in GB; it needs about 18 GB however low this is
pytomography.set_memory_budget(RAM_GB)

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

# The true coincidences (used from "Primary-Only Reconstruction" on), with the normalization weights of every crystal
# pair (below). proj_meta.info describes the scanner
detector_ids, object_meta, proj_meta = read_gate('trues', normalization=path / 'normalization_weights.npy')
info = proj_meta.info

# %% Normalization Correction
if LOAD_FROM_ROOT or not (OUTPUT / 'normalization_sinogram.pt').exists():
    # The weights come with the dataset. They were computed from the calibration scan (36 ROOT files, 46 GB, in
    # PET/GATE-mMR-Brain/normalization_scan), randoms left out (PyTomography PR #290):
    # normalization_weights = io.PET.gate_normalization(normalization_paths, info, cylinder_radius_mm=318)
    torch.save(io.PET.normalization_sinogram(proj_meta.weights_sensitivity, info), OUTPUT / 'normalization_sinogram.pt')
normalization_sinogram = torch.load(OUTPUT / 'normalization_sinogram.pt')
proj_meta.weights_sensitivity = None  # the sinogram holds what the reconstructions need

# %% Primary-Only Reconstruction
sinogram = io.PET.listmode_to_sinogram(detector_ids, info)

# The reconstruction grid: 128 x 128 x 96 voxels of 2 mm, centred where read_gate placed the scanner in the patient
object_meta = object_meta.with_grid(dr=(2, 2, 2), shape=(128, 128, 96))  # mm, voxels
# The phantom's attenuation map (1/cm), an image in GATE's world: gate_image places it in the patient as read_gate
# placed the scanner, and attenuation_map puts it on the grid, in 1/mm
atten_map = io.attenuation_map(io.PET.gate_image(path / 'umap_mMR_brainSimplePhantom.hv', proj_meta, units='1/cm'),
                               object_meta)
# Get projection space metadata from PET geometry information dictionary
proj_meta = PETSinogramPolygonProjMeta(info)
psf_transform = GaussianFilter(3) # 3mm gaussian blurring
# Create system matrix
system_matrix = PETSinogramSystemMatrix(
       object_meta,
       proj_meta,
       obj2obj_transforms = [psf_transform],
       sinogram_sensitivity = normalization_sinogram,
       N_splits=10, # Split FP/BP into 10 loops to save memory
       attenuation_map=atten_map,
       device='cpu' # projections are output on the CPU, but internal computation is on GPU
)
# Create likelihood
likelihood = PoissonLogLikelihood(
    system_matrix,
    sinogram,
)
# Initialize reconstruction algorithm
recon_algorithm = OSEM(likelihood)
print(recon_algorithm.estimate_memory(n_subsets=24)) # its peak RAM and GPU memory, before it runs

# %% Reconstruction Correcting For Randoms + Scatters
# The primary-only reconstruction is done: free its sinogram, likelihood and algorithm before binning all the events
del sinogram, likelihood, recon_algorithm
detector_ids, _, _ = read_gate('prompts')

sinogram = io.PET.listmode_to_sinogram(detector_ids, info)
del detector_ids # the sinogram holds what the reconstructions need

# %% Randoms
detector_ids_delays, _, _ = read_gate('delayeds')

sinogram_delays = io.PET.listmode_to_sinogram(detector_ids_delays, info)

sinogram_randoms_estimate = io.PET.smooth_randoms_sinogram(sinogram_delays, info, sigma_r=4, sigma_theta=4, sigma_z=4)
del sinogram_delays # only the smoothed estimate is used from here on

# %% Scatters
# The same attenuation map and normalization sinogram as above
proj_meta = PETSinogramPolygonProjMeta(info)
psf_transform = GaussianFilter(3)
system_matrix = PETSinogramSystemMatrix(
       object_meta,
       proj_meta,
       obj2obj_transforms = [psf_transform],
       sinogram_sensitivity = normalization_sinogram,
       N_splits=10,
       attenuation_map=atten_map,
       device='cpu' # projections output on cpu, rest is GPU
)

additive_term = sinogram_randoms_estimate / system_matrix._compute_sensitivity_sinogram().cpu()
likelihood = PoissonLogLikelihood(
        system_matrix,
        sinogram,
        additive_term = additive_term
    )

recon_algorithm = OSEM(likelihood)
recon_without_scatter_estimation = recon_algorithm(2,24)
# The likelihood keeps its additive term, a whole sinogram; the scatter estimate below needs only the system matrix
del likelihood, recon_algorithm, additive_term

# its peak RAM and GPU memory, before it runs
print(sss.estimate_sss_memory(object_meta, proj_meta, system_matrix, proj_data=sinogram,
                              sinogram_random=sinogram_randoms_estimate, sinogram_interring_stepsize=4,
                              sinogram_intraring_stepsize=4))
sinogram_scatter = sss.get_sss_scatter_estimate(
    object_meta = object_meta,
    proj_meta = proj_meta,
    pet_image = recon_without_scatter_estimation,
    attenuation_image = atten_map,
    system_matrix = system_matrix,
    proj_data = sinogram,
    image_stepsize = 4,
    attenuation_cutoff = 0.004,
    sinogram_interring_stepsize = 4,
    sinogram_intraring_stepsize = 4,
    sinogram_random = sinogram_randoms_estimate
    )

additive_term = (sinogram_randoms_estimate + sinogram_scatter) / system_matrix._compute_sensitivity_sinogram().cpu()
likelihood = PoissonLogLikelihood(
        system_matrix,
        sinogram,
        additive_term = additive_term
    )
recon_algorithm = OSEM(likelihood)
print(recon_algorithm.estimate_memory(n_subsets=24))
recon_sinogram_noTOF = recon_algorithm(2,24)

# %% Save the images
saved = io.save_dicom(recon_sinogram_noTOF, OUTPUT / 't_PETGATE_scat_sino' / 'osem', object_meta,
                      series_description='OSEM, scatter and randoms corrected', overwrite=True)
