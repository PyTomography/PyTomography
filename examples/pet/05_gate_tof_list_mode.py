"""GATE TOF list mode

Time-of-flight list-mode reconstruction with TOF scatter.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_PETGATE_scat_lmTOF.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% GATE (Listmode Reconstruction; With Time of Flight)
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("PET/GATE-mMR-Brain")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("PET/GATE-mMR-Brain")

from __future__ import annotations
import torch
import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.algorithms import OSEM
from pytomography.io.PET import gate
from pytomography.likelihoods import PoissonLogLikelihood
import os
from pytomography.transforms.shared import GaussianFilter
import matplotlib.pyplot as plt
from pytomography.utils import sss
import gc

# The first run reads the ROOT files and caches each result in OUTPUT; later runs load the cache.
# Set this to True to recompute everything.
LOAD_FROM_ROOT = False

path = DATA / 'PET' / 'GATE-mMR-Brain'
# Macro path where PET scanner geometry file is defined
macro_path = os.path.join(path, 'mMR_Geometry.mac')
# Get information dictionary about the scanner
info = gate.get_detector_info(path = macro_path,
    mean_interaction_depth=9, min_rsector_difference=0)
# Paths to all ROOT files containing data
paths = [os.path.join(path, f'all_physics/mMR_voxBrain_{i}.root') for i in range(1, 55) if i != 24]  # file 24 is empty

speed_of_light = 0.3 #mm/ps
fwhm_tof_resolution = 550 * speed_of_light / 2 #ps to position along LOR
TOF_range = 1000 * speed_of_light #ps to position along LOR (full range)
num_tof_bins = 21
tof_meta = PETTOFMeta(num_tof_bins, TOF_range, fwhm_tof_resolution, n_sigmas=3)

# %% Normalization Correction
if LOAD_FROM_ROOT or not os.path.exists(os.path.join(OUTPUT, 'normalization_weights.pt')):
    normalization_paths = [os.path.join(path, f'normalization_scan/mMR_Norm_{i}.root') for i in range(1,37)]

    # Get normalization weights for all possible detector ID pairs
    normalization_weights = gate.get_normalization_weights_cylinder_calibration(
        normalization_paths,
        info,
        cylinder_radius = 318, # mm (radius of calibration cylindrical shell,
        include_randoms=False 
    )

    torch.save(normalization_weights, os.path.join(OUTPUT, 'normalization_weights.pt'))
normalization_weights = torch.load(os.path.join(OUTPUT, 'normalization_weights.pt'))

# %% Primary-Only Reconstruction
if LOAD_FROM_ROOT or not os.path.exists(os.path.join(OUTPUT, 'detector_ids_tof21bin_primary_only.pt')):
    detector_ids = gate.get_detector_ids_from_root(
        paths,
        info,
        tof_meta = tof_meta,
        include_randoms=False,
        include_scatters=False)
    detector_ids = detector_ids[detector_ids[:,2]>-1] # For TOF, only take events within the TOF bins
    torch.save(detector_ids, os.path.join(OUTPUT, 'detector_ids_tof21bin_primary_only.pt'))
detector_ids = torch.load(os.path.join(OUTPUT, 'detector_ids_tof21bin_primary_only.pt'))

# Specify object space for reconstruction
object_meta = ObjectMeta(
    dr=(2,2,2), #mm
    shape=(128,128,96) #voxels
)
# Get projection space metadata from PET geometry information dictionary
proj_meta = PETLMProjMeta(
    detector_ids,
    info,
    tof_meta=tof_meta,
    weights_sensitivity=normalization_weights
    )
# Get attenuation map and PSF transform from the associated phantom
atten_map = gate.get_attenuation_map_nifti(os.path.join(path, 'fdg_pet_phantom_umap.nii.gz'), object_meta).to(pytomography.dtype).to(pytomography.device)
psf_transform = GaussianFilter(3.) # 4mm gaussian blurring
# Create system matrix.
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
# Reconstruct
recon_primaryonly = recon_algorithm(n_iters=4, n_subsets=14)

# %% Reconstruction With Random/Scatter Estimation
if LOAD_FROM_ROOT or not os.path.exists(os.path.join(OUTPUT, 'detector_ids_tof21bin_all_events.pt')):
    detector_ids = gate.get_detector_ids_from_root(
        paths,
        info,
        tof_meta=tof_meta
        )
    detector_ids = detector_ids[detector_ids[:,2]>-1] # For TOF, only take events within the TOF bins
    torch.save(detector_ids, os.path.join(OUTPUT, 'detector_ids_tof21bin_all_events.pt'))
detector_ids = torch.load(os.path.join(OUTPUT, 'detector_ids_tof21bin_all_events.pt'))

# %% Randoms
if LOAD_FROM_ROOT or not os.path.exists(os.path.join(OUTPUT, 'detector_ids_delays.pt')):
    detector_ids_delays = gate.get_detector_ids_from_root(
        paths,
        info,
        substr = 'delay')
    torch.save(detector_ids_delays, os.path.join(OUTPUT, 'detector_ids_delays.pt'))
detector_ids_delays= torch.load(os.path.join(OUTPUT, 'detector_ids_delays.pt'))

sinogram_randoms_estimate = gate.listmode_to_sinogram(
    detector_ids_delays,
    info
)
sinogram_randoms_estimate = gate.smooth_randoms_sinogram(
    sinogram_randoms_estimate,
    info,
    sigma_r=4,
    sigma_theta=4,
    sigma_z=4
)
sinogram_randoms_estimate = gate.randoms_sinogram_to_sinogramTOF(
    sinogram_randoms_estimate,
    tof_meta = tof_meta,
    coincidence_timing_width = 4300
) # coinicidence timing window for this GATE simulation was set to 4300ps
lm_randoms_estimate = gate.sinogram_to_listmode(
    detector_ids,
    sinogram_randoms_estimate,
    info,
)

# %% Scatters
atten_map = gate.get_attenuation_map_nifti(os.path.join(path, 'fdg_pet_phantom_umap.nii.gz'), object_meta).to(pytomography.dtype).to(pytomography.device)
normalization_weights = torch.load(os.path.join(OUTPUT, 'normalization_weights.pt'))
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
       attenuation_map=atten_map.to(pytomography.device),
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
lm_scatter_estimate = gate.sinogram_to_listmode(proj_meta.detector_ids, scatter_sinogram, proj_meta.info)
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
recon_lm_tof = recon_algorithm(4,14)
