"""GATE TOF sinogram

Time-of-flight sinogram reconstruction with TOF scatter estimation.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_PETGATE_scat_sinoTOF.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% GATE (Sinogram Reconstruction; With Time of Flight)
import os
from pathlib import Path

# Tutorial data: the folder set by the PYTOMOGRAPHY_DATA environment variable (see Tutorial data in the docs)
DATA = Path(os.environ.get("PYTOMOGRAPHY_DATA", "~/pytomography_data")).expanduser()
# Results go here, never into the data folder
OUTPUT = Path(os.environ.get("PYTOMOGRAPHY_OUTPUT", "pytomography_outputs")).expanduser() / "PET/GATE-mMR-Brain"
OUTPUT.mkdir(parents=True, exist_ok=True)

from __future__ import annotations
import torch
import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETTOFMeta, PETSinogramPolygonProjMeta
from pytomography.projectors.PET import PETSinogramSystemMatrix
from pytomography.algorithms import OSEM
from pytomography.io.PET import gate
from pytomography.likelihoods import PoissonLogLikelihood
import os
from pytomography.transforms.shared import GaussianFilter
import matplotlib.pyplot as plt
from pytomography.utils import sss
import gc

LOAD_FROM_ROOT = False # Set to true if .pt files not generated

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
if LOAD_FROM_ROOT:
    normalization_paths = [os.path.join(path, f'normalization_scan/mMR_Norm_{i}.root') for i in range(1,37)]

    # Get eta in listmode format
    normalization_weights = gate.get_normalization_weights_cylinder_calibration(
        normalization_paths,
        info,
        cylinder_radius = 318, # mm (radius of calibration cylindrical shell,
        include_randoms=False 
    )

    normalization_sinogram = gate.get_norm_sinogram_from_listmode_data(normalization_weights, macro_path)
    torch.save(normalization_sinogram, os.path.join(OUTPUT, 'normalization_sinogram.pt'))
normalization_sinogram = torch.load(os.path.join(OUTPUT, 'normalization_sinogram.pt'))

# %% Primary-Only Reconstruction
if LOAD_FROM_ROOT:
    detector_ids = gate.get_detector_ids_from_root(
        paths,
        info,
        tof_meta = tof_meta,
        include_randoms=False,
        include_scatters=False)
    detector_ids = detector_ids[detector_ids[:,2]>-1] # For TOF, only take events within the TOF bins
    torch.save(detector_ids, os.path.join(OUTPUT, 'detector_ids_tof21bin_primary_only.pt'))
detector_ids = torch.load(os.path.join(OUTPUT, 'detector_ids_tof21bin_primary_only.pt'))

LOAD_FROM_ROOT = True
if LOAD_FROM_ROOT:
    detector_ids_randoms_true = gate.get_detector_ids_from_root(
        paths,
        info,
        tof_meta = tof_meta,
        randoms_only=True)
    detector_ids_scatters_true = gate.get_detector_ids_from_root(
        paths,
        info,
        tof_meta = tof_meta,
        scatters_only=True)
    detector_ids_randoms_true = detector_ids_randoms_true[detector_ids_randoms_true[:,2]>-1] # For TOF, only take events within the TOF bins
    detector_ids_scatters_true = detector_ids_scatters_true[detector_ids_scatters_true[:,2]>-1] # For TOF, only take events within the TOF bins
    torch.save(detector_ids_randoms_true, os.path.join(OUTPUT, 'detector_ids_randoms_true_tof21bin.pt'))
    torch.save(detector_ids_scatters_true, os.path.join(OUTPUT, 'detector_ids_scatters_true_tof21bin.pt'))
detector_ids_randoms_true = torch.load(os.path.join(OUTPUT, 'detector_ids_randoms_true_tof21bin.pt'))
detector_ids_scatters_true = torch.load(os.path.join(OUTPUT, 'detector_ids_scatters_true_tof21bin.pt'))
LOAD_FROM_ROOT = False

sinogram = gate.listmode_to_sinogram(detector_ids, info, tof_meta=tof_meta)

# Specify object space for reconstruction
object_meta = ObjectMeta(
    dr=(2,2,2), #mm
    shape=(128,128,96) #voxels
)
# Get projection space metadata from PET geometry information dictionary and TOF metadata
proj_meta = PETSinogramPolygonProjMeta(info, tof_meta=tof_meta)
# Get attenuation map and PSF transform from the associated phantom
atten_map = gate.get_attenuation_map_nifti(os.path.join(path, 'fdg_pet_phantom_umap.nii.gz'), object_meta).to(pytomography.dtype).to(pytomography.device)
psf_transform = GaussianFilter(3.) # 2mm gaussian blurring
# Create system matrix.
system_matrix = PETSinogramSystemMatrix(
       object_meta,
       proj_meta,
       obj2obj_transforms = [psf_transform],
       sinogram_sensitivity = normalization_sinogram,
       N_splits=10,
       attenuation_map=atten_map,
       device='cpu' # projections output on cpu, computation is still on GPU
)
# Create likelihood
likelihood = PoissonLogLikelihood(
    system_matrix,
    sinogram,
)
# Reconstruct
recon_algorithm = OSEM(likelihood)
recon_primaryonly = recon_algorithm(n_iters=4, n_subsets=14)
# delete to save memory
del(sinogram) 
del(system_matrix) 
del(likelihood)
del(recon_algorithm) 
gc.collect()

# %% Reconstruction Correcting For Randoms + Scatters
if LOAD_FROM_ROOT:
    detector_ids = gate.get_detector_ids_from_root(
        paths,
        info,
        tof_meta=tof_meta
        )
    detector_ids = detector_ids[detector_ids[:,2]>-1] # For TOF, only take events within the TOF bins
    torch.save(detector_ids, os.path.join(OUTPUT, 'detector_ids_tof21bin_all_events.pt'))
detector_ids = torch.load(os.path.join(OUTPUT, 'detector_ids_tof21bin_all_events.pt'))

sinogram = gate.listmode_to_sinogram(detector_ids, info, tof_meta=tof_meta)

# %% Randoms
if LOAD_FROM_ROOT:
    detector_ids_delays = gate.get_detector_ids_from_root(
        paths,
        info,
        substr = 'delay')
    torch.save(detector_ids_delays, os.path.join(OUTPUT, 'detector_ids_delays.pt'))
detector_ids_delays= torch.load(os.path.join(OUTPUT, 'detector_ids_delays.pt'))

# Load random events accross all TOF bins
sinogram_randoms_estimate = gate.listmode_to_sinogram(detector_ids_delays, info)
sinogram_randoms_estimate = gate.smooth_randoms_sinogram(sinogram_randoms_estimate, info, sigma_r=4, sigma_theta=4, sigma_z=4)
sinogram_randoms_estimate = gate.randoms_sinogram_to_sinogramTOF(sinogram_randoms_estimate, tof_meta, coincidence_timing_width = 4300) # coinicidence timing window for this GATE simulation was set to 4300ps

# %% Scatters
atten_map = gate.get_attenuation_map_nifti(os.path.join(path, 'fdg_pet_phantom_umap.nii.gz'), object_meta).to(pytomography.dtype).to(pytomography.device)
normalization_sinogram = torch.load(os.path.join(OUTPUT, 'normalization_sinogram.pt')) # assumes this has been saved from the intro tutorial
proj_meta = PETSinogramPolygonProjMeta(info, tof_meta)
psf_transform = GaussianFilter(3.)
system_matrix = PETSinogramSystemMatrix(
       object_meta,
       proj_meta,
       obj2obj_transforms = [psf_transform],
       sinogram_sensitivity = normalization_sinogram,
       N_splits=10,
       attenuation_map=atten_map,
       device='cpu' # projections output on cpu, rest is GPU
)
additive_term = sinogram_randoms_estimate.unsqueeze(-1) / system_matrix._compute_sensitivity_sinogram().cpu()
likelihood = PoissonLogLikelihood(
        system_matrix,
        sinogram,
        additive_term = additive_term
    )
recon_algorithm = OSEM(likelihood)
recon_without_scatter_estimation = recon_algorithm(4,14)

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
recon_sinogram_TOF = recon_algorithm(4,14)
