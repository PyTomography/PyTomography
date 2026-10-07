"""GATE list mode

List-mode reconstruction with scatter and randoms correction.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_PETGATE_scat_lm.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% GATE (Listmode Reconstruction; Without Time of Flight)
from __future__ import annotations
import torch
import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.algorithms import OSEM
from pytomography.io.PET import gate, shared
from pytomography.likelihoods import PoissonLogLikelihood
import os
from pytomography.transforms.shared import GaussianFilter
import matplotlib.pyplot as plt
from pytomography.utils import sss

LOAD_FROM_ROOT = False  # Set to true if .pt files not generated

path = '/disk1/pet_mri_scan/'
# Macro path where PET scanner geometry file is defined
macro_path = os.path.join(path, 'mMR_Geometry.mac')
# Get information dictionary about the scanner
info = gate.get_detector_info(path = macro_path,
    mean_interaction_depth=9, min_rsector_difference=0)
# Paths to all ROOT files containing data
paths = [os.path.join(path, f'gate_simulation/simple_phantom/mMR_voxBrain_withSimplePhantom_{i}.root') for i in range(1, 55)]

info

# %% Normalization Correction
if LOAD_FROM_ROOT:
    normalization_paths = [os.path.join(path, f'normalization_scan/mMR_Norm_{i}.root') for i in range(1,37)]

    # Get normalization weights for all possible detector ID pairs
    normalization_weights = gate.get_normalization_weights_cylinder_calibration(
        normalization_paths,
        info,
        cylinder_radius = 318, # mm (radius of calibration cylindrical shell,
        include_randoms=False 
    )

    torch.save(normalization_weights, os.path.join(path, 'normalization_weights.pt'))
normalization_weights = torch.load(os.path.join(path, 'normalization_weights.pt'))

# %% Primary-Only Reconstruction
if LOAD_FROM_ROOT:
    detector_ids = gate.get_detector_ids_from_root(
        paths,
        info,
        include_randoms=False,
        include_scatters=False)
    torch.save(detector_ids, os.path.join(path, 'detector_ids_primary_only.pt'))
detector_ids = torch.load(os.path.join(path, 'detector_ids_primary_only.pt'))

if LOAD_FROM_ROOT:
    detector_ids_randoms_true = gate.get_detector_ids_from_root(
        paths,
        info,
        randoms_only=True)
    detector_ids_scatters_true = gate.get_detector_ids_from_root(
        paths,
        info,
        scatters_only=True)
    torch.save(detector_ids_randoms_true, os.path.join(path, 'detector_ids_randoms_true.pt'))
    torch.save(detector_ids_scatters_true, os.path.join(path, 'detector_ids_scatters_true.pt'))
detector_ids_randoms_true = torch.load(os.path.join(path, 'detector_ids_randoms_true.pt'))
detector_ids_scatters_true = torch.load(os.path.join(path, 'detector_ids_scatters_true.pt'))

# Specify object space for reconstruction
# Specify object space for reconstruction
object_meta = ObjectMeta(
    dr=(2,2,2), #mm
    shape=(128,128,96) #voxels
)
# Get projection space metadata from PET geometry information dictionary
proj_meta = PETLMProjMeta(
    detector_ids = detector_ids, # list of all detected event detector ID pairs
    info = info,
    weights_sensitivity=normalization_weights
    )
# Get attenuation map and PSF transform from the associated phantom
atten_map = gate.get_aligned_attenuation_map(os.path.join(path, 'gate_simulation/simple_phantom/umap_mMR_brainSimplePhantom.hv'), object_meta).to(pytomography.device)
psf_transform = GaussianFilter(3.) # 3mm gaussian blurring
# Create system matrix
system_matrix = PETLMSystemMatrix(
    object_meta,
    proj_meta,
    obj2obj_transforms = [psf_transform],
    attenuation_map = atten_map,
    N_splits=8,
)
# Create likelihood. For listmode reconstruction, projections don't need to be provided, since all detection events are stored in proj_meta
likelihood = PoissonLogLikelihood(
    system_matrix
)
# Initialize reconstruction algorithm
recon_algorithm = OSEM(likelihood)
# Reconstruct
recon_primaryonly = recon_algorithm(n_iters=4, n_subsets=14)

# %% Reconstructing With Random/Scatter Estimation
if LOAD_FROM_ROOT:
    detector_ids = gate.get_detector_ids_from_root(
        paths,
        info)
    torch.save(detector_ids, os.path.join(path, 'detector_ids_all_events.pt'))
detector_ids = torch.load(os.path.join(path, 'detector_ids_all_events.pt'))

# %% Randoms
if LOAD_FROM_ROOT:
    detector_ids_delays = gate.get_detector_ids_from_root(
        paths,
        info,
        substr = 'delay')
    torch.save(detector_ids_delays, os.path.join(path, 'detector_ids_delays.pt'))
detector_ids_delays= torch.load(os.path.join(path, 'detector_ids_delays.pt'))

sinogram_delays  = gate.listmode_to_sinogram(detector_ids_delays , info)
sinogram_delays  = gate.smooth_randoms_sinogram(sinogram_delays , info, sigma_r=4, sigma_theta=4, sigma_z=4)
lm_delays = shared.sinogram_to_listmode(detector_ids, sinogram_delays , info)

# %% Scatters
atten_map = gate.get_aligned_attenuation_map(os.path.join(path, 'gate_simulation/simple_phantom/umap_mMR_brainSimplePhantom.hv'), object_meta).to(pytomography.device)
normalization_weights = torch.load(os.path.join(path, 'normalization_weights.pt'))
proj_meta = PETLMProjMeta(
    detector_ids[:,:2],
    info,
    weights_sensitivity=normalization_weights
    )
psf_transform = GaussianFilter(3.)
system_matrix = PETLMSystemMatrix(
       object_meta,
       proj_meta,
       obj2obj_transforms = [psf_transform],
       N_splits=10,
       attenuation_map=atten_map.to(pytomography.device),
)

# Get additive term (without scatter term):
lm_sensitivity = system_matrix._compute_sensitivity_projection(all_ids=False)
additive_term = lm_delays / lm_sensitivity
additive_term[additive_term.isnan()] = 0 # remove NaN values
# Recon
likelihood = PoissonLogLikelihood(
        system_matrix,
        additive_term = additive_term
    )
recon_algorithm = OSEM(likelihood)
recon_without_scatter_estimation = recon_algorithm(4,14)

sinogram_scatter = sss.get_sss_scatter_estimate(
    object_meta = object_meta,
    proj_meta = proj_meta,
    pet_image = recon_without_scatter_estimation,
    attenuation_image = atten_map,
    system_matrix = system_matrix,
    proj_data = None, # assumes listmode
    image_stepsize = 4,
    attenuation_cutoff = 0.004,
    sinogram_interring_stepsize = 4,
    sinogram_intraring_stepsize = 4,
    sinogram_random = sinogram_delays
    )
# Now convert to listmode
lm_scatter = shared.sinogram_to_listmode(proj_meta.detector_ids, sinogram_scatter, proj_meta.info)

additive_term = (lm_scatter + lm_delays) / lm_sensitivity
additive_term[additive_term.isnan()] = 0
likelihood = PoissonLogLikelihood(
        system_matrix,
        additive_term = additive_term
    )
recon_algorithm = OSEM(likelihood)
recon_lm_nontof = recon_algorithm(4,14)
