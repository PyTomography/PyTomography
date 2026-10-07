"""Reconstruction algorithms

OSEM, BSREM, OSMAPOSL and KEM side by side on the same data.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_algorithms.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% Algorithms
import os
from pathlib import Path

# Tutorial data: the folder set by the PYTOMOGRAPHY_DATA environment variable (see Tutorial data in the docs)
DATA = Path(os.environ.get("PYTOMOGRAPHY_DATA", "~/pytomography_data")).expanduser()
# Results go here, never into the data folder
OUTPUT = Path(os.environ.get("PYTOMOGRAPHY_OUTPUT", "pytomography_outputs")).expanduser() / "SPECT/SIMIND-Jaszak"
OUTPUT.mkdir(parents=True, exist_ok=True)

import inspect
import matplotlib.pyplot as plt
import torch
from pytomography.io.SPECT import simind, dicom
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM, OSMAPOSL, BSREM, KEM
import matplotlib.pyplot as plt
import torch
from pytomography.priors import RelativeDifferencePrior
from pytomography.priors import TopNAnatomyNeighbourWeight
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.shared import KEMTransform
from pytomography.projectors.shared import KEMSystemMatrix
import os

# Paths are set in the data cell at the top of this tutorial
PATH = DATA / 'SPECT'

TYPE = 'DICOM' # DICOM or SIMIND

if TYPE=='SIMIND':
    data_path = os.path.join(PATH, 'SIMIND-Jaszak', 'lu177_SYME_jaszak')
    IDX_AXIAL = 64 # for plotting
    photopeak_path = os.path.join(data_path,'tot_w4.h00')
    lower_path = os.path.join(data_path, 'tot_w5.h00')
    upper_path = os.path.join(data_path, 'tot_w6.h00')
    path_amap = os.path.join(data_path,'amap.hct')
    object_meta, proj_meta = simind.get_metadata(photopeak_path)
    photopeak = simind.get_projections(photopeak_path)
    activity = 1000 # MBq 
    time_per_proj = 15 # s
    photopeak = torch.poisson(photopeak * activity * time_per_proj)
    lower = simind.get_projections(lower_path)
    upper = simind.get_projections(upper_path)
    lower_realization = torch.poisson(lower * activity * time_per_proj)
    upper_realization = torch.poisson(upper * activity * time_per_proj)
    ww_peak, ww_lower, ww_upper = [simind.get_energy_window_width(path) for path in [photopeak_path, lower_path, upper_path]]
    scatter = simind.compute_EW_scatter(lower_realization, upper_realization , ww_lower, ww_upper, ww_peak)
    amap = simind.get_attenuation_map(path_amap)
    psf_meta = simind.get_psfmeta_from_header(photopeak_path)
    att_transform = SPECTAttenuationTransform(amap)
    psf_transform = SPECTPSFTransform(psf_meta)
elif TYPE=='DICOM':
    data_path = os.path.join(PATH, 'Lu177-NEMA-SymT2')
    IDX_AXIAL = 61 # for plotting
    path_CT = os.path.join(data_path, 'CT')
    files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
    file_NM = os.path.join(data_path, 'projection_data.dcm')
    object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=0)
    photopeak = dicom.get_projections(file_NM, index_peak=0)
    scatter = dicom.get_energy_window_scatter_estimate(file_NM, index_peak=0, index_lower=1, index_upper=2)
    att_transform = SPECTAttenuationTransform(filepath=files_CT)
    att_transform.configure(object_meta, proj_meta)
    amap = att_transform.attenuation_map
    collimator_name = 'SY-ME'
    energy_kev = 208 #keV
    intrinsic_resolution=0.38 #mm
    psf_meta = dicom.get_psfmeta_from_scanner_params(
        collimator_name,
        energy_kev,
        intrinsic_resolution=intrinsic_resolution
    )
    psf_transform = SPECTPSFTransform(psf_meta)
    
system_matrix = SPECTSystemMatrix(
    obj2obj_transforms = [att_transform,psf_transform],
    proj2proj_transforms = [],
    object_meta = object_meta,
    proj_meta = proj_meta
)
likelihood = PoissonLogLikelihood(
    system_matrix = system_matrix,
    projections = photopeak,
    additive_term = scatter
)

# %% 4.1 OSEM
recon_algorithm = OSEM(likelihood)
recon_OSEM = recon_algorithm(n_iters = 4, n_subsets = 8)

# %% 4.2 OSMAPOSL
weight_top8anatomy = TopNAnatomyNeighbourWeight(amap, N_neighbours=8)
prior_rdpap = RelativeDifferencePrior(beta=0.3, gamma=2, weight=weight_top8anatomy)
# to use all nearest neighbours (non attenuation map based) use the code below:
# prior_rdp = RelativeDifferencePrior(beta=0.3, gamma=2)
recon_algorithm_osmaposl = OSMAPOSL(
    likelihood = likelihood,
    prior = prior_rdpap)
recon_osmaposl = recon_algorithm_osmaposl(n_iters = 40, n_subsets = 8)

# %% 4.3 BSREM
weight_top8anatomy = TopNAnatomyNeighbourWeight(amap, N_neighbours=8)
prior_rdpap = RelativeDifferencePrior(beta=0.3, gamma=2, weight=weight_top8anatomy)
recon_algorithm_bsrem = BSREM(
    likelihood = likelihood,
    prior = prior_rdpap,
    relaxation_sequence = lambda n: 1/(n/50+1))
recon_bsrem = recon_algorithm_bsrem(40,8)

# %% 4.4 KEM
kem_transform = KEMTransform(
    support_objects=[amap],
    support_kernels_params=[[0.005]],
    distance_kernel_params=[0.4],
    top_N = 40,
    kernel_on_gpu=True
    )

system_matrix_kem = KEMSystemMatrix(system_matrix, kem_transform)

likelihood_kem = PoissonLogLikelihood(system_matrix_kem, photopeak, additive_term=scatter)

recon_algorithm_kem = KEM(likelihood_kem)
recon_kem = recon_algorithm_kem(40,8)

# %% Comparison
recons = [recon_OSEM, recon_osmaposl, recon_bsrem, recon_kem]
recon_names = ['OSEM', 'OSMAPOSL', 'BSREM', 'KEM']
