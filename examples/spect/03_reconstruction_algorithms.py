"""Reconstruction algorithms

OSEM, BSREM, OSMAPOSL and KEM side by side on the same data.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_algorithms.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% Algorithms
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/SIMIND-Jaszak")
datasets.fetch("SPECT/Lu177-NEMA-SymT2")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/SIMIND-Jaszak")

import os
import matplotlib.pyplot as plt
import torch
import pytomography
from pytomography import io
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM, OSMAPOSL, BSREM, KEM
from pytomography.priors import RelativeDifferencePrior
from pytomography.priors import TopNAnatomyNeighbourWeight
from pytomography.transforms.shared import KEMTransform
from pytomography.projectors.shared import KEMSystemMatrix

# Paths are set in the data cell at the top of this tutorial
PATH = DATA / 'SPECT'

# Memory: this tutorial peaks at about 7 GB of RAM, during KEM, whose kernel (1 GB) is built on the CPU, a chunk of
# voxels at a time, and then kept on the GPU, which on Windows counts as RAM too. Each reconstruction prints its estimate
# before it runs (estimate_memory): about 4.4 GB for OSEM, most of it PyTorch and CUDA themselves, and 6 GB for KEM.
# RAM_GB changes nothing here: SPECT reconstructions this size never need to split their work. kernel_on_gpu=False
# keeps KEM's kernel in RAM instead of GPU memory, three times slower.
RAM_GB = 25
pytomography.set_memory_budget(RAM_GB)

TYPE = 'DICOM' # DICOM or SIMIND

if TYPE=='SIMIND':
    data_path = os.path.join(PATH, 'SIMIND-Jaszak', 'lu177_SYME_jaszak')
    IDX_AXIAL = 64 # for plotting
    # the 208 keV photopeak (tot_w1) and its scatter windows (tot_w2, tot_w3): the energy of amap.hct
    headers = [os.path.join(data_path, f'tot_w{i}.h00') for i in (1, 2, 3)]
    projections, object_meta, proj_meta = io.SPECT.read_simind(headers)
    activity = 1000 # MBq
    time_per_proj = 15 # s
    projections = torch.poisson(projections * activity * time_per_proj)
    photopeak = projections[0]
    scatter = io.SPECT.energy_window_scatter(projections, proj_meta, peak=0, lower=1, upper=2)
    amap = io.attenuation_map(os.path.join(data_path, 'amap.hct'), object_meta,
                              energy_keV=proj_meta.energy_windows[0].photon_keV, layout='mu')
    psf_meta = io.SPECT.simind_psf(headers[0])
elif TYPE=='DICOM':
    data_path = os.path.join(PATH, 'Lu177-NEMA-SymT2')
    IDX_AXIAL = 61 # for plotting
    path_CT = os.path.join(data_path, 'CT')
    files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
    file_NM = os.path.join(data_path, 'projection_data.dcm')
    projections, object_meta, proj_meta = io.SPECT.read_dicom(file_NM)
    photopeak = projections[0]  # the 208 keV window
    scatter = io.SPECT.energy_window_scatter(projections, proj_meta, peak=0, lower=1, upper=2)
    amap = io.attenuation_map(files_CT, object_meta, energy_keV=proj_meta.energy_windows[0].photon_keV)
    psf_meta = io.SPECT.collimator_psf('SY-ME', proj_meta.energy_windows[0].photon_keV, intrinsic_fwhm_mm=3.8)
att_transform = SPECTAttenuationTransform(amap)
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
print(recon_algorithm.estimate_memory(n_subsets=8))
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
    support_kernels_params=[[0.0005]],  # 1/mm, the units of the attenuation map
    distance_kernel_params=[4],  # mm
    top_N = 40,
    kernel_on_gpu=True
    )

system_matrix_kem = KEMSystemMatrix(system_matrix, kem_transform)

likelihood_kem = PoissonLogLikelihood(system_matrix_kem, photopeak, additive_term=scatter)

recon_algorithm_kem = KEM(likelihood_kem)
print(recon_algorithm_kem.estimate_memory(n_subsets=8))
recon_kem = recon_algorithm_kem(40,8)

# %% Comparison
recons = [recon_OSEM, recon_osmaposl, recon_bsrem, recon_kem]
recon_names = ['OSEM', 'OSMAPOSL', 'BSREM', 'KEM']

# %% Save the images
for name, recon in zip(recon_names, recons):
    io.save_dicom(recon, OUTPUT / 't_algorithms' / name, object_meta, reference=file_NM if TYPE == 'DICOM' else None,
                  units='counts', series_description=name, overwrite=True)
