"""Hybrid Monte Carlo (simulated)

Monte Carlo scatter estimated inside the reconstruction loop with SIMIND.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_spect_mc.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% Hybrid-MC: SIMIND Data
import os
from pathlib import Path

# Tutorial data: the folder set by the PYTOMOGRAPHY_DATA environment variable (see Tutorial data in the docs)
DATA = Path(os.environ.get("PYTOMOGRAPHY_DATA", "~/pytomography_data")).expanduser()
# Results go here, never into the data folder
OUTPUT = Path(os.environ.get("PYTOMOGRAPHY_OUTPUT", "pytomography_outputs")).expanduser() / "SPECT/SIMIND-Jaszak"
OUTPUT.mkdir(parents=True, exist_ok=True)

import os
import torch
from pytomography.io.SPECT import simind
from pytomography.projectors.SPECT import SPECTSystemMatrix, MonteCarloHybridSPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood, MonteCarloHybridSPECTPoissonLogLikelihood
from pytomography.utils import simind_mc
import matplotlib.pyplot as plt

# Paths are set in the data cell at the top of this tutorial
PATH = DATA / 'SPECT' / 'SIMIND-Jaszak'

# %% Part 1: Opening the data we want to reconstruct
ISOTOPE='lu177'
if ISOTOPE == 'pb212': # NEED TO RUN
    isotopes_decay = ['pb212', 'bi212', 'tl208']
    isotopes_ratios = [1,1.105,0.4] # bateman equation equilibrium
    idx_peak, idx_lower, idx_upper = 5, 4, 6
    E_window_bounds = [[204.6,215.1],[215.1,262.9],[262.9,276.4]] #lwr,peak,upr
    E = 236
    use_TEW_for_conventional = True
    activity_conc = 50
if ISOTOPE == 'lu177': # NEED TO RUN
    isotopes_decay = ['lu177']
    isotopes_ratios = [1]
    idx_peak, idx_lower, idx_upper = 5, 4, 6
    E_window_bounds = [[166.4,187.2],[187.2,228.8],[228.8,249.6]] #lwr,peak,upr
    E = 208
    use_TEW_for_conventional = True
    activity_conc = 1
if ISOTOPE == 'y90': # NEED TO RUN
    isotopes_decay = ['y90']
    isotopes_ratios = [1]
    idx_peak, idx_lower, idx_upper = 2,1,3
    E_window_bounds = [[50,100],[100,200],[200,300]] #lwr,peak,upr
    E = 150
    use_TEW_for_conventional = False
    activity_conc = 10
dT = 15

files_NM = [[os.path.join(PATH, f'{isotope}', f'tot_w{i}.h00') for isotope in isotopes_decay] for i in [idx_lower, idx_peak, idx_upper]]
object_meta, proj_meta = simind.get_metadata(files_NM[0][0])
activity_concs = [activity_conc * ratio for ratio in isotopes_ratios]
projections = simind.get_projections(files_NM, activity_concs)
projections *= dT
projections = torch.poisson(projections)
# the projections are ordered as [lower, peak, upper]
photopeak = projections[1]
widths = torch.tensor([E2-E1 for E1,E2 in E_window_bounds])
if use_TEW_for_conventional:
    additive_TEW = simind.compute_EW_scatter(projections[0], projections[2], widths[0], widths[2], widths[1])
else:
    additive_TEW = photopeak*0

# We need two attenuation maps for MC monte carlo. The 140keV map is used as input
# to the MC forward simulation. The amap at the isotope energy is used to build 
# an attenuation transform that is used in the analytical back projection
amap140keV =  simind.get_attenuation_map(os.path.join(PATH,'attenuation_maps','amap140.hct'))
amapIsotopeEnergy = simind.get_attenuation_map(os.path.join(PATH,'attenuation_maps',f'amap{E}.hct'))
att_transform = SPECTAttenuationTransform(attenuation_map=amapIsotopeEnergy)

psf_meta = simind.get_psfmeta_from_header(files_NM[1][0])
psf_transform = SPECTPSFTransform(psf_meta)

# %% Part 2: Building the MC system matrix
# note: the collimator name is the one I used to generate the
# projection data
if ISOTOPE == 'pb212':
    isotope_names = ['pb212', 'bi212', 'tl208']
    isotope_ratios = [1, 1.105, 0.4]
    collimator_type = 'SY-HE'
elif ISOTOPE == 'lu177':
    isotope_names = ['lu177']
    isotope_ratios = [1]
    collimator_type = 'SY-ME'
elif ISOTOPE == 'y90':
    isotope_names = ['y90']
    isotope_ratios = [1]
    collimator_type = 'SY-HE'

cover_thickness = 0.1 # assumed to be aluminum
backscatter_thickness = 6.6 # assumed to be pyrex
# the argument below is optional, only siemens currently implemented.
# if not provided, then assumes energy resolution is proportional to 
# 1/sqrt(E)
advanced_energy_resolution_model='siemens' 
# if the energy resolution model above is not used, then
# the energy resolution needs to be provided at 140keV in
# units of percent.
energy_resolution_140keV = 10 # %
crystal_thickness = 0.9525  # assumed to be NaI

energy_window_params = simind_mc.get_energy_window_params_simind(files_NM[1])
energy_window_params

# 200 million events take about 20 minutes on 90 CPU cores, and 45 minutes on 24. Fewer events run faster
# but give a noisier Monte Carlo estimate.
n_events = 200e6
n_parallel = os.cpu_count()  # SIMIND processes run at once, one per CPU core

system_matrix = MonteCarloHybridSPECTSystemMatrix(
        object_meta,
        proj_meta,
        obj2obj_transforms=[att_transform, psf_transform],
        proj2proj_transforms=[],
        attenuation_map_140keV=amap140keV,
        energy_window_params=energy_window_params,
        primary_window_idx=0, # index of energy_window_params to use
        isotope_names=isotope_names,
        isotope_ratios=isotope_ratios,
        collimator_type=collimator_type,
        crystal_thickness=crystal_thickness,
        cover_thickness=cover_thickness,
        backscatter_thickness=backscatter_thickness,
        advanced_energy_resolution_model=advanced_energy_resolution_model,
        advanced_collimator_modeling=True, # include septal penetration/scatter
        n_events=n_events,
        n_parallel=n_parallel
    )

likelihood = MonteCarloHybridSPECTPoissonLogLikelihood(system_matrix, photopeak)
algorithm = OSEM(likelihood)
# for now just 1 iteration but you can change. Takes about 20min to run with 90CPU cores
recon_MC = algorithm(n_iters=1, n_subsets=16)

system_matrix = SPECTSystemMatrix(
        obj2obj_transforms=[att_transform, psf_transform],
        proj2proj_transforms=[],
        object_meta=object_meta,
        proj_meta=proj_meta,
    )
likelihood = PoissonLogLikelihood(system_matrix, photopeak, additive_TEW)
algorithm = OSEM(likelihood)
recon_analytical = algorithm(n_iters=1, n_subsets=16)
