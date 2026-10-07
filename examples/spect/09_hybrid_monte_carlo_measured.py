"""Hybrid Monte Carlo (measured)

The same hybrid approach on measured data.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_spect_mc2.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% Hybrid-MC: DICOM Data
import os
from pathlib import Path

# Tutorial data: the folder set by the PYTOMOGRAPHY_DATA environment variable (see Tutorial data in the docs)
DATA = Path(os.environ.get("PYTOMOGRAPHY_DATA", "~/pytomography_data")).expanduser()
# Results go here, never into the data folder
OUTPUT = Path(os.environ.get("PYTOMOGRAPHY_OUTPUT", "pytomography_outputs")).expanduser() / "SPECT/Lu177-NEMA-SymT2"
OUTPUT.mkdir(parents=True, exist_ok=True)

import os
import sys
import matplotlib.pyplot as plt
import torch
import itk
import numpy as np
from torch.nn.functional import avg_pool3d
import pytomography
from pytomography.io.SPECT import simind, dicom
from pytomography.io.SPECT.shared import subsample_projections_and_modify_metadata, subsample_amap, subsample_projections
from pytomography.projectors.SPECT import MonteCarloHybridSPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.likelihoods import MonteCarloHybridSPECTPoissonLogLikelihood
from pytomography.utils import simind_mc

path = DATA / 'SPECT'

file_NM = os.path.join(path, 'Lu177-NEMA-SymT2', 'projection_data.dcm')
path_CT = os.path.join(path, 'Lu177-NEMA-SymT2', 'CT')
files_CT = [os.path.join(path_CT, f) for f in os.listdir(path_CT)] 
# 140 keV map is needed for MC forward projector
amap140 = dicom.get_attenuation_map_from_CT_slices(files_CT, file_NM, E_SPECT=140.5)
# 208keV is needed for analytical back projection
amap208 = dicom.get_attenuation_map_from_CT_slices(files_CT, file_NM, E_SPECT=208)
projections = dicom.get_projections(file_NM)
object_meta, proj_meta = dicom.get_metadata(file_NM)
att_transform = SPECTAttenuationTransform(amap208)
collimator_name = 'SY-ME'
energy_kev = 208 #keV
intrinsic_resolution=0.38 #mm
psf_meta = dicom.get_psfmeta_from_scanner_params(
    collimator_name,
    energy_kev,
    intrinsic_resolution=intrinsic_resolution
)
psf_transform = SPECTPSFTransform(psf_meta)

energy_window_params = simind_mc.get_energy_window_params_dicom(file_NM) 
energy_window_params

isotope_names = ['lu177'] # isotope we want to simulate
isotope_ratios = [1] # ratio of isotopes (in this case only 1)
collimator_type = 'SY-ME' # collimator type to use
cover_thickness = 0.1 # cover thickness in cm (aluminum assumed)
backscatter_thickness = 6.6 # backscatter thickness in cm (pyrex)
advanced_energy_resolution_model='siemens' 
# if the energy resolution model above is not used, then
# the energy resolution needs to be provided at 140keV in
# units of percent.
energy_resolution_140keV = 10 # %
crystal_thickness = 0.9525 # thickness in crystal in cm (NaI)

# 200 million events take about 20 minutes on 90 CPU cores, and 45 minutes on 24. Fewer events run faster
# but give a noisier Monte Carlo estimate.
n_events = 200e6
n_parallel = os.cpu_count()  # SIMIND processes run at once, one per CPU core

system_matrix = MonteCarloHybridSPECTSystemMatrix(
    object_meta,
    proj_meta,
    obj2obj_transforms=[att_transform, psf_transform],
    proj2proj_transforms=[],
    attenuation_map_140keV=amap140,
    energy_window_params=energy_window_params,
    primary_window_idx=0, # based on the 208keV window from energy_window_params
    isotope_names=isotope_names,
    isotope_ratios=isotope_ratios,
    collimator_type=collimator_type,
    crystal_thickness=crystal_thickness,
    cover_thickness=cover_thickness,
    backscatter_thickness=backscatter_thickness,
    advanced_energy_resolution_model=advanced_energy_resolution_model,
    advanced_collimator_modeling=True,
    n_events=n_events,
    n_parallel=n_parallel
)

likelihood = MonteCarloHybridSPECTPoissonLogLikelihood(system_matrix, projections[0])
algorithm = OSEM(likelihood)
recon_MC = algorithm(n_iters=1, n_subsets=16)
