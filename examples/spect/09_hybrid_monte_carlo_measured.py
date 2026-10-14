"""Hybrid Monte Carlo (measured)

The same hybrid approach on measured data.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_spect_mc2.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% Hybrid-MC: DICOM Data
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Lu177-NEMA-SymT2")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Lu177-NEMA-SymT2")

import os
import matplotlib.pyplot as plt
import torch
import pytomography
from pytomography import io
from pytomography.projectors.SPECT import MonteCarloHybridSPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.likelihoods import MonteCarloHybridSPECTPoissonLogLikelihood
from pytomography.utils import simind_mc

path = DATA / 'SPECT'

file_NM = os.path.join(path, 'Lu177-NEMA-SymT2', 'projection_data.dcm')
path_CT = os.path.join(path, 'Lu177-NEMA-SymT2', 'CT')
files_CT = [os.path.join(path_CT, f) for f in os.listdir(path_CT)]
projections, object_meta, proj_meta = io.SPECT.read_dicom(file_NM)
photon_keV = proj_meta.energy_windows[0].photon_keV  # the 208 keV window
# 140 keV map is needed for MC forward projector
amap140 = io.attenuation_map(files_CT, object_meta, energy_keV=140.5)
# 208keV is needed for analytical back projection
amap208 = io.attenuation_map(files_CT, object_meta, energy_keV=photon_keV)
att_transform = SPECTAttenuationTransform(amap208)
collimator_name = 'SY-ME'
intrinsic_fwhm_mm = 3.8 # mm
psf_meta = io.SPECT.collimator_psf(
    collimator_name,
    photon_keV,
    intrinsic_fwhm_mm=intrinsic_fwhm_mm
)
psf_transform = SPECTPSFTransform(psf_meta)

energy_window_params = simind_mc.get_energy_window_params_dicom(file_NM) 
energy_window_params

isotope_names = ['lu177'] # isotope we want to simulate
isotope_ratios = [1] # ratio of isotopes (in this case only 1)
collimator_type = 'SY-ME' # collimator type to use
cover_thickness_mm = 1.0 # cover thickness in mm (aluminum assumed)
backscatter_thickness_mm = 66.0 # backscatter thickness in mm (pyrex)
advanced_energy_resolution_model='siemens'
# if the energy resolution model above is not used, then
# the energy resolution needs to be provided at 140keV in
# units of percent.
energy_resolution_140keV = 10 # %
crystal_thickness_mm = 9.525 # thickness of the crystal in mm (NaI)

# 200 million events take about 20 minutes on 90 CPU cores, and 45 minutes on 24. Fewer events run faster
# but give a noisier Monte Carlo estimate.
n_events = 200e6
n_parallel = max(1, os.cpu_count() - 4)  # SIMIND processes run at once, one per core; 4 cores left for the rest of the computer

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
    crystal_thickness_mm=crystal_thickness_mm,
    cover_thickness_mm=cover_thickness_mm,
    backscatter_thickness_mm=backscatter_thickness_mm,
    advanced_energy_resolution_model=advanced_energy_resolution_model,
    advanced_collimator_modeling=True,
    n_events=n_events,
    n_parallel=n_parallel
)

likelihood = MonteCarloHybridSPECTPoissonLogLikelihood(system_matrix, projections[0])
algorithm = OSEM(likelihood)
print(algorithm.estimate_memory(n_subsets=16))
recon_MC = algorithm(n_iters=1, n_subsets=16)

# %% Save the images
saved = io.save_dicom(recon_MC, OUTPUT / 't_spect_mc2' / 'osem_monte_carlo', object_meta, reference=file_NM,
                      units='MBq.s', series_description='OSEM 1x16, Monte Carlo', overwrite=True)
