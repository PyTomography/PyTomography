"""GE StarGuide

A 12-head CZT system with body-contour sweeps.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_starguide.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
# %% StarGuide Reconstruction
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Tc99m-NEMA-Starguide")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Tc99m-NEMA-Starguide")

import pydicom
import os
from pytomography import io
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.projectors.SPECT import StarGuideSystemMatrix

PATH = DATA / 'SPECT'

path_NM = os.path.join(PATH, 'Tc99m-NEMA-Starguide', 'NM_files')
# The folder holds two acquisitions, "NoFocus Cont" and "NoFocus SaS", each with a list-mode file. Pick the 12 files of
# "NoFocus Cont" by their series description: the order of os.listdir differs between operating systems.
files_NM = sorted(
    os.path.join(path_NM, f) for f in os.listdir(path_NM)
    if pydicom.dcmread(os.path.join(path_NM, f), stop_before_pixels=True).SeriesDescription == 'NoFocus Cont')

path_CT = os.path.join(PATH, 'Tc99m-NEMA-Starguide', 'CT_files')
files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
projections, object_meta, proj_meta = io.SPECT.read_starguide(files_NM, files_CT=files_CT, nearest_theta=0.1)
print(projections.shape)

attenuation_map = io.attenuation_map(files_CT, object_meta, energy_keV=proj_meta.energy_windows[0].photon_keV)
attenuation_transform = SPECTAttenuationTransform(attenuation_map, assume_padded = False)

psf_meta = io.SPECT.collimator_psf(
    'G8-LEHR', # According the the header, these are the collimator parameters
    proj_meta.energy_windows[0].photon_keV, # 140.5 keV: imaging of Tc-99m
    material='tungsten', # It is known that this is the material
    shape = 'square', # collimator hole shape
)
psf_transform = SPECTPSFTransform(psf_meta, assume_padded=False)

system_matrix = StarGuideSystemMatrix(
    object_meta=object_meta,
    proj_meta=proj_meta,
    obj2obj_transforms=[attenuation_transform, psf_transform],
    proj2proj_transforms=[]
)

photopeak = projections[0]
scatter = io.SPECT.energy_window_scatter(projections, proj_meta, peak=0, lower=1)

likelihood = PoissonLogLikelihood(system_matrix, photopeak, additive_term=scatter)
reconstruction_algorithm = OSEM(likelihood)
print(reconstruction_algorithm.estimate_memory(n_subsets=10))
recon_pytomography = reconstruction_algorithm(n_iters=10, n_subsets=10)

# %% Save the images
saved = io.save_dicom(recon_pytomography, OUTPUT / 't_starguide' / 'osem', object_meta,
                      reference=files_CT[0], series_description='OSEM 10x10', overwrite=True)
