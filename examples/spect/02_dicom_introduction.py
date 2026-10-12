"""DICOM introduction

Reconstruct a phantom exported from a clinical scanner and compare with the vendor image.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_dicomdata.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Lu177-NEMA-SymT2")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Lu177-NEMA-SymT2")

import os
from pytomography import io
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.utils import print_collimator_parameters
import matplotlib.pyplot as plt

# Paths are set in the data cell at the top of this tutorial
PATH = DATA / 'SPECT'
PATH = os.path.join(PATH, 'Lu177-NEMA-SymT2')

# %% Part 1: Opening Data
# initialize the `path`` variable below to specify the location of the required data
path_CT = os.path.join(PATH, 'CT')
files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
file_NM = os.path.join(PATH, 'projection_data.dcm')

projections, object_meta, proj_meta = io.SPECT.read_dicom(file_NM)

for i, window in enumerate(proj_meta.energy_windows):
    print(f'Index {i}:   Name: "{window.name}", Energies: {window.lower_keV:.1f}-{window.upper_keV:.1f} keV')

photopeak = projections[0]
photopeak.shape

scatter = io.SPECT.energy_window_scatter(projections, proj_meta, peak=0, lower=1, upper=2)

# %% Attenuation Transform
attenuation_map = io.attenuation_map(files_CT, object_meta, energy_keV=proj_meta.energy_windows[0].photon_keV)
att_transform = SPECTAttenuationTransform(attenuation_map)

sample_slice = attenuation_map.cpu()[:,70].T

# %% PSF Modeling
#print_collimator_parameters()

collimator_name = 'SY-ME'
photon_keV = proj_meta.energy_windows[0].photon_keV  # 208 keV
intrinsic_fwhm_mm = 3.8  # mm
psf_meta = io.SPECT.collimator_psf(
    collimator_name,
    photon_keV,
    intrinsic_fwhm_mm=intrinsic_fwhm_mm
)
print(psf_meta)

psf_transform = SPECTPSFTransform(psf_meta)

# %% System Matrix
system_matrix = SPECTSystemMatrix(
        obj2obj_transforms = [att_transform,psf_transform],
        proj2proj_transforms = [],
        object_meta = object_meta,
        proj_meta = proj_meta)

# %% Likelihood
likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter)

# %% Reconstruct the object
reconstruction_algorithm = OSEM(likelihood)
print(reconstruction_algorithm.estimate_memory(n_subsets=8))

reconstructed_object = reconstruction_algorithm(n_iters=4, n_subsets=8)

recon_vendor = io.load_image(os.path.join(PATH, 'scanner_recon.dcm')).data / proj_meta.num_projections  # on our grid

idx_z = 61
slice_pytomography = reconstructed_object.cpu()[:,:,idx_z].T
slice_vendor = recon_vendor[:,:,idx_z].T

# %% Save the images
saved = io.save_dicom(reconstructed_object, OUTPUT / 't_dicomdata' / 'osem', object_meta, reference=file_NM,
                      series_description='OSEM 4x8', overwrite=True)
