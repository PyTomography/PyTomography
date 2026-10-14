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
import numpy as np
from pytomography.io.SPECT import dicom
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.utils import print_collimator_parameters
import matplotlib.pyplot as plt
import pydicom
import shutil

# Paths are set in the data cell at the top of this tutorial
PATH = DATA / 'SPECT'
PATH = os.path.join(PATH, 'Lu177-NEMA-SymT2')

# %% Part 1: Opening Data
# initialize the `path`` variable below to specify the location of the required data
path_CT = os.path.join(PATH, 'CT')
files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
file_NM = os.path.join(PATH, 'projection_data.dcm')

object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=0)

dicom.print_energy_window_info(file_NM)

photopeak = dicom.get_projections(file_NM, index_peak=0)
photopeak.shape

scatter = dicom.get_energy_window_scatter_estimate(file_NM, index_peak=0, index_lower=1, index_upper=2)

# %% Attenuation Transform
att_transform = SPECTAttenuationTransform(filepath=files_CT)

att_transform.configure(object_meta, proj_meta)
attenuation_map = att_transform.attenuation_map

sample_slice = attenuation_map.cpu()[:,70].T

# %% PSF Modeling
#print_collimator_parameters()

collimator_name = 'SY-ME'
energy_kev = 208 #keV
intrinsic_resolution=0.38 #cm
psf_meta = dicom.get_psfmeta_from_scanner_params(
    collimator_name,
    energy_kev,
    intrinsic_resolution=intrinsic_resolution
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

reconstructed_object = reconstruction_algorithm(n_iters=4, n_subsets=8)

ds_recon = pydicom.dcmread(os.path.join(PATH, 'scanner_recon.dcm'))
recon_vendor = ds_recon.pixel_array / proj_meta.num_projections
recon_vendor = np.transpose(recon_vendor, (2,1,0))

idx_z = 61
slice_pytomography = reconstructed_object.cpu()[:,:,idx_z].T
slice_vendor = recon_vendor[:,:,idx_z].T

# %% Saving Data
# Paths are set in the data cell at the top of this tutorial
save_path = OUTPUT / 'Pytomo-Recon'
# Code only works if folder doesnt exist, so delete it if present
if os.path.exists(save_path) and os.path.isdir(save_path):
    shutil.rmtree(save_path)
# Save
dicom.save_dcm(
    save_path = save_path,
    object = reconstructed_object,
    file_NM = file_NM,
    recon_name = 'OSEM_4it_8ss')
