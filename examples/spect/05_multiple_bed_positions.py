"""Multiple bed positions

Reconstruct several bed positions and stitch them into one volume.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_dicommultibed.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% DICOM Multiple Bed Positions
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Lu177-PSMA-GEDisc")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Lu177-PSMA-GEDisc")

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
import torch
import shutil

# Paths are set in the data cell at the top of this tutorial
save_path = DATA / 'SPECT'

# %% Part 1: Opening Data
files_NM = [
    os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'bed1_projections.dcm'),
    os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'bed2_projections.dcm'),
]
path_CT = os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'CT')
files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]

dicom.print_energy_window_info(files_NM[0])

projections_upper_fov = dicom.get_projections(files_NM[0])
projections_lower_fov = dicom.get_projections(files_NM[1])

# %% Part 2: Reconstruction
object_meta, proj_meta = dicom.get_metadata(files_NM[0], index_peak=1)

def reconstruct_singlebed(i):
    # Change these depending on your file:
    index_peak = 1
    index_lower = 3
    index_upper = 2
    projections = projectionss[i]
    file_NM = files_NM[i]
    object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=index_peak)
    photopeak = projections[index_peak]
    scatter = dicom.get_energy_window_scatter_estimate_projections(file_NM, projections, index_peak, index_lower, index_upper)
    # Build system matrix
    attenuation_map = dicom.get_attenuation_map_from_CT_slices(files_CT, file_NM, index_peak=index_peak)
    psf_meta = dicom.get_psfmeta_from_scanner_params('GI-MEGP', energy_keV=208)
    att_transform = SPECTAttenuationTransform(attenuation_map)
    psf_transform = SPECTPSFTransform(psf_meta)
    # Create system matrix
    system_matrix = SPECTSystemMatrix(
        obj2obj_transforms = [att_transform,psf_transform],
        proj2proj_transforms= [],
        object_meta = object_meta,
        proj_meta = proj_meta)
    likelihood = PoissonLogLikelihood(system_matrix, photopeak, additive_term=scatter)
    reconstruction_algorithm = OSEM(likelihood)
    return reconstruction_algorithm(n_iters=4, n_subsets=8)

recon_upper = reconstruct_singlebed(0)
recon_lower = reconstruct_singlebed(1)

recon_stitched = dicom.stitch_multibed(
    recons=torch.stack([recon_upper, recon_lower]),
    files_NM = files_NM)

save_path = OUTPUT / 'pytomo_recon'
if os.path.exists(save_path) and os.path.isdir(save_path):
    shutil.rmtree(save_path)
dicom.save_dcm(
    save_path = save_path,
    object = recon_stitched,
    file_NM = files_NM[0],
    recon_name = 'OSEM_4it_8ss',
    scale_by_number_projections=True,
    single_dicom_file=True)
