"""Uncertainty estimation

Voxel and region uncertainty that comes out of the reconstruction itself.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_uncertainty_spect.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

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
from pytomography.algorithms import OSEM, PGAAMultiBedSPECT
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.utils import print_collimator_parameters
import matplotlib.pyplot as plt
import torch
from rt_utils import RTStructBuilder
from pytomography.callbacks import DataStorageCallback

save_path = DATA / 'SPECT'

file_NM = os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'bed2_projections.dcm')
path_CT = os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'CT')
files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=1)
projections = dicom.get_projections(file_NM)
photopeak = projections[1]
scatter, scatter_variance_estimate = dicom.get_energy_window_scatter_estimate(file_NM, index_peak=1, index_lower=3, index_upper=2, return_scatter_variance_estimate=True)
# Build system matrix
attenuation_map = dicom.get_attenuation_map_from_CT_slices(files_CT, file_NM, index_peak=1)
psf_meta = dicom.get_psfmeta_from_scanner_params('GI-MEGP', energy_keV=208)
att_transform = SPECTAttenuationTransform(attenuation_map)
psf_transform = SPECTPSFTransform(psf_meta)
system_matrix = SPECTSystemMatrix(
    obj2obj_transforms = [att_transform,psf_transform],
    proj2proj_transforms = [],
    object_meta = object_meta,
    proj_meta = proj_meta)
# Likelihood
likelihood = PoissonLogLikelihood(system_matrix, photopeak, additive_term=scatter, additive_term_variance_estimate=scatter_variance_estimate)
# Reconstruction algorithm
recon_algorithm = OSEM(likelihood)

data_storage_callback = DataStorageCallback(likelihood, torch.clone(recon_algorithm.object_prediction))

recon_OSEM = recon_algorithm(n_iters = 4, n_subsets = 8, callback=data_storage_callback)

maximum_intensity_projection = recon_OSEM.max(axis=1)[0].cpu()

file_RT = os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'segmentations.dcm')
rtstruct = RTStructBuilder.create_from(
        dicom_series_path=path_CT, 
        rt_struct_path=file_RT
    )
print(rtstruct.get_roi_names())

mask_name = 'kidney_left'
kidney_mask = dicom.get_aligned_rtstruct(
    file_RT = file_RT,
    file_NM = file_NM,
    dicom_series_path = path_CT,
    rt_struct_name = mask_name
)

uncertainty_abs, uncertainty_pct = recon_algorithm.compute_uncertainty(
    mask = kidney_mask,
    data_storage_callback = data_storage_callback,
    return_pct = True,
    include_additive_term=True # must have additive_term_variance_estimate in likelihood
)
print(f'Estimated uncertainty in {mask_name}: {uncertainty_pct:.2f}%')

# %% Multiple Bed Positions
files_NM = [
    os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'bed1_projections.dcm'),
    os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'bed2_projections.dcm'),
]
path_CT = os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'CT')
files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]

projectionss = dicom.load_multibed_projections(files_NM)


def initialize_reconstruction_algorithm_singlebed(i):
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
    # Return only the reconstruction algorithm initialization (not calling it for any iterations/subsets yet)
    return reconstruction_algorithm

recon_algo_upper = initialize_reconstruction_algorithm_singlebed(0)
recon_algo_lower  = initialize_reconstruction_algorithm_singlebed(1)
recon_algo = PGAAMultiBedSPECT(files_NM, [recon_algo_upper, recon_algo_lower])
# Initialize callback for each bed position
callbacks = [DataStorageCallback(r.likelihood, r.object_prediction) for r in recon_algo.reconstruction_algorithms]
reconstructed_image_multibed = recon_algo(4, 8, callback=callbacks)

mask_name = 'liver'
liver_mask = dicom.get_aligned_rtstruct(
    file_RT = file_RT,
    file_NM = files_NM[0],
    dicom_series_path = path_CT,
    rt_struct_name = mask_name,
    shape=reconstructed_image_multibed.shape
)

uncertainty_abs, uncertainty_pct = recon_algo.compute_uncertainty(liver_mask, callbacks, return_pct=True)

print(f'Estimated uncertainty in {mask_name}: {uncertainty_pct:.2f}%')
