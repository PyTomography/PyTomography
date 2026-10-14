"""Uncertainty estimation

Voxel and region uncertainty that comes out of the reconstruction itself.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_uncertainty_spect.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Lu177-PSMA-GEDisc")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Lu177-PSMA-GEDisc")

import os
from pytomography import io
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM, PGAAMultiBedSPECT
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
import torch
from rt_utils import RTStructBuilder
from pytomography.callbacks import DataStorageCallback

save_path = DATA / 'SPECT'

file_NM = os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'bed2_projections.dcm')
path_CT = os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'CT')
files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
projections, object_meta, proj_meta = io.SPECT.read_dicom(file_NM)
photopeak = projections[1]
scatter, scatter_variance_estimate = io.SPECT.energy_window_scatter(projections, proj_meta, peak=1, lower=3, upper=2,
                                                                    return_scatter_variance_estimate=True)
# Build system matrix
photon_keV = proj_meta.energy_windows[1].photon_keV  # the 208 keV window
attenuation_map = io.attenuation_map(files_CT, object_meta, energy_keV=photon_keV)
psf_meta = io.SPECT.collimator_psf('GI-MEGP', photon_keV)
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

print(recon_algorithm.estimate_memory(n_subsets=8))
recon_OSEM = recon_algorithm(n_iters = 4, n_subsets = 8, callback=data_storage_callback)

file_RT = os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'segmentations.dcm')
rtstruct = RTStructBuilder.create_from(
        dicom_series_path=path_CT, 
        rt_struct_path=file_RT
    )
print(rtstruct.get_roi_names())

mask_name = 'kidney_left'
kidney_mask = io.load_mask(file_RT, reference=path_CT, rois=[mask_name]).on(object_meta, label=mask_name, threshold=0.5)

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

beds = io.SPECT.read_dicom_beds(files_NM)  # (projections, object_meta, proj_meta) of each bed, the inferior bed first

def initialize_reconstruction_algorithm_singlebed(projections, object_meta, proj_meta):
    # Change these depending on your file:
    index_peak = 1
    index_lower = 3
    index_upper = 2
    photopeak = projections[index_peak]
    scatter = io.SPECT.energy_window_scatter(projections, proj_meta, index_peak, index_lower, index_upper)
    # Build system matrix
    photon_keV = proj_meta.energy_windows[index_peak].photon_keV
    attenuation_map = io.attenuation_map(files_CT, object_meta, energy_keV=photon_keV)
    psf_meta = io.SPECT.collimator_psf('GI-MEGP', photon_keV)
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

recon_algo = PGAAMultiBedSPECT([initialize_reconstruction_algorithm_singlebed(*bed) for bed in beds])
# Initialize callback for each bed position
callbacks = [DataStorageCallback(r.likelihood, r.object_prediction) for r in recon_algo.reconstruction_algorithms]
print(recon_algo.estimate_memory(n_subsets=8))
reconstructed_image_multibed = recon_algo(4, 8, callback=callbacks)
object_meta_stitched = recon_algo.object_meta  # the grid of the stitched image

mask_name = 'liver'
liver_mask = io.load_mask(file_RT, reference=path_CT, rois=[mask_name]).on(object_meta_stitched, label=mask_name, threshold=0.5)

uncertainty_abs, uncertainty_pct = recon_algo.compute_uncertainty(liver_mask, callbacks, return_pct=True)

print(f'Estimated uncertainty in {mask_name}: {uncertainty_pct:.2f}%')

# %% Save the images
for name, image, meta in [('osem_bed2', recon_OSEM, object_meta), ('osem_stitched', reconstructed_image_multibed, object_meta_stitched)]:
    io.save_dicom(image, OUTPUT / 't_uncertainty_spect' / name, meta, reference=file_NM,
                  series_description=name, overwrite=True)
