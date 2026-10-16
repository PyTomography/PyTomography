"""Multiple bed positions

Reconstruct several bed positions and stitch them into one volume.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_dicommultibed.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
# %% DICOM Multiple Bed Positions
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Lu177-PSMA-GEDisc")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Lu177-PSMA-GEDisc")

import os
import torch
from pytomography import io
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood

# Paths are set in the data cell at the top of this tutorial
save_path = DATA / 'SPECT'

# %% Part 1: Opening Data
files_NM = [
    os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'bed1_projections.dcm'),
    os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'bed2_projections.dcm'),
]
path_CT = os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'CT')
files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]

_, _, proj_meta = io.SPECT.read_dicom(files_NM[0])
for i, window in enumerate(proj_meta.energy_windows):
    print(f'Index {i}:   Name: "{window.name}", Energies: {window.lower:g}-{window.upper:g} keV')

beds = io.SPECT.read_dicom_beds(files_NM)  # the inferior bed first
(projections_lower, object_meta_lower, proj_meta_lower), (projections_upper, object_meta_upper, proj_meta_upper) = beds

# %% Part 2: Reconstruction
def reconstruct_singlebed(projections, object_meta, proj_meta):
    # Change these depending on your file:
    index_peak = 1
    index_lower = 3
    index_upper = 2
    photopeak = projections[index_peak]
    scatter = io.SPECT.energy_window_scatter(projections, proj_meta, index_peak, index_lower, index_upper)
    # Build system matrix
    photon_energy = proj_meta.energy_windows[index_peak].photon_energy
    attenuation_map = io.attenuation_map(files_CT, object_meta, energy=photon_energy)
    psf_meta = io.SPECT.collimator_psf('GI-MEGP', photon_energy)
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
    print(reconstruction_algorithm.estimate_memory(n_subsets=8))
    return reconstruction_algorithm(n_iters=4, n_subsets=8)

recon_upper = reconstruct_singlebed(projections_upper, object_meta_upper, proj_meta_upper)
recon_lower = reconstruct_singlebed(projections_lower, object_meta_lower, proj_meta_lower)

recon_stitched, object_meta_stitched = io.SPECT.stitch_beds(
    recons=torch.stack([recon_upper, recon_lower]),
    object_metas=[object_meta_upper, object_meta_lower])

# %% Save the images
saved = io.save_dicom(recon_stitched, OUTPUT / 't_dicommultibed' / 'osem_stitched', object_meta_stitched,
                      reference=files_NM[1], series_description='OSEM 4x8, two beds stitched', overwrite=True)
