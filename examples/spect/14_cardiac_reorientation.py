"""Cardiac reorientation

Reconstruct myocardial perfusion data and reorient it to the short axis.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_CardiacReorientation.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
# %% Reconstruction
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Tc99m-Cardiac")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Tc99m-Cardiac")

import os
from pytomography import io
from pytomography.transforms.SPECT import SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood

# Paths are set in the data cell at the top of this tutorial
PATH = DATA / 'SPECT' / 'Tc99m-Cardiac'

file_NM = os.path.join(PATH, 'anonymized_file.dcm')
photopeak, object_meta, proj_meta = io.SPECT.read_dicom(file_NM, energy_window=0)

psf_meta = io.SPECT.collimator_psf('SY-LEHR', proj_meta.energy_windows[0].photon_keV)  # 140.5 keV, Tc-99m
psf_transform = SPECTPSFTransform(psf_meta)
system_matrix = SPECTSystemMatrix(
    obj2obj_transforms = [psf_transform],
    proj2proj_transforms = [],
    object_meta = object_meta,
    proj_meta = proj_meta)
likelihood = PoissonLogLikelihood(system_matrix, photopeak)
reconstruction_algorithm = OSEM(likelihood)
print(reconstruction_algorithm.estimate_memory(n_subsets=8))
recon = reconstruction_algorithm(n_iters=4, n_subsets=8)

# %% Save the images
saved = io.save_dicom(recon, OUTPUT / 't_CardiacReorientation' / 'osem', object_meta, reference=file_NM,
                      series_description='OSEM 4x8', overwrite=True)
