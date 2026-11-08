"""Plotting utilities

Fused SPECT/CT views and other figure helpers.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_plotting.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
# %% Plotting Functionality
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Lu177-PSMA-GEDisc")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Lu177-PSMA-GEDisc")

import os
from pytomography import io

save_path = DATA / 'SPECT'

path_CT = os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'CT')
files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
path_SPECT = os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'pytomo_recon')
file_SPECT = os.path.join(path_SPECT, os.listdir(path_SPECT)[0])

spect = io.load_image(file_SPECT)
ct = io.load_image(files_CT)
print(spect)
print(ct)

# %% Save the images
saved = io.save_dicom(spect, OUTPUT / 't_plotting' / 'spect', reference=file_SPECT,
                      series_description='Stitched OSEM', overwrite=True)
