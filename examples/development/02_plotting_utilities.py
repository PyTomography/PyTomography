"""Plotting utilities

Fused SPECT/CT views and other figure helpers.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_plotting.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% Plotting Functionality
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Lu177-PSMA-GEDisc")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Lu177-PSMA-GEDisc")

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pytomography.io.SPECT import dicom
from pytomography.utils import plot_utils

save_path = DATA / 'SPECT'

path_CT = os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'CT')
files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
path_SPECT = os.path.join(save_path, 'Lu177-PSMA-GEDisc', 'pytomo_recon')
file_SPECT = os.path.join(path_SPECT, os.listdir(path_SPECT)[0])

affine_SPECT = dicom._get_affine_single_file(file_SPECT)
affine_CT = dicom._get_affine_multifile(files_CT)
SPECT = dicom.open_singlefile(file_SPECT)
CT = dicom.open_multifile(files_CT)

SPECT_imshow_kwargs = {
    'cmap': plot_utils.pet_cmap,
    'interpolation': 'Gaussian',
    'alpha': 0.6,
    'vmin': 0,
    'vmax': 350,
    'zorder': 1, # this will ensure SPECT is on top of CT
    'origin': 'lower'
}
CT_imshow_kwargs = {
    'cmap': 'Greys_r',
    'interpolation': 'Gaussian',
    'vmin': -150, # HU
    'vmax': 475, # HU
    'zorder': 0,
    'origin': 'lower'
}

fig, ax = plt.subplots(1,3,figsize=(9,4.5), gridspec_kw={'wspace':0.0}, facecolor='k')
plt.sca(ax[0])
plot_utils.dual_imshow_coronal(
    im1 = SPECT,
    im2 = CT,
    im1_idx = 80,
    affine1 = affine_SPECT,
    affine2 = affine_CT,
    imshow1_kwargs=SPECT_imshow_kwargs,
    imshow2_kwargs=CT_imshow_kwargs,
)
plt.xlim(-200,220)
plt.ylim(-350,370)
plt.axis('off')
plt.text(0.03, 0.97, 'Coronal', color='white', fontsize=15, fontweight='bold', transform=plt.gca().transAxes, ha='left', va='top')

plt.sca(ax[1])
plot_utils.dual_imshow_sagittal(
    im1 = SPECT,
    im2 = CT,
    im1_idx = 50,
    affine1 = affine_SPECT,
    affine2 = affine_CT,
    imshow1_kwargs=SPECT_imshow_kwargs,
    imshow2_kwargs=CT_imshow_kwargs,
)
plt.xlim(-200,200)
plt.ylim(-350,370)
plt.axis('off')
plt.text(0.03, 0.97, 'Sagittal', color='white', fontsize=15, fontweight='bold', transform=plt.gca().transAxes, ha='left', va='top')

plt.sca(ax[2])
plot_utils.dual_imshow_axial(
    im1 = SPECT,
    im2 = CT,
    im1_idx = 117,
    affine1 = affine_SPECT,
    affine2 = affine_CT,
    imshow1_kwargs=SPECT_imshow_kwargs,
    imshow2_kwargs=CT_imshow_kwargs,
)
plt.xlim(-200,220)
plt.ylim(200,-100)
plt.axis('off')
plt.text(0.03, 0.97, 'Axial', color='white', fontsize=15, fontweight='bold', transform=plt.gca().transAxes, ha='left', va='top')

def cmap_f(x,a):
    return (1-np.exp(-x/a))/(1+np.exp(-x/a))
x = np.linspace(0,1,256)
colors = plot_utils.pet_cmap(x)
colors[:, -1] = 0.8*cmap_f(x, a=3e-2) # this value works well
new_pet_cmap = mcolors.LinearSegmentedColormap.from_list('NewColormap', colors)

SPECT_imshow_kwargs = {
    'cmap': new_pet_cmap,
    'interpolation': 'Gaussian',
    'alpha': 0.8, # Change the alpha higher now for more visibilty
    'vmin': 0,
    'vmax': 350,
    'zorder': 1,
    'origin': 'lower'
}
CT_imshow_kwargs = {
    'cmap': 'Greys_r',
    'interpolation': 'Gaussian',
    'vmin': -150, # HU
    'vmax': 475, # HU
    'zorder': 0,
    'origin': 'lower'
}

fig, ax = plt.subplots(1,3,figsize=(9,4.5), gridspec_kw={'wspace':0.0}, facecolor='k')
plt.sca(ax[0])
plot_utils.dual_imshow_coronal(
    im1 = SPECT,
    im2 = CT,
    im1_idx = 80,
    affine1 = affine_SPECT,
    affine2 = affine_CT,
    imshow1_kwargs=SPECT_imshow_kwargs,
    imshow2_kwargs=CT_imshow_kwargs,
)
plt.xlim(-200,220)
plt.ylim(-350,370)
plt.axis('off')
plt.text(0.03, 0.97, 'Coronal', color='white', fontsize=15, fontweight='bold', transform=plt.gca().transAxes, ha='left', va='top')

plt.sca(ax[1])
plot_utils.dual_imshow_sagittal(
    im1 = SPECT,
    im2 = CT,
    im1_idx = 50,
    affine1 = affine_SPECT,
    affine2 = affine_CT,
    imshow1_kwargs=SPECT_imshow_kwargs,
    imshow2_kwargs=CT_imshow_kwargs,
)
plt.xlim(-200,200)
plt.ylim(-350,370)
plt.axis('off')
plt.text(0.03, 0.97, 'Sagittal', color='white', fontsize=15, fontweight='bold', transform=plt.gca().transAxes, ha='left', va='top')

plt.sca(ax[2])
plot_utils.dual_imshow_axial(
    im1 = SPECT,
    im2 = CT,
    im1_idx = 117,
    affine1 = affine_SPECT,
    affine2 = affine_CT,
    imshow1_kwargs=SPECT_imshow_kwargs,
    imshow2_kwargs=CT_imshow_kwargs,
)
plt.xlim(-200,220)
plt.ylim(200,-100)
plt.axis('off')
plt.text(0.03, 0.97, 'Axial', color='white', fontsize=15, fontweight='bold', transform=plt.gca().transAxes, ha='left', va='top')
