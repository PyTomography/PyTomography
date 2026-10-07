"""Cardiac reorientation

Reconstruct myocardial perfusion data and reorient it to the short axis.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_CardiacReorientation.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% Reconstruction
import os
from pathlib import Path

# Tutorial data: the folder set by the PYTOMOGRAPHY_DATA environment variable (see Tutorial data in the docs)
DATA = Path(os.environ.get("PYTOMOGRAPHY_DATA", "~/pytomography_data")).expanduser()
# Results go here, never into the data folder
OUTPUT = Path(os.environ.get("PYTOMOGRAPHY_OUTPUT", "pytomography_outputs")).expanduser() / "SPECT/Tc99m-Cardiac"
OUTPUT.mkdir(parents=True, exist_ok=True)

import os
from pytomography.io.SPECT import dicom
from pytomography.transforms.SPECT import SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
import pytomography
import matplotlib.pyplot as plt
import torch

# Paths are set in the data cell at the top of this tutorial
PATH = DATA / 'SPECT' / 'Tc99m-Cardiac'

file_NM = os.path.join(PATH, 'anonymized_file.dcm')
object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=0)
photopeak = dicom.get_projections(file_NM, index_peak=0)

psf_meta = dicom.get_psfmeta_from_scanner_params('SY-LEHR', energy_keV=140.5)
psf_transform = SPECTPSFTransform(psf_meta)
system_matrix = SPECTSystemMatrix(
    obj2obj_transforms = [psf_transform],
    proj2proj_transforms = [],
    object_meta = object_meta,
    proj_meta = proj_meta)
likelihood = PoissonLogLikelihood(system_matrix, photopeak)
reconstruction_algorithm = OSEM(likelihood)
recon = reconstruction_algorithm(n_iters=4, n_subsets=8)

# # Modify the path below to a location on your computer where you want to save the data
# # # Code only works if folder doesnt exist, so delete it if present
# if os.path.exists(save_path) and os.path.isdir(save_path):
#     shutil.rmtree(save_path)
# # Save
# dicom.save_dcm(
#     save_path = save_path,
#     object = recon,
#     file_NM = file_NM,
#     recon_name = 'OSEM_4it_8ss')

# %% Reorientation
from pytomography.utils.cardiac_spect import get_mask, get_shift_values, get_angle, shift_object, rotate_object, plot_arrow, create_circular_mask, masking

# Define the object and get the mask
testObject = recon[:,:,30:45]
testObjectMask = get_mask(testObject, 0.55, 0.9)

# Finding the shift values
ShiftedAxial, ty, tx = get_shift_values(testObjectMask[:,:,7])
ShiftedSagittal, _, tz = get_shift_values(testObjectMask[40,:,:])

# Finding the angles values
AxialAngle = get_angle(
    ShiftedAxial,
    torch.linspace(240, 270, 100).to(pytomography.device)
)
SagittalAngle = get_angle(
    ShiftedSagittal,
    torch.linspace(335, 355, 100).to(pytomography.device)
)

# Shifting the object
ShiftedObject = shift_object(testObject, -tx, -ty, -tz)

# Rotating the object
RotatedObject = rotate_object(ShiftedObject, AxialAngle, SagittalAngle)

# Make sure the object does not contain negatinve values
RotatedObject = torch.clamp(RotatedObject, min=0)

# Permute the object to the correct orientation
RotatedObject = RotatedObject.permute(2, 1, 0)

# Perform masking on the RotatedObject
maskedObject = masking(RotatedObject, radius=10)

plot_arrow(ShiftedAxial, 235)

def plot_views(tensor: torch.Tensor):
    """
    Plots three different views of the given 3D tensor

    Args:
        tensor (torch.Tensor): 3D tensor with shape [depth, height, width].
    """
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Vertical Long Axis (VLA)
    vla_image = torch.flip(tensor[7, :, :].cpu(), dims=[0]).T
    axs[0].imshow(vla_image, interpolation='gaussian', cmap='jet')

    # Horizontal Long Axis (HLA)
    hla_image = torch.flip(torch.flip(tensor[:, 32, :].cpu(), dims=[0]), dims=[1])
    axs[1].imshow(hla_image, interpolation='gaussian', cmap='jet')

    # Short Axis (SA)
    sa_image = torch.flip(tensor[:, :, 32].cpu(), dims=[0])
    axs[2].imshow(sa_image, interpolation='gaussian', cmap='jet')

    # Adjust layout and show the plot
    plt.tight_layout()
    
plot_views(RotatedObject)
plot_views(maskedObject)
