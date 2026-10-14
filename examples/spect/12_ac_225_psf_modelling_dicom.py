"""Ac-225 PSF modelling (DICOM)

The Ac-225 PSF model applied to measured data.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_ac225_dicom_recon.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% Ac-225 Advanced PSF Modeling (DICOM)
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Ac225-NEMA-SymT2")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Ac225-NEMA-SymT2")

import matplotlib.pyplot as plt
import torch # needed for kernels
import pytomography
from pytomography import io
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.transforms.shared import GaussianFilter
import json
import os
from spectpsftoolbox.kernel1d import ArbitraryKernel1D, FunctionKernel1D
from spectpsftoolbox.operator2d import GaussianOperator, Rotate1DConvOperator, RotateSeperable2DConvOperator
from pytomography.utils import plot_utils

path = DATA / 'SPECT'

pathCT = os.path.join(path, 'Ac225-NEMA-SymT2', 'CT')
files_CT = [os.path.join(pathCT, file) for file in os.listdir(pathCT)]
file_NM = os.path.join(path, 'Ac225-NEMA-SymT2', 'projection_data.IMA')

with open(os.path.join(path, 'Ac225-NEMA-SymT2', 'ac225_psf_model.json')) as f:
    fit = {k: torch.tensor(v) if isinstance(v, list) else v for k, v in json.load(f).items()}
two_exponentials = lambda a, b: b[0] * torch.exp(-a * b[1]) + b[2] * torch.exp(-a * b[3])
width = lambda a, b: b[0] + b[1] * (torch.sqrt(a**2 + b[2]**2) - torch.abs(b[2]))
tail_width = lambda a, b: 1 + b[0] * (torch.sqrt((a - fit['tail_sigma_a_min'])**2 + b[1]**2) - torch.abs(b[1]))
gaussian = GaussianOperator(two_exponentials, width, fit['gaussian_amplitude'], fit['gaussian_sigma'])
tails = Rotate1DConvOperator(
    ArbitraryKernel1D(fit['tail_kernel'], two_exponentials, tail_width, fit['tail_amplitude'], fit['tail_sigma'],
                      fit['tail_kernel_spacing'], grid_sample_mode='bicubic'),
    N_angles=3, additive=True, rot=90)
background = RotateSeperable2DConvOperator(
    FunctionKernel1D(lambda x: torch.exp(-torch.abs(x)), two_exponentials, width, fit['background_amplitude'],
                     fit['background_sigma'], a_min=fit['background_a_min'], a_max=fit['background_a_max']),
    N_angles=1, additive=False)
psf_operator = (tails + background) * gaussian + gaussian
psf_operator.set_device(pytomography.device)

class OperatorInCm:
    """The fitted model takes lengths in cm, SPECTPSFToolbox's unit when it was fitted; PyTomography's are mm. This
    passes the PSF transform's coordinates and distances to the model in cm (the kernel is normalised, so nothing else
    changes)."""
    def __init__(self, operator):
        self.operator = operator
    def __call__(self, input, xv, yv, distances, normalize=False):
        return self.operator(input, xv / 10, yv / 10, distances / 10, normalize=normalize)

index_peak = 3
index_lower = 4
index_upper = 5
projections, object_meta, proj_meta = io.SPECT.read_dicom(file_NM)
attenuation_map = io.attenuation_map(files_CT, object_meta, energy_keV=proj_meta.energy_windows[index_peak].photon_keV)

photopeak = projections[index_peak]
scatter = io.SPECT.energy_window_scatter(projections, proj_meta, peak=index_peak, lower=index_lower, upper=index_upper,
                                         sigma_theta=2, sigma_r_mm=4.8, sigma_z_mm=4.8)

att_transform = SPECTAttenuationTransform(attenuation_map=attenuation_map)
psf_transform = SPECTPSFTransform(psf_operator=OperatorInCm(psf_operator))
system_matrix = SPECTSystemMatrix(
        obj2obj_transforms = [att_transform,psf_transform],
        proj2proj_transforms = [],
        object_meta = object_meta,
        proj_meta = proj_meta)

likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter)
algorithm = OSEM(likelihood)
print(algorithm.estimate_memory(n_subsets=3))

recon = algorithm(n_iters=100, n_subsets=3)

filter = GaussianFilter(20) # FWHM in mm
filter.configure(object_meta, proj_meta)
recon_smoothed = filter(recon)

ct = io.load_image(files_CT)
SPECT_imshow_kwargs = {
    'cmap': plot_utils.pet_cmap,
    'interpolation': 'Gaussian',
    'alpha': 0.6,
    'vmin': 0,
    'vmax': 0.1,
    'zorder': 1, # this will ensure SPECT is on top of CT
    'origin': 'lower'
}
CT_imshow_kwargs = {
    'cmap': 'Greys_r',
    'interpolation': 'Gaussian',
    'vmin': -25, # HU
    'vmax': 300, # HU
    'zorder': 0,
    'origin': 'lower'
}

plt.figure(figsize=(5,5), facecolor='k')
plt.title('440keV Peak: OSEM100it3ss', color='white')
plot_utils.dual_imshow_axial(
    im1 = recon_smoothed,
    im2 = ct.data,
    im1_idx = 70,
    affine1 = object_meta.affine,
    affine2 = ct.affine,
    imshow1_kwargs=SPECT_imshow_kwargs,
    imshow2_kwargs=CT_imshow_kwargs,
)
plt.axis('off')
plt.xlim(-120,130)
plt.ylim(-275,-25)
plt.tight_layout()

# %% Save the images
for name, image in [('osem', recon), ('osem_smoothed', recon_smoothed)]:
    io.save_dicom(image, OUTPUT / 't_ac225_dicom_recon' / name, object_meta, reference=file_NM,
                  series_description=f'OSEM 100x3, {name}', overwrite=True)
