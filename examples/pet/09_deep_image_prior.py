"""Deep Image Prior

Build a PyTorch network and use it as the image model inside the reconstruction.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_PETGATE_DIP.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% GATE (Deep Image Prior)
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("PET/GATE-mMR-BrainInWater", extras=["mri"])  # with the MRI, the network's input
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("PET/GATE-mMR-BrainInWater")

import torch
import torch.nn as nn
from torch.optim import LBFGS
import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.algorithms import OSEM, DIPRecon
from pytomography import io
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.shared import GaussianFilter
from pytomography.utils import sss
import matplotlib.pyplot as plt
import gc
import numpy as np
from monai.transforms import ScaleIntensityd, CropForeground, Compose, DivisiblePadd, SpatialCropd, ThresholdIntensityd

# %% How much memory this needs
RAM_GB = 25 # memory (RAM) this tutorial may use, in GB; it needs about 22 GB however low this is
NETWORK_GB = 7 # training the network takes about 7 GB of GPU memory, which on Windows counts as RAM too
pytomography.set_memory_budget(RAM_GB - NETWORK_GB) # what the scatter estimate and the reconstructions may use

LOAD_FROM_ROOT = False # True: read the ROOT files again instead of the results cached in OUTPUT

path = DATA / 'PET' / 'GATE-mMR-BrainInWater'
macro = path / 'mMR_Geometry.mac'  # the scanner, as the simulation's GATE macro describes it
paths = [path / 'simple_phantom' / f'mMR_voxBrain_withSimplePhantom_{i}.root' for i in range(1, 55)]  # its 54 runs


def read_gate(events, tof_meta=None, normalization=None):
    """io.PET.read_gate for one kind of coincidence: (events, object_meta, proj_meta). Reading the 54 ROOT files takes
    a few minutes, so the first run keeps the result in OUTPUT and later runs load it from there."""
    name = f'gate_{events}' + ('' if tof_meta is None else f'_tof{tof_meta.num_bins}') + ('' if normalization is None else '_norm')
    cache = OUTPUT / f'{name}.pt'
    if LOAD_FROM_ROOT or not cache.exists():
        # doi_mm: the lines of response end 9 mm inside the crystals, the mean depth of interaction
        torch.save(io.PET.read_gate(paths, macro, events=events, tof_meta=tof_meta, normalization=normalization,
                                    doi_mm=9), cache)
    return torch.load(cache, weights_only=False)


speed_of_light = 0.3 #mm/ps
fwhm_tof_resolution = 550 * speed_of_light / 2 #ps to position along LOR
TOF_range = 1000 * speed_of_light #ps to position along LOR (full range)
num_tof_bins = 21
tof_meta = PETTOFMeta(num_tof_bins, TOF_range, fwhm_tof_resolution, n_sigmas=3)

# Every coincidence in its TOF bin, with the normalization weights of every crystal pair (read_gate sets those without
# calibration data to 0), and the delayed coincidences
detector_ids, object_meta, proj_meta = read_gate('prompts', tof_meta=tof_meta,
                                                 normalization=path / 'normalization_weights.npy')
detector_ids_delays, _, _ = read_gate('delayeds')
info = proj_meta.info

sinogram_randoms_estimate = io.PET.listmode_to_sinogram(
    detector_ids_delays,
    info
)
del detector_ids_delays # binned: the events are not needed again
sinogram_randoms_estimate = io.PET.smooth_randoms_sinogram(
    sinogram_randoms_estimate,
    info,
    sigma_r=4,
    sigma_theta=4,
    sigma_z=4
)
sinogram_randoms_estimate = io.PET.randoms_sinogram_to_sinogramTOF(
    sinogram_randoms_estimate,
    tof_meta = tof_meta,
    coincidence_timing_width = 4300
) # coinicidence timing window for this GATE simulation was set to 4300ps
lm_randoms_estimate = io.PET.sinogram_to_listmode(
    detector_ids,
    sinogram_randoms_estimate,
    info,
)

# The grid: 1.25 mm voxels, centred where read_gate placed the scanner in the patient
object_meta = object_meta.with_grid(dr=(1.25, 1.25, 1.25), shape=(204, 204, 154))  # mm, voxels
# The phantom's attenuation map (1/cm, in GATE's world), placed in the patient and put on the grid, in 1/mm
atten_map = io.attenuation_map(io.PET.gate_image(path / 'umap_mMR_brainSimplePhantom.hv', proj_meta, units='1/cm'),
                               object_meta)
# read_gate's proj_meta holds the events, the scanner's info, the TOF bins and the normalization weights
psf_transform = GaussianFilter(4)
system_matrix = PETLMSystemMatrix(
       object_meta,
       proj_meta,
       obj2obj_transforms = [psf_transform],
       N_splits=10,
       attenuation_map=atten_map,
)
lm_norm = system_matrix._compute_sensitivity_projection(all_ids=False)
additive_term = lm_randoms_estimate / lm_norm
additive_term[additive_term.isnan()] = 0 # remove NaN values
# Provide the random-only
likelihood = PoissonLogLikelihood(
        system_matrix,
        additive_term = additive_term
    )
recon_algorithm = OSEM(likelihood)
recon_without_scatter_estimation = recon_algorithm(50,1)

# Scatter correction (its peak RAM and GPU memory first, before it runs)
print(sss.estimate_sss_memory(object_meta, proj_meta, system_matrix, sinogram_random=sinogram_randoms_estimate,
                              tof_meta=tof_meta, sinogram_interring_stepsize=6, sinogram_intraring_stepsize=6,
                              num_dense_tof_bins=25))
scatter_sinogram = sss.get_sss_scatter_estimate(
        object_meta,
        proj_meta,
        recon_without_scatter_estimation,
        atten_map,
        system_matrix,
        sinogram_random=sinogram_randoms_estimate,
        tof_meta=tof_meta,
        num_dense_tof_bins=25,
        image_stepsize=6,
        sinogram_interring_stepsize=6,
        sinogram_intraring_stepsize=6,
        N_splits=1)
lm_scatter_estimate = io.PET.sinogram_to_listmode(proj_meta.detector_ids, scatter_sinogram, proj_meta.info)
# Save memory, these are not needed anymore
del(system_matrix)
del(likelihood)
del(scatter_sinogram)
del(sinogram_randoms_estimate)
gc.collect()

# System matrix with no PSF modeling
system_matrix = PETLMSystemMatrix(
       object_meta,
       proj_meta,
       obj2obj_transforms = [],
       N_splits=10,
       attenuation_map=atten_map,
)
additive_term = (lm_scatter_estimate + lm_randoms_estimate) / lm_norm
additive_term[additive_term.isnan()] = 0
likelihood = PoissonLogLikelihood(
        system_matrix,
        additive_term = additive_term
    )
recon_algorithm = OSEM(likelihood)
print(recon_algorithm.estimate_memory(n_subsets=1)) # its peak RAM and GPU memory, before it runs
recon_lm_tof = recon_algorithm(50,1)
# The events' host copy (the system matrix keeps them on the GPU), the per-event estimates (in additive_term now) and
# the normalization weights (used for the sensitivity image, computed above) are not needed again: freeing them
# leaves room for the network
del detector_ids, lm_randoms_estimate, lm_scatter_estimate, lm_norm
proj_meta.weights_sensitivity = None
gc.collect()

filter = GaussianFilter(3)
filter.configure(object_meta, proj_meta)
recon_lm_tof_filtered = filter(recon_lm_tof)

def get_downward_block(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(), 
        ) 
    
def get_downsample_block(out_channels):
    return nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
        ) 
    
def get_bottleneck_block(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
        ) 
    
def get_bilinear_upsample_block(in_channels, out_channels):
    return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding='same'),
        )
    
def get_upward_block(in_channels):
    return nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding='same'),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding='same'),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(),
        )
    
def get_final_block(in_channels):
    return nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding='same'),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels, 1, kernel_size=3, padding='same'),
        )
    
class UNetCustom(nn.Module):
    def __init__(self, n_channels=[4, 8, 16, 32, 64]):
        super().__init__()       
        self.downward_block1 = get_downward_block(1, n_channels[0])
        self.downward_block2 = get_downward_block(n_channels[0], n_channels[1])
        self.downward_block3 = get_downward_block(n_channels[1], n_channels[2])
        self.downward_block4 = get_downward_block(n_channels[2], n_channels[3])
        self.downsample_block1 = get_downsample_block(n_channels[0])
        self.downsample_block2 = get_downsample_block(n_channels[1])
        self.downsample_block3 = get_downsample_block(n_channels[2])
        self.downsample_block4 = get_downsample_block(n_channels[3])
        self.bottleneck_block = get_bottleneck_block(n_channels[3], n_channels[4])
        self.upsample_block1 = get_bilinear_upsample_block(n_channels[4], n_channels[3])
        self.upsample_block2 = get_bilinear_upsample_block(n_channels[3], n_channels[2])
        self.upsample_block3 = get_bilinear_upsample_block(n_channels[2], n_channels[1])
        self.upsample_block4 = get_bilinear_upsample_block(n_channels[1], n_channels[0])
        self.upward_block1 = get_upward_block(n_channels[3])
        self.upward_block2 = get_upward_block(n_channels[2])
        self.upward_block3 = get_upward_block(n_channels[1])
        self.final_block = get_final_block(n_channels[0])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.downward_block1(x)
        x = self.downsample_block1(x1)
        x2 = self.downward_block2(x)
        x = self.downsample_block2(x2)
        x3 = self.downward_block3(x)
        x = self.downsample_block3(x3)
        x4 = self.downward_block4(x)
        x = self.downsample_block4(x4)
        x = self.bottleneck_block(x)
        x = self.upsample_block1(x) + x4
        x = self.upward_block1(x)
        x = self.upsample_block2(x) + x3
        x = self.upward_block2(x)
        x = self.upsample_block3(x) + x2
        x = self.upward_block3(x)
        x = self.upsample_block4(x) + x1
        x = self.final_block(x)
        return x

test_input = torch.ones((1,1,128,128,128))
test_output = UNetCustom()(test_input)
print(test_input.shape)
print(test_output.shape)

class DIPPrior():
    def __init__(
        self,
        network,
        anatomical_image,
        pipeline, # pipeline for preprocessing MRI image
        scale_factor=1, # constant to scale MRI image by
        n_epochs=10, # how many epochs the network trains for when fitting
        lr = 0.1, # learning rate when fitting
    ):
        self.network = network
        self.anatomical_image = anatomical_image
        self.pipeline = pipeline
        self.n_epochs = n_epochs
        self.scale_factor = scale_factor
        self.lr = lr
        self.max_iter = 20 # L-BFGS iterations per epoch
        self.line_search_fn = None # L-BFGS line search: None (fixed step lr) or 'strong_wolfe'
        
    def fit(self, object):
        # This method trains the network for n_epochs at a learning rate of lr
        data = self.pipeline({'NM': object.unsqueeze(0), 'MR': self.anatomical_image.unsqueeze(0)})
        # history_size=10 as in the paper (100 kept 200 copies of the network's parameters on the GPU)
        optimizer_lfbgs = LBFGS(self.network.parameters(), lr=self.lr, max_iter=self.max_iter, history_size=10, line_search_fn=self.line_search_fn)
        NM_truth = data['NM'].unsqueeze(0) * self.scale_factor
        network_input = data['MR'].unsqueeze(0)
        criterion = torch.nn.MSELoss()
        def closure(optimizer):
            optimizer.zero_grad()
            NM_prediction = self.network(network_input)
            loss = criterion(NM_prediction, NM_truth)
            loss.backward()
            return loss
        for epoch in range(self.n_epochs):  
            loss = optimizer_lfbgs.step(lambda: closure(optimizer_lfbgs))
        self.network.zero_grad(set_to_none=True)
        with torch.no_grad():
            # Add batch/channel dimension
            network_prediction = self.network(data['MR'].unsqueeze(0)).squeeze()
        self.prior_object = self.pipeline.inverse({'NM': network_prediction.unsqueeze(0)})['NM'].as_tensor().squeeze() / self.scale_factor
        
    def predict(self):        
        return self.prior_object.detach()

# The phantom's MR (NIfTI, 0.4 mm voxels) on the PET grid: load_image reads it with its own place in the patient, and
# .on() averages it over each 1.25 mm voxel
mri_aligned = io.load_image(path / 'fdg_pet_phantom_mri.nii.gz').on(object_meta)

mri_crop_above = 250
mri_crop_below = 120

roi_start, roi_end = CropForeground().compute_bounding_box(mri_aligned.unsqueeze(0))
pipeline = Compose([
    SpatialCropd(['MR', 'NM'], roi_start=roi_start, roi_end=roi_end, allow_missing_keys=True),
    DivisiblePadd(['MR', 'NM'], 16, allow_missing_keys=True),
    ThresholdIntensityd(['MR'], mri_crop_above, above=False, cval=mri_crop_above),
    ThresholdIntensityd(['MR'], mri_crop_below, above=True, cval=mri_crop_below),
    ScaleIntensityd(['MR'], 0, 1)
])

start_channels = 12
torch.manual_seed(0) # the network's starting weights: the same network, and the same images, on every run
net = UNetCustom([start_channels,2*start_channels,4*start_channels,8*start_channels,16*start_channels]).to(pytomography.device)

dip_prior = DIPPrior(
    net,
    mri_aligned,
    pipeline,
    n_epochs= 100,
    scale_factor= 50,
    lr=0.01
    )

dip_prior.fit(recon_lm_tof)
# the pretrained network's image, the starting point shown on the contrast-noise plot at the end
pretrained_prediction = dip_prior.prior_object.clone()

dip_prior.n_epochs = 1
dip_prior.max_iter = 10
dip_prior.lr = 1
dip_prior.line_search_fn = 'strong_wolfe'

from scipy.ndimage import binary_erosion
from pytomography.callbacks import Callback
# grey and white matter, from the MR intensity classes, one voxel in from their edges
mr = mri_aligned.cpu().numpy()
grey = torch.tensor(binary_erosion((mr >= 180) & (mr < 195), np.ones((3, 3, 3)))).to(pytomography.device)
white = torch.tensor(binary_erosion((mr >= 135) & (mr < 150), np.ones((3, 3, 3)))).to(pytomography.device)

def contrast_noise(image):
    """Mean grey matter over mean white matter (contrast), and the standard deviation over the mean in white matter (noise)."""
    w = image[white]
    return (image[grey].mean() / w.mean()).item(), (w.std() / w.mean()).item()

class ContrastNoise(Callback):
    """Records the contrast and noise of the image after every iteration."""
    def __init__(self):
        self.contrast, self.noise = [], []
    def run(self, object, n_iter, n_subset=None):
        c, n = contrast_noise(object)
        self.contrast.append(c)
        self.noise.append(n)
        return object

rho = 240
recon_algorithm = DIPRecon(
    likelihood = likelihood,
    prior_network=dip_prior,
    rho=rho,
)

curve_DIP = ContrastNoise()
recon_DIP = recon_algorithm(n_iters=100, subit1=2, callback=curve_DIP)

for name, image in (('OSEM', recon_lm_tof), ('OSEM, 3 mm filter', recon_lm_tof_filtered), ('Deep Image Prior', recon_DIP)):
    c, n = contrast_noise(image)
    print(f'{name:18s} contrast {c:.2f}, noise {n:.2f}')

# %% Contrast against noise
# post-filtered OSEM
osem_curve = []
for fwhm in (0, 2, 3, 4, 5, 6):
    image = recon_lm_tof
    if fwhm:
        gauss = GaussianFilter(fwhm)
        gauss.configure(object_meta, proj_meta)
        image = gauss(recon_lm_tof)
    osem_curve.append(contrast_noise(image))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([n for c, n in osem_curve], [c for c, n in osem_curve], 'k--o', ms=4, label='OSEM 50x1, Gaussian filter 0-6 mm')
c, n = contrast_noise(pretrained_prediction)
ax.plot(n, c, 'ks', label='pretrained network')
line, = ax.plot(curve_DIP.noise, curve_DIP.contrast, '.-', ms=3, lw=1, label=f'Deep Image Prior, rho = {rho:g}, 100 iterations')
ax.plot(curve_DIP.noise[-1], curve_DIP.contrast[-1], 'o', color=line.get_color(), ms=6)
ax.set_xlabel('noise in white matter (standard deviation / mean)')
ax.set_ylabel('grey / white matter contrast')
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# %% Save the images
for name, image in [('deep_image_prior', recon_DIP), ('osem', recon_lm_tof)]:
    io.save_dicom(image, OUTPUT / 't_PETGATE_DIP' / name, object_meta, series_description=name, overwrite=True)
