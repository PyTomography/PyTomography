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
datasets.fetch("PET/GATE-mMR-Brain")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("PET/GATE-mMR-Brain")

import torch
import torch.nn as nn
from torch.optim import LBFGS
import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.algorithms import OSEM, DIPRecon
from pytomography.io.PET import gate
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.shared import GaussianFilter
from pytomography.utils import sss
import matplotlib.pyplot as plt
import gc
import os
import numpy as np
import numpy as np
from monai.transforms import ScaleIntensityd, CropForeground, Compose, DivisiblePadd, SpatialCropd, ThresholdIntensityd

path = DATA / 'PET' / 'GATE-mMR-Brain'
# Macro path where PET scanner geometry file is defined
macro_path = os.path.join(path, 'mMR_Geometry.mac')
# Get information dictionary about the scanner
info = gate.get_detector_info(path = macro_path,
    mean_interaction_depth=9, min_rsector_difference=0)

path = DATA / 'PET' / 'GATE-mMR-Brain'
# Macro path where PET scanner geometry file is defined
macro_path = os.path.join(path, 'mMR_Geometry.mac')
# Get information dictionary about the scanner
info = gate.get_detector_info(path = macro_path,
    mean_interaction_depth=9, min_rsector_difference=0)
# Paths to all ROOT files containing data
paths = [os.path.join(path, f'all_physics/mMR_voxBrain_{i}.root') for i in range(1, 55)]; del(paths[23])  # file 24 is empty

speed_of_light = 0.3 #mm/ps
fwhm_tof_resolution = 550 * speed_of_light / 2 #ps to position along LOR
TOF_range = 1000 * speed_of_light #ps to position along LOR (full range)
num_tof_bins = 21
tof_meta = PETTOFMeta(num_tof_bins, TOF_range, fwhm_tof_resolution, n_sigmas=3)

# Cached in OUTPUT by the GATE list-mode TOF tutorial; compute any that is missing (this reads the ROOT files once)
if not os.path.exists(os.path.join(OUTPUT, 'normalization_weights.pt')):
    normalization_paths = [os.path.join(path, f'normalization_scan/mMR_Norm_{i}.root') for i in range(1,37)]
    normalization_weights = gate.get_normalization_weights_cylinder_calibration(
        normalization_paths,
        info,
        cylinder_radius = 318, # mm (radius of calibration cylindrical shell)
        include_randoms=False
    )
    torch.save(normalization_weights, os.path.join(OUTPUT, 'normalization_weights.pt'))
if not os.path.exists(os.path.join(OUTPUT, 'detector_ids_tof21bin_all_events.pt')):
    detector_ids = gate.get_detector_ids_from_root(paths, info, tof_meta=tof_meta)
    detector_ids = detector_ids[detector_ids[:,2]>-1] # For TOF, only take events within the TOF bins
    torch.save(detector_ids, os.path.join(OUTPUT, 'detector_ids_tof21bin_all_events.pt'))
if not os.path.exists(os.path.join(OUTPUT, 'detector_ids_delays.pt')):
    detector_ids_delays = gate.get_detector_ids_from_root(paths, info, substr = 'delay')
    torch.save(detector_ids_delays, os.path.join(OUTPUT, 'detector_ids_delays.pt'))
normalization_weights = torch.load(os.path.join(OUTPUT, 'normalization_weights.pt'))
detector_ids = torch.load(os.path.join(OUTPUT, 'detector_ids_tof21bin_all_events.pt'))
detector_ids_delays= torch.load(os.path.join(OUTPUT, 'detector_ids_delays.pt'))

sinogram_randoms_estimate = gate.listmode_to_sinogram(
    detector_ids_delays,
    info
)
sinogram_randoms_estimate = gate.smooth_randoms_sinogram(
    sinogram_randoms_estimate,
    info,
    sigma_r=4,
    sigma_theta=4,
    sigma_z=4
)
sinogram_randoms_estimate = gate.randoms_sinogram_to_sinogramTOF(
    sinogram_randoms_estimate,
    tof_meta = tof_meta,
    coincidence_timing_width = 4300
) # coinicidence timing window for this GATE simulation was set to 4300ps
lm_randoms_estimate = gate.sinogram_to_listmode(
    detector_ids,
    sinogram_randoms_estimate,
    info,
)

object_meta = ObjectMeta(
    dr=(1.25,1.25,1.25), #mm
    shape=(204,204,154) #voxels
)
# Get projection space metadata from PET geometry information dictionary
proj_meta = PETLMProjMeta(
    detector_ids,
    info,
    tof_meta=tof_meta,
    weights_sensitivity=normalization_weights
    )
atten_map = gate.get_attenuation_map_nifti(os.path.join(path, 'fdg_pet_phantom_umap.nii.gz'), object_meta).to(pytomography.dtype).to(pytomography.device)
normalization_weights = torch.load(os.path.join(OUTPUT, 'normalization_weights.pt'))
proj_meta = PETLMProjMeta(
    detector_ids,
    info,
    weights_sensitivity=normalization_weights,
    tof_meta=tof_meta
    )
psf_transform = GaussianFilter(4)
system_matrix = PETLMSystemMatrix(
       object_meta,
       proj_meta,
       obj2obj_transforms = [psf_transform],
       N_splits=10,
       attenuation_map=atten_map.to(pytomography.device),
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

# Scatter correction
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
lm_scatter_estimate = gate.sinogram_to_listmode(proj_meta.detector_ids, scatter_sinogram, proj_meta.info)
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
       attenuation_map=atten_map.to(pytomography.device),
)
additive_term = (lm_scatter_estimate + lm_randoms_estimate) / lm_norm
additive_term[additive_term.isnan()] = 0
likelihood = PoissonLogLikelihood(
        system_matrix,
        additive_term = additive_term
    )
recon_algorithm = OSEM(likelihood)
recon_lm_tof = recon_algorithm(50,1)

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
        self.max_iter = 20
        
    def fit(self, object):
        # This method trains the network for n_epochs at a learning rate of lr
        data = self.pipeline({'NM': object.unsqueeze(0), 'MR': self.anatomical_image.unsqueeze(0)})
        optimizer_lfbgs = LBFGS(self.network.parameters(), lr=self.lr, max_iter=self.max_iter, history_size=100)
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

import nibabel as nib
import numpy.linalg as npl 
from scipy.ndimage import affine_transform
def align_highres_image(path, img=None):
    # If img is none, extract data from path
    data = nib.load(path)
    # If img is none, extract data from path
    if img is None:
        img = data.get_fdata()
    Sx, Sy, Sz = -(np.array(img.shape)-1) / 2
    dx, dy, dz = data.header['pixdim'][1:4]
    # Convert from RAS to LPS space for DICOM
    dx*=-1; dy*=-1
    M_highres = np.zeros((4,4))
    M_highres[0] = np.array([dx, 0, 0, Sx*dx])
    M_highres[1] = np.array([0, dy, 0, Sy*dy])
    M_highres[2] = np.array([0, 0, dz, Sz*dz])
    M_highres[3] = np.array([0, 0, 0, 1])
    dx, dy, dz = object_meta.dr
    Sx, Sy, Sz = -(np.array(object_meta.shape)-1) / 2
    M_pet = np.zeros((4,4))
    M_pet[0] = np.array([dx, 0, 0, Sx*dx])
    M_pet[1] = np.array([0, dy, 0, Sy*dy])
    M_pet[2] = np.array([0, 0, dz, Sz*dz])
    M_pet[3] = np.array([0, 0, 0, 1])
    M = npl.inv(M_highres) @ M_pet
    return affine_transform(img, M, output_shape=object_meta.shape, mode='constant', order=1)
mri_aligned = torch.tensor(align_highres_image(os.path.join(path, 'fdg_pet_phantom_mri.nii.gz'))).to(pytomography.device).to(torch.float32)

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

dip_prior.n_epochs = 10
dip_prior.max_iter = 20
dip_prior.lr = 1

recon_algorithm = DIPRecon(
    likelihood = likelihood,
    prior_network=dip_prior,
    rho=5e5,
)

recon_DIP = recon_algorithm(n_iters=100, subit1=2)
