"""Accuracy against known truth

Measure bias and noise in each sphere of a simulated phantom, with OSEM and BSREM.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_accuracy_known_truth.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/SIMIND-Jaszak")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/SIMIND-Jaszak")

import re
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytomography
from pytomography.io.SPECT import simind
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM, BSREM
from pytomography.priors import RelativeDifferencePrior
from pytomography.callbacks import Callback

# %% 1. A noisy acquisition
path = DATA / 'SPECT' / 'SIMIND-Jaszak' / 'lu177_SYME_jaszak'
windows = {'photopeak': path / 'tot_w4.h00', 'lower': path / 'tot_w5.h00', 'upper': path / 'tot_w6.h00'}
object_meta, proj_meta = simind.get_metadata(str(windows['photopeak']))
activity = 1000        # MBq in the whole phantom
acquisition_time = 15  # seconds per projection

torch.manual_seed(0)
projections = {w: torch.poisson(simind.get_projections(str(p)) * activity * acquisition_time)
               for w, p in windows.items()}
photopeak = projections['photopeak']
widths = {w: simind.get_energy_window_width(str(p)) for w, p in windows.items()}
scatter = simind.compute_EW_scatter(projections['lower'], projections['upper'],
                                    widths['lower'], widths['upper'], widths['photopeak'])
print(f"{photopeak.sum().item():.3g} counts in the photopeak window")

# %% 2. The truth: sphere masks and concentrations
block = (path / 'results.res').read_text(errors='replace').split('MULTIPLE SOURCE CONFIGURATION')[1].split('-----')[0]
volume_mL = {('background' if name == 'BKG' else f'sphere {name}'): float(v)
             for name, v in re.findall(r'^\s+(\d+|BKG)\s+([\d.]+)', block, re.M)}
sphere_MBq_per_mL = activity / (sum(v for r, v in volume_mL.items() if r != 'background') + 0.1 * volume_mL['background'])
truth_MBq_per_mL = {r: sphere_MBq_per_mL * (0.1 if r == 'background' else 1) for r in volume_mL}

labels = torch.from_numpy(np.load(path.parent / 'jaszak_spheres.npz')['labels'])  # 0 outside the spheres
masks = {}
for k in range(1, 7):
    fraction = F.avg_pool3d((labels == k).float()[None, None], kernel_size=4)[0, 0]
    masks[f'sphere {k}'] = (fraction > 0.5).to(pytomography.device)

attenuation_map = simind.get_attenuation_map(str(path / 'amap.hct'))
water = attenuation_map > 0.5 * attenuation_map.max()
dilate = lambda mask: F.max_pool3d(mask.float()[None, None], 7, stride=1, padding=3)[0, 0] > 0
erode = lambda mask: ~dilate(~mask)
spheres = torch.stack(list(masks.values())).any(0)
masks['background'] = erode(water) & ~dilate(spheres)

regions = list(masks)
for r in regions:
    print(f"{r:10s} {volume_mL[r]:8.2f} mL  {int(masks[r].sum()):6d} voxels  truth {truth_MBq_per_mL[r]:.3f} MBq/mL")

# %% 3. System matrix and likelihood
psf_meta = simind.get_psfmeta_from_header(str(windows['photopeak']))
system_matrix = SPECTSystemMatrix(
    obj2obj_transforms=[SPECTAttenuationTransform(attenuation_map), SPECTPSFTransform(psf_meta)],
    proj2proj_transforms=[],
    object_meta=object_meta,
    proj_meta=proj_meta)
likelihood = PoissonLogLikelihood(system_matrix, photopeak, additive_term=scatter)

# %% 4. Region statistics during reconstruction
class RegionStatistics(Callback):
    def __init__(self, masks, n_subsets):
        self.masks = masks
        self.n_subsets = n_subsets
        self.mean = {region: [] for region in masks}
        self.std = {region: [] for region in masks}

    def run(self, object, n_iter, n_subset):
        if n_subset == self.n_subsets - 1:  # last subset: the end of an iteration
            for region, mask in self.masks.items():
                values = object[mask]
                self.mean[region].append(values.mean().item())
                self.std[region].append(values.std().item())
        return object

# %% 5. Reconstruct with OSEM and BSREM
n_iters, n_subsets = 40, 8

t0 = time.time()
stats_osem = RegionStatistics(masks, n_subsets)
recon_osem = OSEM(likelihood)(n_iters=n_iters, n_subsets=n_subsets, callback=stats_osem)
print(f"OSEM:  {time.time() - t0:.1f} s")

t0 = time.time()
stats_bsrem = RegionStatistics(masks, n_subsets)
bsrem = BSREM(likelihood, prior=RelativeDifferencePrior(beta=0.3, gamma=2), relaxation_sequence=lambda n: 1 / (n / 50 + 1))
recon_bsrem = bsrem(n_iters=n_iters, n_subsets=n_subsets, callback=stats_bsrem)
print(f"BSREM: {time.time() - t0:.1f} s")

# %% 6. From counts to activity
voxel_mL = float(np.prod(object_meta.dr))  # SPECT voxel sizes are in cm

def bias_and_noise(stats, recon, region):
    """Bias and noise of a region at every iteration, in percent of its true concentration."""
    to_MBq_per_mL = activity / recon.sum().item() / voxel_mL
    mean = np.array(stats.mean[region]) * to_MBq_per_mL
    std = np.array(stats.std[region]) * to_MBq_per_mL
    truth = truth_MBq_per_mL[region]
    return 100 * (mean / truth - 1), 100 * std / truth

for region in regions:
    b_osem, n_osem = bias_and_noise(stats_osem, recon_osem, region)
    b_bsrem, n_bsrem = bias_and_noise(stats_bsrem, recon_bsrem, region)
    print(f"{region:10s} after {n_iters} iterations: OSEM bias {b_osem[-1]:+6.1f}% noise {n_osem[-1]:5.1f}%   "
          f"BSREM bias {b_bsrem[-1]:+6.1f}% noise {n_bsrem[-1]:5.1f}%")

# %% 7. Bias and noise in each region
fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
for ax, region in zip(axes.ravel(), regions):
    for stats, recon, name, color in [(stats_osem, recon_osem, 'OSEM', 'tab:blue'), (stats_bsrem, recon_bsrem, 'BSREM', 'tab:orange')]:
        bias, noise = bias_and_noise(stats, recon, region)
        ax.plot(bias, noise, '-', color=color, lw=1.5, label=name)
        ax.plot(bias[0], noise[0], 'o', mfc='none', color=color)
        ax.plot(bias[-1], noise[-1], 'o', color=color)
    ax.axvline(0, color='0.6', lw=0.8)
    ax.set_title(f'{region} ({volume_mL[region]:.1f} mL)')
    ax.set_xlabel('bias (%)')
    ax.set_ylabel('noise (%)')
    ax.grid(alpha=0.3)
axes[0, 0].legend()
axes[-1, -1].axis('off')

z = int(torch.nonzero(spheres)[:, 2].float().mean())  # the slice through the spheres' centres
truth_image = (water.float() * truth_MBq_per_mL['background']
               + spheres.float() * (sphere_MBq_per_mL - truth_MBq_per_mL['background'])).cpu()
images = [truth_image,
          recon_osem.cpu() * activity / recon_osem.sum().item() / voxel_mL,
          recon_bsrem.cpu() * activity / recon_bsrem.sum().item() / voxel_mL]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), constrained_layout=True)
for ax, image, title in zip(axes, images, ['Truth', f'OSEM {n_iters}×{n_subsets}', f'BSREM {n_iters}×{n_subsets}']):
    im = ax.imshow(image[32:96, 32:96, z].T, cmap='magma', origin='lower', vmax=1.2 * sphere_MBq_per_mL)
    ax.set_title(title)
    ax.axis('off')
fig.colorbar(im, ax=axes, shrink=0.8, label='MBq/mL')
