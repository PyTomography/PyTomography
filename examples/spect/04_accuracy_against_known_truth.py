"""Accuracy against known truth

Measure bias and noise in every organ of a simulated patient, with OSEM and BSREM.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_accuracy_known_truth.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

import os
from pathlib import Path

# Tutorial data: the folder set by the PYTOMOGRAPHY_DATA environment variable (see Tutorial data in the docs)
DATA = Path(os.environ.get("PYTOMOGRAPHY_DATA", "~/pytomography_data")).expanduser()
# Results go here, never into the data folder
OUTPUT = Path(os.environ.get("PYTOMOGRAPHY_OUTPUT", "pytomography_outputs")).expanduser() / "SPECT/SIMIND-MultiOrgan"
OUTPUT.mkdir(parents=True, exist_ok=True)

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

# %% 1. A noisy acquisition from per-organ simulations
path = DATA / 'SPECT' / 'SIMIND-MultiOrgan'
organs = ['bkg', 'liver', 'l_lung', 'r_lung', 'l_kidney', 'r_kidney', 'salivary', 'bladder']
activities = [2500, 450, 7, 7, 100, 100, 20, 90]  # MBq
acquisition_time = 15  # seconds per projection

windows = {w: [str(path / 'multi_projections' / organ / f'{w}.h00') for organ in organs]
           for w in ['photopeak', 'lowerscatter', 'upperscatter']}
object_meta, proj_meta = simind.get_metadata(windows['photopeak'][0])

torch.manual_seed(0)
projections = {w: torch.poisson(simind.combine_projection_data(files, activities) * acquisition_time)
               for w, files in windows.items()}
photopeak = projections['photopeak']

# Scatter in the photopeak window, estimated from the two neighbouring windows (triple energy window)
ww_peak, ww_lower, ww_upper = [simind.get_energy_window_width(windows[w][0]) for w in ['photopeak', 'lowerscatter', 'upperscatter']]
scatter = simind.compute_EW_scatter(projections['lowerscatter'], projections['upperscatter'], ww_lower, ww_upper, ww_peak)
print(f"{tuple(photopeak.shape)} projections, {photopeak.sum().item() / 1e6:.1f} million photopeak counts")

# %% 2. Organ masks from the ground truth
truth_shape = (768, 512, 512)            # z, y, x as stored
truth_voxel_mL = 0.075 * 0.075 * 0.15    # cm^3
masks, volume_mL = {}, {}
for organ in organs:
    truth = np.fromfile(path / 'phantom_organs' / f'{organ}_act_av.bin', dtype=np.float32).reshape(truth_shape)
    inside = torch.from_numpy((truth > 0).astype(np.float32)).permute(2, 1, 0)   # to x, y, z
    volume_mL[organ] = inside.sum().item() * truth_voxel_mL
    fraction = F.avg_pool3d(inside[None, None], kernel_size=(4, 4, 2))[0, 0]
    masks[organ] = (fraction > 0.5).to(pytomography.device)
truth_MBq_per_mL = {organ: a / volume_mL[organ] for organ, a in zip(organs, activities)}
for organ in organs:
    print(f"{organ:9s} {volume_mL[organ]:8.0f} mL   {truth_MBq_per_mL[organ]:.4f} MBq/mL   {int(masks[organ].sum()):7d} voxels")

# %% 3. System matrix and likelihood
attenuation_map = simind.get_attenuation_map(str(path / 'multi_projections' / 'mu208.hct'))
psf_meta = simind.get_psfmeta_from_header(windows['photopeak'][0])
system_matrix = SPECTSystemMatrix(
    obj2obj_transforms=[SPECTAttenuationTransform(attenuation_map), SPECTPSFTransform(psf_meta)],
    proj2proj_transforms=[],
    object_meta=object_meta,
    proj_meta=proj_meta)
likelihood = PoissonLogLikelihood(system_matrix, photopeak, additive_term=scatter)

# %% 4. Organ statistics during reconstruction
class OrganStatistics(Callback):
    def __init__(self, masks, n_subsets):
        self.masks = masks
        self.n_subsets = n_subsets
        self.mean = {organ: [] for organ in masks}
        self.std = {organ: [] for organ in masks}

    def run(self, object, n_iter, n_subset):
        if n_subset == self.n_subsets - 1:  # last subset: the end of an iteration
            for organ, mask in self.masks.items():
                values = object[mask]
                self.mean[organ].append(values.mean().item())
                self.std[organ].append(values.std().item())
        return object

# %% 5. Reconstruct with OSEM and BSREM
n_iters, n_subsets = 40, 8

t0 = time.time()
stats_osem = OrganStatistics(masks, n_subsets)
recon_osem = OSEM(likelihood)(n_iters=n_iters, n_subsets=n_subsets, callback=stats_osem)
print(f"OSEM:  {time.time() - t0:.1f} s")

t0 = time.time()
stats_bsrem = OrganStatistics(masks, n_subsets)
bsrem = BSREM(likelihood, prior=RelativeDifferencePrior(beta=0.3, gamma=2), relaxation_sequence=lambda n: 1 / (n / 50 + 1))
recon_bsrem = bsrem(n_iters=n_iters, n_subsets=n_subsets, callback=stats_bsrem)
print(f"BSREM: {time.time() - t0:.1f} s")

# %% 6. From counts to activity
voxel_mL = float(np.prod(object_meta.dr))  # SPECT voxel sizes are in cm

def bias_and_noise(stats, recon, organ):
    """Bias and noise of an organ at every iteration, in percent of its true concentration."""
    to_MBq_per_mL = sum(activities) / recon.sum().item() / voxel_mL
    mean = np.array(stats.mean[organ]) * to_MBq_per_mL
    std = np.array(stats.std[organ]) * to_MBq_per_mL
    truth = truth_MBq_per_mL[organ]
    return 100 * (mean / truth - 1), 100 * std / truth

for organ in organs:
    b_osem, n_osem = bias_and_noise(stats_osem, recon_osem, organ)
    b_bsrem, n_bsrem = bias_and_noise(stats_bsrem, recon_bsrem, organ)
    print(f"{organ:9s} after {n_iters} iterations: OSEM bias {b_osem[-1]:+6.1f}% noise {n_osem[-1]:5.1f}%   "
          f"BSREM bias {b_bsrem[-1]:+6.1f}% noise {n_bsrem[-1]:5.1f}%")
