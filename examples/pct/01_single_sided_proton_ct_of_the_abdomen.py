"""Single-sided proton CT of the abdomen

A simulated 330 MeV scan with a calorimeter behind the patient and nothing in front, reconstructed by two-pass filtered back projection and SART along curved paths, and scored on organs and range errors against the true stopping power.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_pct_abdomen.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
# %% Proton CT: a single-sided scan of the abdomen
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("PCT/GATE-Abdomen330-DTC")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("PCT/GATE-Abdomen330-DTC")

import time
import numpy as np
import torch
import pytomography
from pytomography import io
from pytomography.projectors.PCT import PCTSystemMatrix, hull_from_image
from pytomography.algorithms import FilteredBackProjection, SART

# Memory: the scan (a quarter of the simulation's protons, 47 million) takes 1.9 GB of RAM, and the tutorial peaks at
# about 7 GB of RAM and 1 GB of GPU memory (measured on Windows); the filtered back projection prints its estimate
# before it runs (estimate_memory). Protons go to the GPU a view at a time, in calls of at most 1.5 GB, so RAM_GB
# changes nothing here. To use less RAM, read fewer protons: io.PCT.load_listmode(..., fraction=0.5) keeps half of
# each view's protons as it reads (noisier images). The record's full file (all 188.6 million protons) holds 7.5 GB
# of protons.
RAM_GB = 25
pytomography.set_memory_budget(RAM_GB)

# %% 1. The scan
folder = DATA / 'PCT' / 'GATE-Abdomen330-DTC'
truth = io.load_image(folder / 'truth_rsp.nii.gz')  # the RSP the simulation used

t0 = time.time()
wepl, object_meta, proj_meta = io.PCT.load_listmode(folder / 'abdomen330_spot_likely_quarter.h5',
                                                    dr=truth.voxel_size, shape=truth.shape)
print(proj_meta)
print(f"read in {time.time() - t0:.0f} s; WEPL from {float(wepl.min()):.0f} to {float(wepl.max()):.0f} mm")

fwhm = proj_meta.entry_fwhm.numpy()
print(f"entry FWHM: median {np.median(fwhm):.1f} mm (5th to 95th percentile {np.percentile(fwhm, 5):.1f} to "
      f"{np.percentile(fwhm, 95):.1f} mm)")

# %% 2. Filtered back projection, in two passes
system_matrix = PCTSystemMatrix(object_meta, proj_meta, path='straight', band=5)
fbp = FilteredBackProjection(wepl, system_matrix, filter='hann')
print(fbp.estimate_memory())  # its peak RAM and GPU memory, before it runs
t0 = time.time()
first_pass = fbp()
hull = hull_from_image(first_pass, object_meta)
system_matrix_hull = PCTSystemMatrix(object_meta, proj_meta, path='straight', band=5, hull=hull)
second_pass = FilteredBackProjection(wepl, system_matrix_hull, filter='hann', line='hull')()
print(f"two passes and the outline: {time.time() - t0:.1f} s")

# %% 3. SART along curved paths
support = hull_from_image(first_pass, object_meta, grow=2)  # the outline grown by 2 voxels in each slice
image = second_pass.clamp(min=0)
t0 = time.time()
for band, n_subsets in [(3, 180), (3, 180), (3, 60), (5, 20)]:
    system_matrix = PCTSystemMatrix(object_meta, proj_meta, band=band, hull=hull)
    image = SART(system_matrix, wepl, object_initial=image, relaxation_sequence=lambda _: 0.5,
                 support=support)(n_iters=1, n_subsets=n_subsets)
sart = image
print(f"SART: {time.time() - t0:.0f} s")

# %% 4. Against the truth
true_rsp = truth.on(object_meta).to(pytomography.device)
z = (torch.arange(object_meta.shape[2]) - (object_meta.shape[2] - 1) / 2) * object_meta.dr[2]
scanned = (z.abs() <= 50).to(pytomography.device)[None, None, :]
labels = io.load_mask(folder / 'organs.nii.gz')
organs = {name: labels.on(object_meta, label=value, threshold=0.5).to(pytomography.device) & scanned
          for name, value in [('liver', 1), ('left kidney', 2), ('right kidney', 3)]}
print(f"{'':13s}{'true RSP':>9s}{'FBP':>9s}{'SART':>9s}   noise (SD of the error), FBP and SART")
for name, mask in organs.items():
    t = float(true_rsp[mask].mean())
    err = [100 * (float(r[mask].mean()) / t - 1) for r in (second_pass, sart)]
    sd = [float((r - true_rsp)[mask].std()) for r in (second_pass, sart)]
    print(f"{name:13s}{t:9.4f}{err[0]:+8.2f}%{err[1]:+8.2f}%   {sd[0]:.3f}  {sd[1]:.3f}")

# %% 5. Range errors
system_matrix = PCTSystemMatrix(object_meta, proj_meta, band=1, hull=hull)
system_matrix.set_n_subsets(36)  # subset 0: views 0, 36, 72, ...
true_sums = system_matrix.forward(true_rsp, subset_idx=0)
through = true_sums > 20  # protons that crossed the body
for image, name in [(second_pass, 'FBP'), (sart, 'SART')]:
    error = (system_matrix.forward(image, subset_idx=0) - true_sums)[through]
    print(f"{name:5s} range error: mean {float(error.mean()):+.2f} mm, 95% within "
          f"{float(error.abs().quantile(0.95)):.2f} mm ({int(through.sum()):,} protons)")

# %% Save the image
saved = io.save_nifti(sart, OUTPUT / 't_pct_abdomen' / 'sart_rsp.nii.gz', object_meta, units='RSP')
