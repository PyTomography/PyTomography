"""DICOM-CT-PD

Filtered back projection of raw projections from a 3rd-generation clinical CT scanner, then the settings that reproduce the scanner's own images.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_CT_GEN3.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% DICOM-CT-PD (3rd generation CT)
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("CT/ldct-c145")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("CT/ldct-c145")

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pytomography
from pytomography import io
from pytomography.metadata import ObjectMeta
from pytomography.io.CT import preprocessing
from pytomography.projectors.CT import CTGen3SystemMatrix
from pytomography.algorithms import FilteredBackProjection

# Memory: this tutorial needs about 16 GB of RAM (15.7 GB measured on Windows). RAM_GB caps what PyTomography's
# reconstructions use for their own work: with less, they rebin and filter fewer views at a time, which is a little
# slower and gives the same images. The first reconstruction prints its estimate before it runs (estimate_memory): its
# peak RAM and GPU memory, what they are made of, and what to change to use less. Most of the 16 GB is what the
# tutorial keeps (2 GB of projections, five images) and the GPU libraries, so a lower RAM_GB saves little: the
# tutorial needs about 16 GB whatever it is set to.
RAM_GB = 25
pytomography.set_memory_budget(RAM_GB)

# %% 1. Load the projections
folder = DATA / 'CT' / 'ldct-c145'
grid = ObjectMeta(dr=(0.662109, 0.662109, 1.0), shape=(512, 512, 316))  # mm, voxels: the scanner's grid (section 2)
t0 = time.time()
proj, object_meta, proj_meta = io.CT.read_dicom_ct_pd(folder / 'full_dose_projections', object_meta=grid)
print(f"{proj.shape[0]} views of {proj.shape[1]} × {proj.shape[2]} detector elements, read in {time.time() - t0:.0f} s")
print(f"water: {proj_meta.water_attenuation} mm^-1; central detector column: {float(proj_meta.detector_centers_col_idx[0])}")

fig, axes = plt.subplots(5, 1, figsize=(6, 5), constrained_layout=True)
for ax, i in zip(axes, [0, 2000, 4000, 6000, 8000]):
    ax.imshow(proj[i].T, cmap='Greys_r', aspect='auto')
    ax.set_ylabel(f'view {i}', rotation=0, ha='right', va='center')
    ax.set_xticks([]); ax.set_yticks([])

# %% 2. Object space
system_matrix = CTGen3SystemMatrix(object_meta, proj_meta)

# %% 3. Filtered back projection
reconstruction = FilteredBackProjection(proj, system_matrix, filter='hann', slice_thickness=1.25)  # slices 1.25 mm thick
print(reconstruction.estimate_memory())  # its peak RAM and GPU memory, and what to change to use less
t0 = time.time()
fbp = reconstruction()
print(f"filtered back projection: {time.time() - t0:.0f} s")
hu = io.CT.to_hu(fbp, proj_meta).cpu().numpy()

# %% 4. The scanner's reconstruction
scanner_files = sorted((folder / 'full_dose_images').glob('*.dcm'))
scanner = io.load_image(scanner_files)  # in HU, with its place in the patient
scanner_hu = scanner.on(object_meta, fill=-3000).cpu().numpy()  # -3000: no scanner image here

k, j = 157, 256  # an axial slice through the chest, a coronal slice through the middle (k runs from the feet up)
dx, dz = object_meta.dr[0], object_meta.dr[2]
# The body and the soft tissue of the scanner's images (above -700 HU after a 2 mm blur; 10 to 70 HU after a 7 x 7 x 3
# voxel mean, eroded): they come with PyTomography, in the dataset's folder
regions = io.load_mask(folder / 'ldct_c145_rois.seg.nrrd')
r = np.hypot(*np.meshgrid(*[(np.arange(n) - (n - 1) / 2) * dx for n in object_meta.shape[:2]], indexing='ij'))[:, :, None]

def blurred(a, valid):
    # 2 mm Gaussian blur, with everything outside the scanner's field of view set to air
    return ndimage.gaussian_filter(np.where(valid, a, -1000.0), (2 / dx, 2 / dx, 2 / dz))

def compare(ours, title):
    valid = scanner_hu > -1500  # inside the scanner's field of view (it pads outside with a lower value)
    diff = np.where(valid, blurred(ours, valid) - blurred(scanner_hu, valid), np.nan)
    body = valid & regions.on(object_meta, label='body', threshold=0.5).cpu().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    # axial: anterior at the top; coronal: the head at the top
    for row, cut, aspect, origin in ((0, lambda a: a[:, :, k].T, 1, 'upper'), (1, lambda a: a[:, j, :].T, dz / dx, 'lower')):
        axes[row, 0].imshow(cut(scanner_hu), cmap='Greys_r', vmin=-160, vmax=240, aspect=aspect, origin=origin)
        axes[row, 1].imshow(cut(ours), cmap='Greys_r', vmin=-160, vmax=240, aspect=aspect, origin=origin)
        im = axes[row, 2].imshow(cut(diff), cmap='RdBu_r', vmin=-80, vmax=80, aspect=aspect, origin=origin)
    for ax, t in zip(axes[0], ('scanner', title, 'ours - scanner, 2 mm blur')):
        ax.set_title(t)
    for ax in axes.flat:
        ax.axis('off')
    fig.colorbar(im, ax=axes[:, 2], shrink=0.6, label='HU')
    print(f"RMS difference over the body after a 2 mm blur: {np.sqrt((diff[body] ** 2).mean()):.1f} HU")

def radial_trend(ours):
    # mean of ours - scanner in soft tissue, in 20 mm rings
    valid = scanner_hu > -1500
    soft_tissue = valid & regions.on(object_meta, label='soft_tissue', threshold=0.5).cpu().numpy()
    edges = np.arange(0, 201, 20)
    d = ours - scanner_hu
    return edges[:-1] + 10, [d[soft_tissue & (r >= a) & (r < b)].mean() for a, b in zip(edges[:-1], edges[1:])]

compare(hu, 'PyTomography FBP')
trend_basic = radial_trend(hu)

# %% A. The table feed
del proj  # 2 GB: free it before reading the projections again
proj, object_meta, proj_meta = io.CT.read_dicom_ct_pd(folder / 'full_dose_projections', object_meta=grid, table_feed='pitch')
system_matrix = CTGen3SystemMatrix(object_meta, proj_meta)
scanner_hu = scanner.on(object_meta, fill=-3000).cpu().numpy()  # our slices moved slightly, so resample the scanner's images again

# %% B. Fit the per-column scale
t0 = time.time()
fit = preprocessing.fit_column_scale(proj, proj_meta, system_matrix, io.CT.from_hu(scanner_hu, proj_meta),
                                     filter='hann', slice_thickness=1.25)
print(f"g0 = {fit['g0']:+.4f}, g2 = {fit['g2']:+.4f}, fitted in {time.time() - t0:.0f} s")
proj = preprocessing.scale_columns(proj, proj_meta, **fit)

# %% C. Fit the scanner's window
t0 = time.time()
window = preprocessing.fit_window(proj, proj_meta, system_matrix, io.CT.from_hu(scanner_hu, proj_meta), slice_thickness=1.25)
print(f"fitted in {time.time() - t0:.0f} s")

fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
ax.plot(window.frequencies, window.values, 'o-')
ax.axhline(1, color='k', lw=0.8)
ax.set(xlabel='spatial frequency (cycles per mm)', ylabel='window on the ramp filter', ylim=(0, 1.15))

matched = FilteredBackProjection(proj, system_matrix, filter=window, slice_thickness=1.25)()
hu_matched = io.CT.to_hu(matched, proj_meta).cpu().numpy()
compare(hu_matched, 'PyTomography FBP, advanced settings')

fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax.plot(*trend_basic, 'o-', label='basic')
ax.plot(*radial_trend(hu_matched), 'o-', label='table feed, per-column scale, scanner window')
ax.axhline(0, color='k', lw=0.8)
ax.set(xlabel='distance from the axis of rotation (mm)', ylabel='ours - scanner in soft tissue (HU)')
ax.legend()

# %% Save the images
saved = io.save_dicom(hu_matched, OUTPUT / 't_CT_GEN3' / 'fbp_matched', object_meta, units='HU', slice_thickness=1.25,
                      reference=scanner_files[0], series_description='FBP, advanced settings', overwrite=True)
