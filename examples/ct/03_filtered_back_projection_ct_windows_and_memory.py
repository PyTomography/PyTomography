"""Filtered back projection (CT): windows and memory

Helical filtered back projection of a clinical chest CT with six windows, their noise against resolution, and the memory settings that fit a given machine.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_fbp_ct.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("CT/ldct-c145")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("CT/ldct-c145")

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import ndimage
import pytomography
from pytomography import io
from pytomography.metadata import ObjectMeta
from pytomography.projectors.CT import CTGen3SystemMatrix
from pytomography.algorithms import FilteredBackProjection
from pytomography.utils import get_fbp_filter, HannFilter, PeakMemory

# Memory: this tutorial needs about 13 GB of RAM: 2 GB of projections, the images it keeps, the reconstruction's
# chunks of views and the GPU libraries. RAM_GB caps what the reconstruction uses for its own work: with less, it
# rebins and filters fewer views at a time, which is a little slower and gives the same image. Section 3 prints the
# estimate before anything runs (estimate_memory) and the settings that fit less memory (plan_memory); section 4
# reconstructs with them. Most of the 13 GB is not the reconstruction's own, so a lower RAM_GB saves a few GB at most.
RAM_GB = 25
pytomography.set_memory_budget(RAM_GB)

# %% 1. The projections
folder = DATA / 'CT' / 'ldct-c145'
t0 = time.time()
proj, object_meta, proj_meta = io.CT.read_dicom_ct_pd(
    folder / 'full_dose_projections',
    object_meta=ObjectMeta(dr=(0.662109, 0.662109, 1.0), shape=(512, 512, 316)))  # mm, voxels: the scanner's grid
print(f"{proj.shape[0]} views of {proj.shape[1]} x {proj.shape[2]} detector elements, read in {time.time() - t0:.0f} s")
behind = proj_meta.photon_counts[::100, :, None] * torch.exp(-proj[::100])  # every 100th view
print(f"photons reaching a detector element, median over the rays through the patient: {behind[proj[::100] > 1].median():.0f}")
system_matrix = CTGen3SystemMatrix(object_meta, proj_meta)
to_hu = lambda image: io.CT.to_hu(image, proj_meta).cpu().numpy()

# %% 3. Memory, before reconstructing
fbp = FilteredBackProjection(proj, system_matrix, filter='hann', slice_thickness=1.25)
print(fbp.estimate_memory())
print()
print(system_matrix.estimate_memory(n_subsets=40, held={'projections': proj}))

plan = fbp.plan_memory(ram_gb=10, gpu_gb=2)
print(plan)

# %% 4. Reconstruct, with the default settings and with the plan's
t0 = time.time()
with PeakMemory() as peak:
    hu = to_hu(fbp())
print(f"default settings: {time.time() - t0:.0f} s, GPU peak {peak.peak / 1e9:.2f} GB")

pytomography.set_memory_budget(plan.settings['memory_budget_gb'])
fbp_planned = FilteredBackProjection(proj, system_matrix, filter='hann', slice_thickness=1.25, gpu_budget=plan.settings['gpu_budget'])
print(fbp_planned.estimate_memory())
t0 = time.time()
with PeakMemory() as peak:
    hu_planned = to_hu(fbp_planned())
print(f"the plan's settings: {time.time() - t0:.0f} s, GPU peak {peak.peak / 1e9:.2f} GB; "
      f"largest difference from the default image: {np.abs(hu_planned - hu).max():.1e} HU")
pytomography.set_memory_budget(RAM_GB)
del hu_planned

# %% 5. Windows compared
scanner_files = sorted((folder / 'full_dose_images').glob('*.dcm'))
scanner = io.load_image(scanner_files)  # in HU, with its place in the patient
scanner_hu = scanner.on(object_meta, fill=-3000).cpu().numpy()  # -3000: outside the scanner's images
# The body in the scanner's images and a block of liver: they come with the dataset, in its folder
regions = io.load_mask(folder / 'ldct_c145_rois.seg.nrrd')

liver = regions.on(object_meta, label='liver', threshold=0.5).cpu().numpy()  # 26 x 26 x 16 mm of liver
noise = {'Hann': float(hu[liver].std())}
for name, window, thickness in [*[(n, w, 1.25) for n, w in windows.items() if n != 'Hann'], ('Hann, no slice_thickness', 'hann', None)]:
    reconstruction = FilteredBackProjection(proj, system_matrix, filter=window, slice_thickness=thickness)
    estimate = reconstruction.estimate_memory()
    t0 = time.time()
    image = to_hu(reconstruction())
    noise[name] = float(image[liver].std())
    print(f"{name:24s} estimate {estimate.ram_gb:4.1f} GB RAM, {estimate.gpu_gb:.1f} GB GPU; {time.time() - t0:3.0f} s; "
          f"liver noise {noise[name]:5.1f} HU")
    if name == 'Ram-Lak':
        hu_ramlak = image
    del image
print(f"{'scanner':24s} liver noise {scanner_hu[liver].std():5.1f} HU; liver mean: scanner {scanner_hu[liver].mean():.1f} HU, "
      f"ours {hu[liver].mean():.1f} HU")

valid = scanner_hu > -1500  # inside the scanner's field of view
blur = lambda a: ndimage.gaussian_filter(np.where(valid, a, -1000.0), [2 / d for d in object_meta.dr])  # 2 mm
body = regions.on(object_meta, label='body', threshold=0.5).cpu().numpy()
for name, image in images[1:]:
    rms = np.sqrt(((blur(image) - blur(scanner_hu))[body] ** 2).mean())
    print(f"{name:13s} RMS difference from the scanner's image after a 2 mm blur: {rms:4.1f} HU")

# %% Save the images
saved = io.save_dicom(hu, OUTPUT / 't_fbp_ct' / 'fbp_hann', object_meta, units='HU', slice_thickness=1.25,
                      reference=scanner_files[0], series_description='FBP, Hann, 1.25 mm', overwrite=True)
