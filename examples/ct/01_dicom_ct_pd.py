"""DICOM-CT-PD

Filtered back projection of raw projections from a 3rd-generation clinical CT scanner, then the settings that reproduce the scanner's own images.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_CT_GEN3.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
# %% DICOM-CT-PD (3rd generation CT)
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("CT/ldct-c145")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("CT/ldct-c145")

import time
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

# %% 2. Object space
system_matrix = CTGen3SystemMatrix(object_meta, proj_meta)

# %% 3. Filtered back projection
reconstruction = FilteredBackProjection(proj, system_matrix, filter='hann', slice_thickness=1.25)  # slices 1.25 mm thick
print(reconstruction.estimate_memory())  # its peak RAM and GPU memory, and what to change to use less
t0 = time.time()
print(f"filtered back projection: {time.time() - t0:.0f} s")

# %% 4. The scanner's reconstruction
scanner_files = sorted((folder / 'full_dose_images').glob('*.dcm'))
scanner = io.load_image(scanner_files)  # in HU, with its place in the patient

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

matched = FilteredBackProjection(proj, system_matrix, filter=window, slice_thickness=1.25)()
hu_matched = io.CT.to_hu(matched, proj_meta).cpu().numpy()

# %% Save the images
saved = io.save_dicom(hu_matched, OUTPUT / 't_CT_GEN3' / 'fbp_matched', object_meta, units='HU', slice_thickness=1.25,
                      reference=scanner_files[0], series_description='FBP, advanced settings', overwrite=True)
