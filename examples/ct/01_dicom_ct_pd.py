"""DICOM-CT-PD

OS-SART on raw projections from a 3rd-generation clinical CT scanner.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_CT_GEN3.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% DICOM-CT-PD (3rd generation CT)
import os
from pathlib import Path

# Tutorial data: the folder set by the PYTOMOGRAPHY_DATA environment variable (see Tutorial data in the docs)
DATA = Path(os.environ.get("PYTOMOGRAPHY_DATA", "~/pytomography_data")).expanduser()
# Results go here, never into the data folder
OUTPUT = Path(os.environ.get("PYTOMOGRAPHY_OUTPUT", "pytomography_outputs")).expanduser() / "CT/ldct-c145"
OUTPUT.mkdir(parents=True, exist_ok=True)

import struct
import time
import numpy as np
import pydicom
import torch
import matplotlib.pyplot as plt
from pytomography.metadata import ObjectMeta
from pytomography.io.CT import dicom_ct_pd
from pytomography.projectors.CT import CTGen3SystemMatrix
from pytomography.algorithms import SART

# %% 1. Load the projections
folder = DATA / 'CT' / 'ldct-c145'
paths = [str(p) for p in (folder / 'full_dose_projections').glob('*.dcm')]
headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in paths]
order = np.argsort([h.InstanceNumber for h in headers])
paths = [paths[i] for i in order]
source_z_mean = np.mean([struct.unpack('<f', h[0x7031, 0x1002].value)[0] for h in headers])  # mm, table position
mu_water = float(headers[0][0x7041, 0x1001].value.decode())  # attenuation of water, mm^-1

t0 = time.time()
proj, proj_meta = dicom_ct_pd.get_projections_and_metadata_gen3(paths)
print(f"{proj.shape[0]} views of {proj.shape[1]} × {proj.shape[2]} detector elements, read in {time.time() - t0:.0f} s")
print(f"source travels {float(proj_meta.source_zs.max() - proj_meta.source_zs.min()):.0f} mm; water: {mu_water} mm^-1")

# %% 2. Object space
object_meta = ObjectMeta(dr=(1.0, 1.0, 1.0), shape=(512, 512, 384))  # mm, voxels
system_matrix = CTGen3SystemMatrix(object_meta, proj_meta, N_splits=1, device='cpu')

# %% 3. Reconstruct
t0 = time.time()
recon = SART(system_matrix, proj)(n_iters=3, n_subsets=40)
print(f"OS-SART 3 × 40: {time.time() - t0:.0f} s")

# %% 4. Hounsfield units, and the scanner's reconstruction
hu = (1000 * (recon - mu_water) / mu_water).cpu().numpy()
Nx, Ny, Nz = object_meta.shape
k = 200
z_k = source_z_mean - (k - (Nz - 1) / 2) * object_meta.dr[2]

scanner_files = sorted((folder / 'full_dose_images').glob('*.dcm'))
scanner_headers = [pydicom.dcmread(f, stop_before_pixels=True) for f in scanner_files]
nearest = int(np.argmin([abs(float(h.ImagePositionPatient[2]) - z_k) for h in scanner_headers]))
scanner = pydicom.dcmread(scanner_files[nearest])
scanner_hu = scanner.pixel_array * float(scanner.RescaleSlope) + float(scanner.RescaleIntercept)
print(f"slice {k} is at z = {z_k:.1f} mm; nearest scanner slice at z = {float(scanner.ImagePositionPatient[2]):.1f} mm")
