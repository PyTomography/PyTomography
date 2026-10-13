"""Cone-beam micro-CT

FDK and OS-SART on a laboratory micro-CT scan of glass beads, with the centre of rotation found from the data.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_CT_microct.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("CT/SophiaBeads-256")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("CT/SophiaBeads-256")

import time
import numpy as np
import torch
import pytomography
from pytomography import io
from pytomography.metadata.CT import CTConeBeamFlatPanelProjMeta
from pytomography.projectors.CT import CTConeBeamFlatPanelSystemMatrix
from pytomography.algorithms import FilteredBackProjection, SART

# Memory: the FDK and OS-SART reconstructions below print their estimates before they run (estimate_memory): their
# peak RAM and GPU memory, what they are made of, and what to change to use less. The tutorial peaks at about 13 GB of
# RAM and 7 GB of GPU memory (measured on Windows). Its reconstructions do not split their work by the memory budget,
# so a lower RAM_GB changes nothing here: with binning = 4 it needs about 13 GB whatever RAM_GB is. To use less RAM and
# GPU memory, set binning = 8 below (a grid with eight times fewer voxels), or use fewer OS-SART subsets (each keeps an
# image on the GPU).
RAM_GB = 25
pytomography.set_memory_budget(RAM_GB)

# %% 1. The scan
folder = DATA / 'CT' / 'SophiaBeads-256' / 'SophiaBeads_256_averaged'
binning = 4  # 8 gives a grid with eight times fewer voxels, if your GPU has less than 8 GB of memory

t0 = time.time()
proj, object_meta, proj_meta = io.CT.read_nikon(folder, binning=binning)
proj = proj.to(pytomography.device)
magnification = proj_meta.DSD / proj_meta.DSO
step = np.degrees(float(torch.diff(proj_meta.angles).abs().mean()))
print(f"{len(proj_meta.angles)} projections {step:.2f}° apart, binned to {proj.shape[1]} × {proj.shape[2]} pixels of "
      f"{proj_meta.dr[0]:.1f} mm, magnification {magnification:.2f}")
print(f"projections {tuple(proj.shape)} read in {time.time() - t0:.0f} s")

# %% 3. Geometry
n = object_meta.shape[0]  # 500 voxels across

def system_matrix_for(cor, n_slices):
    """read_nikon's geometry with the rotation axis cor mm from the central ray, on n_slices central slices of its grid"""
    proj_meta_cor = CTConeBeamFlatPanelProjMeta(
        proj_meta.angles, proj_meta.z_locations,
        detector_radius=proj_meta.detector_radius, beam_radius=proj_meta.beam_radius,
        shape=proj_meta.shape, dr=proj_meta.dr, COR=cor)
    return CTConeBeamFlatPanelSystemMatrix(object_meta.with_grid(shape=(n, n, n_slices)), proj_meta_cor)

# %% 4. The centre of rotation
def sharpness(image):
    s = image[:, :, image.shape[2] // 2]
    return float(((s[1:] - s[:-1]) ** 2).mean() + ((s[:, 1:] - s[:, :-1]) ** 2).mean())

cors = np.round(np.arange(0.0, 0.5, 0.02), 2)  # mm
trials = {cor: FilteredBackProjection(proj, system_matrix_for(cor, 4), filter='ram-lak')() for cor in cors}
scores = [sharpness(trials[cor]) for cor in cors]
best_cor = float(cors[int(np.argmax(scores))])
print(f"sharpest at COR = {best_cor} mm")

# %% 5. FDK
system_matrix = system_matrix_for(best_cor, n)
fdk = FilteredBackProjection(proj, system_matrix, filter='ram-lak')
print(fdk.estimate_memory())  # its peak RAM and GPU memory, before it runs
t0 = time.time()
recon_fdk = fdk()
print(f"FDK: {time.time() - t0:.1f} s")

# %% 6. OS-SART
# OS-SART keeps one H_m^T 1 image per subset (on the GPU), so here fewer subsets take less memory
print(system_matrix.estimate_memory(n_subsets=5, held={'projections': proj}))
t0 = time.time()
recon_sart = SART(system_matrix, proj, object_initial=recon_fdk.clamp(min=0))(n_iters=3, n_subsets=5)
print(f"OS-SART 3 × 5: {time.time() - t0:.1f} s")

# %% 7. Comparison
air = (slice(0, 40), slice(0, 40), slice(n // 2 - 20, n // 2 + 20))  # a corner of the grid outside the tube
for recon, name in [(recon_fdk, 'FDK'), (recon_sart, 'OS-SART')]:
    a = recon[air]
    print(f"{name:8s} air: mean {float(a.mean()):+.4f} mm^-1, standard deviation {float(a.std()):.4f} mm^-1")

# %% Save the images
saved = io.save_dicom(recon_sart, OUTPUT / 't_CT_microct' / 'os_sart', system_matrix.object_meta,
                      series_description='OS-SART 3x5 from FDK', overwrite=True)
