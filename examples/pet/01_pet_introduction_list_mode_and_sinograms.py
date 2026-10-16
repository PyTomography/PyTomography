"""PET introduction: list mode and sinograms

Reconstruct the same true coincidences as list mode and as sinograms, with normalisation and attenuation.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_pet_introduction.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("PET/GATE-mMR-BrainInWater")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("PET/GATE-mMR-BrainInWater")

import time
import numpy as np
import torch
import pytomography
from pytomography import io
from pytomography.metadata.PET import PETSinogramPolygonProjMeta
from pytomography.projectors.PET import PETLMSystemMatrix, PETSinogramSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM
from pytomography.transforms.shared import GaussianFilter

# %% How much memory this needs
RAM_GB = 25 # memory (RAM) this tutorial may use, in GB (at least 10); it needs about 16.5 GB however low this is
pytomography.set_memory_budget(RAM_GB)

# %% 1. The scanner and the true coincidences
path = DATA / 'PET' / 'GATE-mMR-BrainInWater'
macro = path / 'mMR_Geometry.mac'  # the scanner, as the simulation's GATE macro describes it
paths = [path / 'simple_phantom' / f'mMR_voxBrain_withSimplePhantom_{i}.root' for i in range(1, 55)]  # its 54 runs
cache = OUTPUT / 'gate_trues_norm.pt'
if not cache.exists():
    # doi: the lines of response end 9 mm inside the crystals, the mean depth of interaction. normalization: the
    # weights of section 2
    torch.save(io.PET.read_gate(paths, macro, events='trues', doi=9,
                                normalization=path / 'normalization_weights.npy'), cache)
detector_ids, object_meta, proj_meta = torch.load(cache, weights_only=False)
info = proj_meta.info
print(f"{info['NrRings']} rings of {info['NrCrystalsPerRing']} crystals, radius {info['radius']} mm")
print(f"{detector_ids.shape[0] / 1e6:.1f} million true coincidences")

# %% 2. Normalisation
# The weights come with the dataset. They were computed from the calibration scan (36 ROOT files, 46 GB, in
# PET/GATE-mMR-Brain/normalization_scan), randoms left out (PyTomography PR #290):
# normalization_weights = io.PET.gate_normalization(normalization_paths, info, cylinder_radius=318)
normalization_weights = proj_meta.weights_sensitivity  # one per crystal pair, in torch.combinations order
print(f"{normalization_weights.numel() / 1e6:.0f} million crystal pairs, "
      f"{int((normalization_weights == 0).sum()) / 1e6:.0f} million of them without calibration data")

# %% 3. Object space, attenuation and resolution
object_meta = object_meta.with_grid(dr=(2, 2, 2), shape=(128, 128, 96))  # mm, voxels
# The phantom's attenuation map (1/cm), an image in GATE's world: gate_image places it in the patient as read_gate
# placed the scanner, and attenuation_map puts it on the grid, in 1/mm
atten_map = io.attenuation_map(io.PET.gate_image(path / 'umap_mMR_brainSimplePhantom.hv', proj_meta, units='1/cm'),
                               object_meta)
psf_transform = GaussianFilter(3.)  # mm

# %% 4. List-mode reconstruction
system_matrix_lm = PETLMSystemMatrix(object_meta, proj_meta, obj2obj_transforms=[psf_transform],
                                     N_splits=10, attenuation_map=atten_map)
likelihood_lm = PoissonLogLikelihood(system_matrix_lm)
osem_lm = OSEM(likelihood_lm)
print(osem_lm.estimate_memory(n_subsets=14))  # its peak RAM and GPU memory, before it runs

t0 = time.time()
recon_lm = osem_lm(n_iters=4, n_subsets=14)
print(f"list mode: {time.time() - t0:.1f} s")

# %% 5. Sinogram reconstruction
sinogram = io.PET.listmode_to_sinogram(detector_ids, info)
normalization_sinogram = io.PET.normalization_sinogram(normalization_weights, info)
print(f"sinogram {tuple(sinogram.shape)}: {sinogram.numel() / 1e6:.0f} million bins for {detector_ids.shape[0] / 1e6:.1f} million events")

proj_meta_sino = PETSinogramPolygonProjMeta(info)
system_matrix_sino = PETSinogramSystemMatrix(
    object_meta, proj_meta_sino,
    obj2obj_transforms=[psf_transform],
    sinogram_sensitivity=normalization_sinogram,
    attenuation_map=atten_map,
    N_splits=10,
    device='cpu')  # keep the large sinograms in host memory
likelihood_sino = PoissonLogLikelihood(system_matrix_sino, sinogram)
osem_sino = OSEM(likelihood_sino)
print(osem_sino.estimate_memory(n_subsets=14))

t0 = time.time()
recon_sino = osem_sino(n_iters=4, n_subsets=14)
print(f"sinogram: {time.time() - t0:.1f} s")

# %% 6. Comparison
def difference(a, b):
    return 100 * float((a - b).abs().sum() / a.abs().sum())

print(f"OSEM 4 × 14: the images differ by {difference(recon_lm, recon_sino):.1f}% voxel by voxel, "
      f"and their totals by {100 * float(recon_sino.sum() / recon_lm.sum() - 1):+.2f}%")

t0 = time.time()
mlem_lm = OSEM(likelihood_lm)(n_iters=40, n_subsets=1)
mlem_sino = OSEM(likelihood_sino)(n_iters=40, n_subsets=1)
print(f"MLEM, 40 iterations ({time.time() - t0:.0f} s): the images differ by {difference(mlem_lm, mlem_sino):.1f}%")

# %% Save the images
for name, image in [('list_mode', recon_lm), ('sinogram', recon_sino)]:
    io.save_dicom(image, OUTPUT / 't_pet_introduction' / name, object_meta, series_description=f'OSEM 4x14, {name}',
                  overwrite=True)
