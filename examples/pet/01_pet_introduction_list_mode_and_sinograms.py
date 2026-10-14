"""PET introduction: list mode and sinograms

Reconstruct the same true coincidences as list mode and as sinograms, with normalisation and attenuation.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_pet_introduction.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("PET/GATE-mMR-BrainInWater")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("PET/GATE-mMR-BrainInWater")

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETSinogramPolygonProjMeta
from pytomography.projectors.PET import PETLMSystemMatrix, PETSinogramSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM
from pytomography.transforms.shared import GaussianFilter
from pytomography.io.PET import gate

# %% How much memory this needs
RAM_GB = 25 # memory (RAM) this tutorial may use, in GB
pytomography.set_memory_budget(RAM_GB)

# %% 1. The scanner
path = DATA / 'PET' / 'GATE-mMR-BrainInWater'
info = gate.get_detector_info(path=os.path.join(path, 'mMR_Geometry.mac'), mean_interaction_depth=9, min_rsector_difference=0)
print(f"{info['NrRings']} rings of {info['NrCrystalsPerRing']} crystals, radius {info['radius']} mm")

# %% 2. Normalisation
# The weights come with the dataset. They were computed from the calibration scan (36 ROOT files, 46 GB, in
# PET/GATE-mMR-Brain/normalization_scan) with randoms left out (PyTomography PR #290):
# normalization_weights = gate.get_normalization_weights_cylinder_calibration(
#     normalization_paths, info, cylinder_radius=318, include_randoms=False)  # shell radius in mm
normalization_weights = torch.from_numpy(np.load(os.path.join(path, 'normalization_weights.npy')))

# %% 3. The true coincidences
ids_file = os.path.join(OUTPUT, 'detector_ids_primary_only.pt')
if not os.path.exists(ids_file):
    paths = [os.path.join(path, f'simple_phantom/mMR_voxBrain_withSimplePhantom_{i}.root') for i in range(1, 55)]
    detector_ids = gate.get_detector_ids_from_root(paths, info, include_randoms=False, include_scatters=False)
    torch.save(detector_ids, ids_file)
detector_ids = torch.load(ids_file)
print(f"{detector_ids.shape[0] / 1e6:.1f} million true coincidences")

# %% 4. Object space, attenuation and resolution
object_meta = ObjectMeta(dr=(2, 2, 2), shape=(128, 128, 96))  # mm, voxels
atten_map = gate.get_aligned_attenuation_map(os.path.join(path, 'umap_mMR_brainSimplePhantom.hv'), object_meta)
atten_map = atten_map.to(pytomography.dtype).to(pytomography.device)
psf_transform = GaussianFilter(3.)  # mm

# %% 5. List-mode reconstruction
proj_meta_lm = PETLMProjMeta(detector_ids=detector_ids[:, :2], info=info, weights_sensitivity=normalization_weights)
system_matrix_lm = PETLMSystemMatrix(object_meta, proj_meta_lm, obj2obj_transforms=[psf_transform],
                                     N_splits=10, attenuation_map=atten_map)
likelihood_lm = PoissonLogLikelihood(system_matrix_lm)

t0 = time.time()
recon_lm = OSEM(likelihood_lm)(n_iters=4, n_subsets=14)
print(f"list mode: {time.time() - t0:.1f} s")

# %% 6. Sinogram reconstruction
sinogram = gate.listmode_to_sinogram(detector_ids, info)
normalization_sinogram = gate.get_norm_sinogram_from_listmode_data(normalization_weights, info)
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

t0 = time.time()
recon_sino = OSEM(likelihood_sino)(n_iters=4, n_subsets=14)
print(f"sinogram: {time.time() - t0:.1f} s")

# %% 7. Comparison
def difference(a, b):
    return 100 * float((a - b).abs().sum() / a.abs().sum())

print(f"OSEM 4 × 14: the images differ by {difference(recon_lm, recon_sino):.1f}% voxel by voxel, "
      f"and their totals by {100 * float(recon_sino.sum() / recon_lm.sum() - 1):+.2f}%")

t0 = time.time()
mlem_lm = OSEM(likelihood_lm)(n_iters=40, n_subsets=1)
mlem_sino = OSEM(likelihood_sino)(n_iters=40, n_subsets=1)
print(f"MLEM, 40 iterations ({time.time() - t0:.0f} s): the images differ by {difference(mlem_lm, mlem_sino):.1f}%")

# %% Save the images
from pytomography.io import save_dicom

affine_mr = gate.get_patient_affine_from_nifti(os.path.join(path, 'fdg_pet_phantom_mri.nii.gz'), object_meta)
for name, image in [('list_mode', recon_lm), ('sinogram', recon_sino)]:
    save_dicom(image, OUTPUT / 't_pet_introduction' / name, affine=affine_mr, modality='PT',
               series_description=f'OSEM 4x14, {name}', overwrite=True)
