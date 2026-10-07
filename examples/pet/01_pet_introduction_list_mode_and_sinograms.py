"""PET introduction: list mode and sinograms

Reconstruct the same true coincidences as list mode and as sinograms, with normalisation and attenuation.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_pet_introduction.html
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
OUTPUT = Path(os.environ.get("PYTOMOGRAPHY_OUTPUT", "pytomography_outputs")).expanduser() / "PET/GATE-mMR-Brain"
OUTPUT.mkdir(parents=True, exist_ok=True)

import time
import torch
import matplotlib.pyplot as plt
import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETSinogramPolygonProjMeta
from pytomography.projectors.PET import PETLMSystemMatrix, PETSinogramSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM
from pytomography.io.PET import gate

# %% 1. The scanner
path = DATA / 'PET' / 'GATE-mMR-Brain'
info = gate.get_detector_info(path=os.path.join(path, 'mMR_Geometry.mac'), mean_interaction_depth=9, min_rsector_difference=0)
print(f"{info['NrRings']} rings of {info['NrCrystalsPerRing']} crystals, radius {info['radius']} mm")

# %% 2. Normalisation
weights_file = os.path.join(OUTPUT, 'normalization_weights.pt')
if not os.path.exists(weights_file):
    normalization_paths = [os.path.join(path, f'normalization_scan/mMR_Norm_{i}.root') for i in range(1, 37)]
    normalization_weights = gate.get_normalization_weights_cylinder_calibration(
        normalization_paths, info, cylinder_radius=318, include_randoms=False)  # shell radius in mm
    torch.save(normalization_weights, weights_file)
normalization_weights = torch.load(weights_file)

# %% 3. The true coincidences
ids_file = os.path.join(OUTPUT, 'detector_ids_primary_only.pt')
if not os.path.exists(ids_file):
    paths = [os.path.join(path, f'all_physics/mMR_voxBrain_{i}.root') for i in range(1, 55) if i != 24]
    detector_ids = gate.get_detector_ids_from_root(paths, info, include_randoms=False, include_scatters=False)
    torch.save(detector_ids, ids_file)
detector_ids = torch.load(ids_file)
print(f"{detector_ids.shape[0] / 1e6:.1f} million true coincidences")

# %% 4. Object space and attenuation
object_meta = ObjectMeta(dr=(2, 2, 2), shape=(128, 128, 96))  # mm, voxels
atten_map = gate.get_attenuation_map_nifti(os.path.join(path, 'fdg_pet_phantom_umap.nii.gz'), object_meta)
atten_map = atten_map.to(pytomography.dtype).to(pytomography.device)

# %% 5. List-mode reconstruction
proj_meta_lm = PETLMProjMeta(detector_ids=detector_ids[:, :2], info=info, weights_sensitivity=normalization_weights)
system_matrix_lm = PETLMSystemMatrix(object_meta, proj_meta_lm, N_splits=10, attenuation_map=atten_map)
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
    sinogram_sensitivity=normalization_sinogram,
    attenuation_map=atten_map,
    N_splits=10,
    device='cpu')  # keep the large sinograms in host memory
likelihood_sino = PoissonLogLikelihood(system_matrix_sino, sinogram)

t0 = time.time()
recon_sino = OSEM(likelihood_sino)(n_iters=4, n_subsets=14)
print(f"sinogram: {time.time() - t0:.1f} s")

# %% 7. Comparison
difference = (recon_lm - recon_sino).abs().sum() / recon_lm.abs().sum()
print(f"relative difference between the two reconstructions: {100 * difference.item():.1f}%")
