"""Kernel EM with a PET prior

Lu-177 SPECT of the Zubal phantom reconstructed with kernels from a Ga-68 PET image: one from its intensities, one learned by a 3D U-Net, beside MLEM.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_kernel_deep_kernel_spect_recon.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
# %% Kernel EM with a PET prior: Lu-177 SPECT guided by Ga-68 PET
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Lu177-DOTATATE-Zubal")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Lu177-DOTATATE-Zubal")

import time
import numpy as np
import torch
import pytomography
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.transforms.shared import ExternalKEMTransform
from pytomography.transforms.shared.external_kem import (UNet3D, construct_deep_kernel, construct_kernel,
                                                         train_deep_kernel)
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.projectors.shared import KEMSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import KEM, OSEM

# Memory: a kernel holds k = 50 neighbours for each of the 128^3 voxels (float32 weights and int64 indices, 1.3 GB). It
# stays in RAM and goes to the GPU a chunk at a time; the tutorial peaks at about 9 GB of RAM (measured on Windows).
# Training the U-Net takes about 1.4 GB of GPU memory with chunk_size=8192 below: halve chunk_size to halve it.
# Nothing here grows with RAM_GB.
RAM_GB = 25
pytomography.set_memory_budget(RAM_GB)

# %% 1. The data
path = DATA / 'SPECT' / 'Lu177-DOTATATE-Zubal'
pet_prior = np.load(path / 'pet_prior.npy')
attenuation_map = torch.as_tensor(np.load(path / 'attenuation_map.npy'), device=pytomography.device)
projections = torch.as_tensor(np.load(path / 'projections.npy'), device=pytomography.device)

object_meta = SPECTObjectMeta(dr=(4.0, 4.0, 4.0), shape=(128, 128, 128))
proj_meta = SPECTProjMeta(projection_shape=(128, 128), dr=(4.0, 4.0),
                          angles=np.linspace(0, 360, 120, endpoint=False), radii=np.full(120, 300.0))
psf = SPECTPSFTransform(psf_meta=SPECTPSFMeta(sigma_fit_params=(0.02, 1.0)))  # sigma = 0.02 d + 1.0 mm
system_matrix = SPECTSystemMatrix([SPECTAttenuationTransform(attenuation_map), psf], [], object_meta, proj_meta)
print(f"{float(projections.sum()) / 1e6:.1f} million counts in {tuple(projections.shape)} projections (views, r, z)")

# %% 2. MLEM
t0 = time.time()
recon_mlem = OSEM(PoissonLogLikelihood(system_matrix, projections))(n_iters=50, n_subsets=1)
print(f"MLEM, 50 iterations: {time.time() - t0:.0f} s")

# %% 3. A kernel from the PET intensities
t0 = time.time()
weights, indices = construct_kernel(pet_prior, k=50, sigmaf=1.0, sigmac=1.0, patch=1, local_window=9, mode='local',
                                    device=pytomography.device, chunk_size=4096)
print(f"kernel: weights {weights.shape} {weights.dtype}, indices {indices.shape} {indices.dtype}, "
      f"{time.time() - t0:.0f} s")

t0 = time.time()
kem = KEM(PoissonLogLikelihood(KEMSystemMatrix(system_matrix, ExternalKEMTransform(weights, indices)), projections))
recon_kem = kem(n_iters=50, n_subsets=1)
print(f"KEM, 50 iterations: {time.time() - t0:.0f} s")
del kem, weights, indices  # 1.3 GB of RAM

# %% 4. A deep-learned kernel
deep_params = dict(k=50, local_window=9, include_self=False)

torch.manual_seed(2026)
model = UNet3D(base_channels=16, feature_channels=16)
t0 = time.time()
history = train_deep_kernel(model, pet_prior, recon_mlem.cpu().numpy(), epochs=10, patches_per_epoch=300,
                            patch_size=32, chunk_size=8192, learning_rate=1e-3, device=pytomography.device,
                            verbose=False, **deep_params)
print(f"training: {time.time() - t0:.0f} s, loss {history[0]:.2e} in the first epoch, {history[-1]:.2e} in the last")

t0 = time.time()
deep_weights, deep_indices = construct_deep_kernel(model, pet_prior, chunk_size=512, device=pytomography.device,
                                                   **deep_params)
deep_kem = KEM(PoissonLogLikelihood(KEMSystemMatrix(system_matrix, ExternalKEMTransform(deep_weights, deep_indices)),
                                    projections))
recon_deep_kem = deep_kem(n_iters=50, n_subsets=1)
print(f"deep kernel and KEM, 50 iterations: {time.time() - t0:.0f} s")
del deep_kem, deep_weights, deep_indices

# %% 5. The three reconstructions
np.savez_compressed(OUTPUT / 'kem_reconstructions.npz', mlem=recon_mlem.cpu().numpy(), kem=recon_kem.cpu().numpy(),
                    deep_kem=recon_deep_kem.cpu().numpy())
print(f"saved to {OUTPUT / 'kem_reconstructions.npz'}")
