"""CGLS on a patient

Conjugate gradients for least squares on a two-bed Lu-177 PSMA patient, beside OSEM: the exact transpose it needs, when to stop, a diagonal preconditioner, and its negative values.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_cgls.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
# %% CGLS: least-squares reconstruction of a Lu-177 patient
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Lu177-PSMA-GEDisc")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Lu177-PSMA-GEDisc")

import os
import time
import torch
import pytomography
from pytomography import io
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM

# Memory: each bed is 128 x 128 x 128 voxels and 60 views, so a reconstruction needs about 0.5 GB of GPU memory and the
# tutorial peaks at about 6 GB of RAM (measured on Windows); nothing here grows with RAM_GB. CGLS holds four images and
# two sets of projections per bed while it runs.
RAM_GB = 25
pytomography.set_memory_budget(RAM_GB)

# %% 1. The patient
path = DATA / 'SPECT' / 'Lu177-PSMA-GEDisc'
files_NM = [str(path / 'bed1_projections.dcm'), str(path / 'bed2_projections.dcm')]
files_CT = [str(path / 'CT' / f) for f in os.listdir(path / 'CT')]

beds = []
for projections, object_meta, proj_meta in io.SPECT.read_dicom_beds(files_NM):  # the inferior bed first
    energy = proj_meta.energy_windows[1].photon_energy
    attenuation_map = io.attenuation_map(files_CT, object_meta, energy=energy)
    psf = SPECTPSFTransform(io.SPECT.collimator_psf('GI-MEGP', energy))
    system_matrix = SPECTSystemMatrix([SPECTAttenuationTransform(attenuation_map), psf], [], object_meta, proj_meta)
    scatter = io.SPECT.energy_window_scatter(projections, proj_meta, 1, 3, 2)
    beds.append(dict(system_matrix=system_matrix, photopeak=projections[1], scatter=scatter, object_meta=object_meta))
print(f"{len(beds)} beds of {tuple(beds[0]['photopeak'].shape)} projections (views, r, z)")

# %% 2. CGLS needs an exact transpose
H = beds[0]['system_matrix']
g = torch.Generator().manual_seed(0)
x = torch.rand(H.object_meta.shape, generator=g).to(pytomography.device)
y = torch.rand(beds[0]['photopeak'].shape, generator=g).to(pytomography.device)
mismatch = float((H.forward(x) * y).sum() / (x * H.backward(y)).sum() - 1)
print(f"<Hx, y> / <x, H^T y> - 1 = {mismatch:.1e}")

# %% 3. OSEM, for comparison
def reconstruct(make, run):
    """Reconstruct both beds with make(bed) and run(algorithm), and stitch them; also return the algorithms."""
    algorithms = [make(bed) for bed in beds]
    t0 = time.time()
    recons = [run(algorithm) for algorithm in algorithms]
    seconds = time.time() - t0
    stitched, object_meta = io.SPECT.stitch_beds(recons=torch.stack(recons[::-1]),
                                                 object_metas=[bed['object_meta'] for bed in beds[::-1]])
    return stitched, object_meta, algorithms, seconds

osem, object_meta_stitched, _, seconds = reconstruct(
    lambda bed: OSEM(PoissonLogLikelihood(bed['system_matrix'], bed['photopeak'], additive_term=bed['scatter'])),
    lambda algorithm: algorithm(n_iters=4, n_subsets=8))
print(f"OSEM, 4 iterations of 8 subsets: {seconds:.1f} s for both beds")

# %% 4. CGLS, and when to stop
cgls, residuals = {}, {}

# %% A diagonal preconditioner
pcgls = {}

# %% 5. Bias and noise
rois = ['liver', 'kidney_left', 'kidney_right', 'spleen']

# %% 6. Negative values
body = io.attenuation_map(files_CT, object_meta_stitched, energy=208.0).to(pytomography.device) > 0.005
chosen = pcgls[10]  # preconditioned CGLS, 10 iterations: about CGLS's 40
for name, image in [('OSEM 4 x 8', osem), ('CGLS 40', cgls[40]), ('Preconditioned CGLS 10', chosen)]:
    print(f"{name:24s} negative voxels in the body: {100 * float((image[body] < 0).float().mean()):4.1f}%   "
          f"total: {float(image.sum()):9.0f} counts")

# %% 7. OSEM and CGLS side by side
print(f"{'':26s}" + ''.join(f'{roi:>14s}' for roi in rois) + f"{'liver noise':>14s}")

# %% Save the images
saved = io.save_dicom(chosen.clamp(min=0), OUTPUT / 't_cgls' / 'pcgls_10', object_meta_stitched,
                      reference=files_NM[1], series_description='Preconditioned CGLS 10, two beds stitched',
                      overwrite=True)
