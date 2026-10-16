"""Filtered back projection (SPECT)

Filtered back projection of a Lu-177 NEMA phantom scan, with Chang's attenuation correction, next to OSEM.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_fbp_spect.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Lu177-NEMA-SymT2")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Lu177-NEMA-SymT2")

import time
import torch
import pytomography
from pytomography import io
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import FilteredBackProjection, OSEM

# Memory: this tutorial needs about 5 GB of RAM, most of it PyTorch and CUDA themselves. Each reconstruction
# prints its estimate before it runs (estimate_memory). RAM_GB changes nothing here: SPECT reconstructions this size
# never need to split their work.
RAM_GB = 25
pytomography.set_memory_budget(RAM_GB)

# %% 2. The scan
path = DATA / 'SPECT' / 'Lu177-NEMA-SymT2'
file_NM = str(path / 'projection_data.dcm')
files_CT = sorted(str(f) for f in (path / 'CT').glob('*'))
projections, object_meta, proj_meta = io.SPECT.read_dicom(file_NM)
photopeak = projections[0]  # the 208 keV window
scatter = io.SPECT.energy_window_scatter(projections, proj_meta, peak=0, lower=1, upper=2)
photon_energy = proj_meta.energy_windows[0].photon_energy
att_transform = SPECTAttenuationTransform(io.attenuation_map(files_CT, object_meta, energy=photon_energy))
psf_transform = SPECTPSFTransform(io.SPECT.collimator_psf('SY-ME', photon_energy, intrinsic_fwhm=3.8))
system_matrix = SPECTSystemMatrix(obj2obj_transforms=[att_transform, psf_transform], proj2proj_transforms=[],
                                  object_meta=object_meta, proj_meta=proj_meta)
print(f"{proj_meta.num_projections} views of {photopeak.shape[1]} x {photopeak.shape[2]} pixels of {object_meta.dr[0]:.2f} mm; "
      f"{photopeak.sum():.3g} counts, of which TEW estimates {scatter.sum() / photopeak.sum():.0%} are scatter")
print(f"counts in a pixel of one view: {photopeak[photopeak > 0.1 * photopeak.max()].median():.0f} over the phantom, {photopeak.max():.0f} at most")

# %% 3. Filtered back projection
butterworth = lambda f: 1 / torch.sqrt(1 + (f / 0.05) ** 10)  # cut-off 0.05 cycles/mm, order 5
fbp = FilteredBackProjection(photopeak - scatter, system_matrix, filter=butterworth)
print(fbp.estimate_memory())
t0 = time.time()
recon_fbp = fbp()
print(f"FBP: {time.time() - t0:.1f} s")

attenuation_only = SPECTSystemMatrix(obj2obj_transforms=[att_transform], proj2proj_transforms=[],
                                     object_meta=object_meta, proj_meta=proj_meta)
chang = proj_meta.num_projections / attenuation_only.compute_normalization_factor()
recon_fbp_chang = recon_fbp * chang

# %% 4. OSEM for comparison
likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter)
osem = OSEM(likelihood)
print(osem.estimate_memory(n_subsets=8))
t0 = time.time()
recon_osem = osem(n_iters=4, n_subsets=8)
print(f"OSEM 4 x 8: {time.time() - t0:.1f} s")

# %% Save the images
for name, image, description in [('fbp', recon_fbp, 'FBP, Butterworth'), ('fbp_chang', recon_fbp_chang, 'FBP, Butterworth, Chang'),
                                 ('osem', recon_osem, 'OSEM 4x8')]:
    io.save_dicom(image, OUTPUT / 't_fbp_spect' / name, object_meta, reference=file_NM, units='counts',
                  series_description=description, overwrite=True)
