"""Filtered back projection: SPECT and CT

One FilteredBackProjection for a SPECT phantom scan and a helical chest CT, with what the two share and what differs, next to OSEM and the scanner's image.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_fbp.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Lu177-NEMA-SymT2")
datasets.fetch("CT/ldct-c145")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Lu177-NEMA-SymT2")

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import ndimage
import pytomography
from pytomography.io.SPECT import dicom
from pytomography.io.CT import dicom_ct_pd
from pytomography.io.shared import open_multifile, align_images_affine
from pytomography.metadata import ObjectMeta
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.projectors.CT import CTGen3SystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import FilteredBackProjection, OSEM
from pytomography.utils import get_fbp_filter

# Memory: this tutorial needs about 13 GB of RAM, nearly all of it for the CT: its 2 GB of projections, the images it
# keeps, and the GPU libraries. RAM_GB caps what PyTomography's reconstructions use for their own work: with less,
# the CT reconstruction rebins and filters fewer views at a time, which is a little slower and gives the same image.
# Each reconstruction prints its estimate before it runs (estimate_memory), and section 3 shows plan_memory, which
# picks the settings that fit the memory you give it. The SPECT part alone needs about 3 GB.
RAM_GB = 25
pytomography.set_memory_budget(RAM_GB)

# %% 2. SPECT
path = DATA / 'SPECT' / 'Lu177-NEMA-SymT2'
file_NM = str(path / 'projection_data.dcm')
files_CT = sorted(str(f) for f in (path / 'CT').glob('*'))
object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=0)
photopeak = dicom.get_projections(file_NM, index_peak=0)
scatter = dicom.get_energy_window_scatter_estimate(file_NM, index_peak=0, index_lower=1, index_upper=2)
att_transform = SPECTAttenuationTransform(filepath=files_CT)
psf_transform = SPECTPSFTransform(dicom.get_psfmeta_from_scanner_params('SY-ME', 208, intrinsic_resolution=0.38))
system_matrix = SPECTSystemMatrix(obj2obj_transforms=[att_transform, psf_transform], proj2proj_transforms=[],
                                  object_meta=object_meta, proj_meta=proj_meta)
print(f"{proj_meta.num_projections} views of {photopeak.shape[1]} x {photopeak.shape[2]} pixels of {10 * object_meta.dr[0]:.2f} mm; "
      f"{photopeak.sum():.3g} counts, of which TEW estimates {scatter.sum() / photopeak.sum():.0%} are scatter")
print(f"counts in a pixel of one view: {photopeak[photopeak > 0.1 * photopeak.max()].median():.0f} over the phantom, {photopeak.max():.0f} at most")

butterworth = lambda f: 1 / torch.sqrt(1 + (f / 0.05) ** 10)  # cut-off 0.05 cycles/mm, order 5
fbp_spect = FilteredBackProjection(photopeak - scatter, system_matrix, filter=butterworth)
print(fbp_spect.estimate_memory())
t0 = time.time()
recon_fbp = fbp_spect()
print(f"SPECT FBP: {time.time() - t0:.1f} s")

attenuation_only = SPECTSystemMatrix(obj2obj_transforms=[att_transform], proj2proj_transforms=[],
                                     object_meta=object_meta, proj_meta=proj_meta)
chang = proj_meta.num_projections / attenuation_only.compute_normalization_factor()
recon_fbp_chang = recon_fbp * chang

likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter)
osem = OSEM(likelihood)
print(osem.estimate_memory(n_subsets=8))
t0 = time.time()
recon_osem = osem(n_iters=4, n_subsets=8)
print(f"OSEM 4 x 8: {time.time() - t0:.1f} s")

# %% 3. CT
folder = DATA / 'CT' / 'ldct-c145'
t0 = time.time()
ct_proj, ct_proj_meta = dicom_ct_pd.get_projections_and_metadata_gen3(folder / 'full_dose_projections')
mu_water = ct_proj_meta.water_attenuation  # mm^-1
print(f"{ct_proj.shape[0]} views of {ct_proj.shape[1]} x {ct_proj.shape[2]} detector elements, read in {time.time() - t0:.0f} s")
behind = ct_proj_meta.photon_counts[::100, :, None] * torch.exp(-ct_proj[::100])  # every 100th view
print(f"photons reaching a detector element, median over the rays through the patient: {behind[ct_proj[::100] > 1].median():.0f}")
ct_object_meta = ObjectMeta(dr=(0.662109, 0.662109, 1.0), shape=(512, 512, 316))  # mm, voxels
ct_system_matrix = CTGen3SystemMatrix(ct_object_meta, ct_proj_meta)

fbp_ct = FilteredBackProjection(ct_proj, ct_system_matrix, filter='hann', slice_thickness=1.25)
print(fbp_ct.estimate_memory())
print(fbp_ct.plan_memory(ram_gb=10, gpu_gb=4))

t0 = time.time()
hu = (1000 * (fbp_ct() / mu_water - 1)).cpu().numpy()
print(f"CT FBP, Hann window: {time.time() - t0:.0f} s")
hu_ramlak = (1000 * (FilteredBackProjection(ct_proj, ct_system_matrix, filter='ram-lak', slice_thickness=1.25)() / mu_water - 1)).cpu().numpy()

scanner_files = sorted((folder / 'full_dose_images').glob('*.dcm'))
scanner, scanner_meta = open_multifile(scanner_files, return_object_meta=True)
scanner_affine = scanner_meta.affine_matrix.copy()
scanner_affine[:2, 3] += 0.5 * np.array(scanner_meta.dr[:2])  # GE: the corner of the first pixel, not its centre
ct_affine = ct_proj_meta.get_patient_affine(ct_object_meta).numpy()
scanner_hu = align_images_affine(hu, scanner.cpu().numpy(), ct_affine, scanner_affine, cval=-3000)  # -3000: outside the scanner's images

valid = scanner_hu > -1500  # inside the scanner's field of view
blur = lambda a: ndimage.gaussian_filter(np.where(valid, a, -1000.0), 2 / ct_object_meta.dr[0])  # 2 mm, in plane and along z
body = valid & (blur(scanner_hu) > -700)
liver = (slice(100, 140), slice(200, 240), slice(280, 296))  # 26 x 26 x 16 mm of liver
for name, image in ct_images[1:]:
    rms = np.sqrt(((blur(image) - blur(scanner_hu))[body] ** 2).mean())
    print(f"{name:13s} RMS difference from the scanner's image after a 2 mm blur: {rms:4.1f} HU")
for name, image in ct_images:
    print(f"{name:25s} liver: mean {image[liver].mean():5.1f} HU, standard deviation {image[liver].std():5.1f} HU")

print(ct_system_matrix.estimate_memory(n_subsets=40, held={'projections': ct_proj}))

# %% Save the images
from pytomography.io import save_dicom

for name, image in [('fbp', recon_fbp), ('fbp_chang', recon_fbp_chang), ('osem', recon_osem)]:
    save_dicom(image, OUTPUT / 't_fbp' / name, object_meta, reference=file_NM,
               series_description={'fbp': 'FBP, Butterworth', 'fbp_chang': 'FBP, Butterworth, Chang', 'osem': 'OSEM 4x8'}[name],
               overwrite=True)
saved = save_dicom(hu, OUTPUT / 't_fbp' / 'ct_fbp', affine=ct_affine, modality='CT', reference=scanner_files[0],
                   series_description='FBP, Hann, 1.25 mm', overwrite=True)
