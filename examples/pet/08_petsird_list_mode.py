"""PETSIRD list mode

A Siemens Biograph mMR scan in ETSI's open PETSIRD format, reconstructed without and with scatter correction.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_PETSIRD.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
# %% PETSIRD list mode: a Siemens Biograph mMR scan
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("PET/PETSIRD-mMR-NEMA")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("PET/PETSIRD-mMR-NEMA")

import time
import numpy as np
import torch
import pytomography
from pytomography import io
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.transforms.shared import GaussianFilter
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM
from pytomography.utils import sss

# Memory: the scan's header holds 2.9 GB of normalisation tables, and the weights of the scanner's 520 million crystal
# pairs take 2.1 GB. RAM_GB caps what the steps that go through every crystal pair (the sensitivity image, the scatter
# estimate) use at once: with less, they work in smaller blocks, which is slower and gives the same images. Each
# reconstruction prints its own estimate before it runs (estimate_memory), with what more subsets would need, and so
# does the scatter estimate. MINUTES is how much of the 60-minute scan is reconstructed: the first 10 minutes (38
# million prompts) peak at about 19 GB of RAM and 2.6 GB of GPU memory (measured on Windows), and still about 19 GB
# with RAM_GB = 12, so lowering RAM_GB saves little; all 60 minutes (197 million prompts) take more. To use less,
# lower MINUTES.
RAM_GB = 25
pytomography.set_memory_budget(RAM_GB)
MINUTES = 10

# %% 1. The scan in PyTomography's standard form
path = DATA / 'PET' / 'PETSIRD-mMR-NEMA' / 'mMR_NEMA_norm.petsird'
t0 = time.time()
header = io.PET.petsird.read_header(path)
window_s = (0, MINUTES * 60)  # the time blocks that start in the first MINUTES minutes
# The image grid: the scanner's own voxel size, 200 x 200 x 127 voxels (mm), centred on the scanner (section 2)
prompts, object_meta, proj_meta = io.PET.read_petsird(
    path, object_meta=ObjectMeta(dr=(2.08626, 2.08626, 2.03125), shape=(200, 200, 127)),
    up='+x', towards_bed='+z', patient_position='HFS', time_interval_s=window_s, header=header)
delayeds, _, _ = io.PET.read_petsird(path, events='delayeds', up='+x', towards_bed='+z', patient_position='HFS',
                                     time_interval_s=window_s, header=header)
del header  # everything needed from it is now in the events and proj_meta
info = proj_meta.info
weights = proj_meta.weights_sensitivity  # one per crystal pair, in torch.combinations order
print(f"{info['NrRings']} rings of {info['NrCrystalsPerRing']} crystals, radius {info['radius']:.1f} mm, energy window "
      f"{info['energy_window_low_keV']:.0f} to {info['energy_window_high_keV']:.0f} keV")
print(f"first {MINUTES} min: {len(prompts):,} prompts and {len(delayeds):,} delayed coincidences; weights of "
      f"{len(weights):,} crystal pairs; read in {time.time() - t0:.0f} s")

# %% 2. The image grid and the attenuation map
# Keep the events whose line of response crosses the image
prompts = io.PET.remove_events_outside(prompts, info, object_meta)
n_events = len(prompts)
print(f"{n_events:,} prompts cross the image")

# The scan's attenuation map (Siemens Interfile, 1/cm) on the grid, in 1/mm
folder = DATA / 'PET' / 'PETSIRD-mMR-NEMA'
attenuation_map = io.attenuation_map(folder / '20170809_NEMA_MUMAP_UCL.v.hdr', object_meta)

# %% 3. The system matrix and the randoms
t0 = time.time()
proj_meta = PETLMProjMeta(prompts, info, weights_sensitivity=weights)  # the prompts that cross the image
system_matrix = PETLMSystemMatrix(object_meta, proj_meta, obj2obj_transforms=[GaussianFilter(4.0)],
                                  attenuation_map=attenuation_map, N_splits=4)
del prompts  # the system matrix holds the events now
print(f"system matrix and sensitivity image: {time.time() - t0:.0f} s")

t0 = time.time()
sinogram_delays = io.PET.listmode_to_sinogram(delayeds, info)
del delayeds
sinogram_delays = io.PET.smooth_randoms_sinogram(sinogram_delays, info, sigma_r=4, sigma_theta=4, sigma_z=4)
lm_randoms = io.PET.sinogram_to_listmode(proj_meta.detector_ids, sinogram_delays, info)
lm_sensitivity = system_matrix._compute_sensitivity_projection(all_ids=False)
additive_term = lm_randoms.cpu() / lm_sensitivity.cpu()
additive_term[additive_term.isnan()] = 0
print(f"randoms: {time.time() - t0:.0f} s")

# %% 4. Reconstruction without scatter correction
likelihood = PoissonLogLikelihood(system_matrix, additive_term=additive_term.to(pytomography.device))
print(system_matrix.estimate_memory(n_subsets=21, held={'additive term': likelihood.additive_term}))
t0 = time.time()
recon_without_scatter = OSEM(likelihood)(n_iters=3, n_subsets=21)
seconds_without = time.time() - t0
print(f"OSEM without scatter correction: {seconds_without:.1f} s")

# The phantom and its largest sphere, through whose centre the figures look
regions = io.load_mask(folder / 'petsird_mmr_nema_rois.seg.nrrd')
sphere = regions.on(object_meta, label='sphere_37mm', threshold=0.5).cpu().numpy()
i, j, k = np.argwhere(sphere).mean(axis=0).round().astype(int)
print(f"centre of the 37 mm sphere: voxel ({i}, {j}, {k})")

# %% 5. Scatter correction
# The estimate's peak RAM and GPU memory, before it runs
print(sss.estimate_sss_memory(object_meta, proj_meta, system_matrix, sinogram_random=sinogram_delays,
                              sinogram_interring_stepsize=4, sinogram_intraring_stepsize=4))
t0 = time.time()
sinogram_scatter = sss.get_sss_scatter_estimate(
    object_meta=object_meta, proj_meta=proj_meta, pet_image=recon_without_scatter, attenuation_image=attenuation_map,
    system_matrix=system_matrix, proj_data=None,  # None: list mode, whose events are in proj_meta
    image_stepsize=4, attenuation_cutoff=0.004, sinogram_interring_stepsize=4, sinogram_intraring_stepsize=4,
    sinogram_random=sinogram_delays)
lm_scatter = io.PET.sinogram_to_listmode(proj_meta.detector_ids, sinogram_scatter, info)
seconds_sss = time.time() - t0
print(f"scatter estimate: {seconds_sss:.0f} s")

additive_term = (lm_scatter.cpu() + lm_randoms.cpu()) / lm_sensitivity.cpu()
additive_term[additive_term.isnan()] = 0
likelihood = PoissonLogLikelihood(system_matrix, additive_term=additive_term.to(pytomography.device))
t0 = time.time()
recon_with_scatter = OSEM(likelihood)(n_iters=3, n_subsets=21)
seconds_with = time.time() - t0
print(f"OSEM with scatter correction: {seconds_with:.1f} s")

# The background: the phantom at least 30 mm from the plane of the spheres, 4 voxels in from its walls and 5 slices
# from its ends (where the image's edges are blurred); and the same in rings 10 mm wide around the phantom's axis, ring
# n from 10 n to 10 (n + 1) mm, here by the middle of each ring
background = regions.on(object_meta, label='background', threshold=0.5).cpu().numpy()
rings = {10 * int(name.removeprefix('ring_')) + 5: regions.on(object_meta, label=name, threshold=0.5).cpu().numpy()
         for name in regions.labels if name.startswith('ring_')}

def radial_profile(img):
    # mean of each ring, relative to the mean of the whole background
    return np.array(list(rings)), np.array([img[ring].mean() / img[background].mean() for ring in rings.values()])

images = {'without': recon_without_scatter.cpu().numpy(), 'with': recon_with_scatter.cpu().numpy()}
profiles = {name: radial_profile(img) for name, img in images.items()}
for name, img in images.items():
    contrast = img[sphere].mean() / img[background].mean()
    p = profiles[name][1]
    print(f"{name} scatter correction: background from the centre to the edge {p[0]:.2f} -> {p[-1]:.2f} of its mean; "
          f"37 mm sphere {contrast:.1f} times the background")

# %% 6. Save the images
for name, image in [('osem_with_scatter_correction', recon_with_scatter),
                    ('osem_without_scatter_correction', recon_without_scatter)]:
    io.save_dicom(image, OUTPUT / 't_PETSIRD' / name, object_meta, series_description=name.replace('_', ' '),
                  overwrite=True)
io.save_nifti(attenuation_map, OUTPUT / 't_PETSIRD' / 'attenuation_map.nii.gz', object_meta, units='1/mm')
