"""PETSIRD list mode

A Siemens Biograph mMR scan in ETSI's open PETSIRD format, reconstructed without and with scatter correction.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_PETSIRD.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

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
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform, binary_erosion, gaussian_filter, label
import pytomography
from pytomography.io.PET import petsird, shared, gate
from pytomography.io.shared import centred_affine, get_header_value, save_nifti
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
# reconstruction prints its own estimate before it runs (estimate_memory), with what more subsets would need.
# MINUTES is how much of the 60-minute scan is reconstructed: the first 10 minutes (38 million prompts) peak at about
# 15 GB of RAM and 2 GB of GPU memory, all 60 minutes (197 million prompts) at about 20 GB and 5 GB. To use less,
# lower MINUTES first, then RAM_GB.
RAM_GB = 25
pytomography.set_memory_budget(RAM_GB)
MINUTES = 10

# %% 1. The scan in PyTomography's standard form
path = DATA / 'PET' / 'PETSIRD-mMR-NEMA' / 'mMR_NEMA_norm.petsird'
t0 = time.time()
header = petsird.read_header(path)
info = petsird.get_detector_info(header, up='+x', towards_bed='+z', patient_orientation='HFS')
window_ms = (0, MINUTES * 60_000)  # the time blocks that start in the first MINUTES minutes
prompts = petsird.get_detector_ids(path, info=info, header=header, time_interval=window_ms)
delayeds = petsird.get_detector_ids(path, info=info, header=header, delayeds=True, time_interval=window_ms)
weights = petsird.get_sensitivity_weights(header, info=info)  # one per crystal pair, in torch.combinations order
del header  # everything needed from it is now in info, the events and the weights
print(f"{info['NrRings']} rings of {info['NrCrystalsPerRing']} crystals, radius {info['radius']:.1f} mm, energy window "
      f"{info['energy_window_low']:.0f} to {info['energy_window_high']:.0f} keV, crystals turned by "
      f"{info['petsird_alignment_deg']:.3f} degrees to PyTomography's layout")
print(f"first {MINUTES} min: {len(prompts):,} prompts and {len(delayeds):,} delayed coincidences; weights of "
      f"{len(weights):,} crystal pairs; read in {time.time() - t0:.0f} s")

# %% 2. The image grid and the attenuation map
object_meta = ObjectMeta(dr=(2.08626, 2.08626, 2.03125), shape=(200, 200, 127))  # mm, voxels
scanner_LUT = shared.get_scanner_LUT(info)
# Keep the events whose line of response crosses the image, 16 million at a time (the test makes a few arrays per event)
prompts = torch.cat([gate.remove_events_out_of_bounds(prompts[s:s + 2**24], scanner_LUT, object_meta)
                     for s in range(0, len(prompts), 2**24)])
n_events = len(prompts)
print(f"{n_events:,} prompts cross the image")

# The scan's attenuation map, in 1/cm, as x, y, z: towards the patient's left, posterior and superior
folder = DATA / 'PET' / 'PETSIRD-mMR-NEMA'
mu_header = np.array((folder / '20170809_NEMA_MUMAP_UCL.v.hdr').read_text().splitlines())
mu_shape = [get_header_value(mu_header, f'matrix size[{n}]', int) for n in (1, 2, 3)]
mu_dr = [float(get_header_value(mu_header, f'scale factor (mm/pixel) [{n}]')) for n in (1, 2, 3)]
mu_siemens = np.fromfile(folder / '20170809_NEMA_MUMAP_UCL.v', dtype='<f4').reshape(mu_shape[::-1]).transpose(2, 1, 0)

# The grid's voxels in LPS (mm): centred on the scanner, turned by the alignment angle about its axis
a = np.radians(info['petsird_alignment_deg'])
turn = np.eye(4)
turn[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
grid_affine = turn @ centred_affine(object_meta.dr, object_meta.shape)
# For each voxel of the grid, the attenuation map's voxel at the same place
grid_to_mu = np.linalg.inv(centred_affine(mu_dr, mu_shape)) @ grid_affine
mu_map = affine_transform(np.ascontiguousarray(mu_siemens), grid_to_mu, output_shape=object_meta.shape, order=1)  # 1/cm
attenuation_map = torch.tensor(mu_map / 10, dtype=torch.float32, device=pytomography.device)  # 1/mm

# %% 3. The system matrix and the randoms
t0 = time.time()
proj_meta = PETLMProjMeta(prompts, info, scanner_LUT=scanner_LUT, weights_sensitivity=weights)
system_matrix = PETLMSystemMatrix(object_meta, proj_meta, obj2obj_transforms=[GaussianFilter(4.0)],
                                  attenuation_map=attenuation_map, N_splits=4)
del prompts  # the system matrix holds the events now
print(f"system matrix and sensitivity image: {time.time() - t0:.0f} s")

t0 = time.time()
sinogram_delays = shared.listmode_to_sinogram(delayeds, info)
del delayeds
sinogram_delays = shared.smooth_randoms_sinogram(sinogram_delays, info, sigma_r=4, sigma_theta=4, sigma_z=4)
lm_randoms = shared.sinogram_to_listmode(proj_meta.detector_ids, sinogram_delays, info)
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

# Where to look: the phantom (from the attenuation map), the plane of the spheres (the slice with the hottest voxels
# inside the phantom) and the centre of the hottest sphere in it
dx, dy, dz = object_meta.dr
regions, _ = label(mu_map > 0.05)  # 1/cm: the phantom's water (0.093/cm) and walls, not the air around it
body = regions == np.argmax(np.bincount(regions.ravel())[1:]) + 1  # the phantom, the largest region
inner = binary_erosion(body, structure=np.ones((9, 9, 1)))  # 4 voxels in from its edges
smooth = gaussian_filter(recon_without_scatter.cpu().numpy(), 1.0)
score = [np.percentile(smooth[:, :, n][inner[:, :, n]], 99.9) if inner[:, :, n].sum() > 100 else 0
         for n in range(smooth.shape[2])]
k = int(np.argmax(score))
i, j = (int(v) for v in np.unravel_index(np.argmax(np.where(inner[:, :, k], smooth[:, :, k], 0)), smooth.shape[:2]))
# The phantom's extent with a margin, for the figures
xs, ys, zs = np.nonzero(body)
x0, x1 = max(xs.min() - 4, 0), min(xs.max() + 5, body.shape[0])
y0, y1 = max(ys.min() - 4, 0), min(ys.max() + 5, body.shape[1])
z0, z1 = max(zs.min() - 3, 0), min(zs.max() + 4, body.shape[2])
print(f"plane of the spheres: slice {k}; centre of the hottest sphere: voxel ({i}, {j})")

# %% 5. Scatter correction
t0 = time.time()
sinogram_scatter = sss.get_sss_scatter_estimate(
    object_meta=object_meta, proj_meta=proj_meta, pet_image=recon_without_scatter, attenuation_image=attenuation_map,
    system_matrix=system_matrix, proj_data=None,  # None: list mode, whose events are in proj_meta
    image_stepsize=4, attenuation_cutoff=0.004, sinogram_interring_stepsize=4, sinogram_intraring_stepsize=4,
    sinogram_random=sinogram_delays)
lm_scatter = shared.sinogram_to_listmode(proj_meta.detector_ids, sinogram_scatter, info)
seconds_sss = time.time() - t0
print(f"scatter estimate: {seconds_sss:.0f} s")

additive_term = (lm_scatter.cpu() + lm_randoms.cpu()) / lm_sensitivity.cpu()
additive_term[additive_term.isnan()] = 0
likelihood = PoissonLogLikelihood(system_matrix, additive_term=additive_term.to(pytomography.device))
t0 = time.time()
recon_with_scatter = OSEM(likelihood)(n_iters=3, n_subsets=21)
seconds_with = time.time() - t0
print(f"OSEM with scatter correction: {seconds_with:.1f} s")

# The background: the phantom at least 30 mm from the plane of the spheres, 4 voxels in from its walls (where the
# image's edge is blurred) and 5 slices from its ends
far = np.abs(np.arange(body.shape[2]) - k) * dz >= 30
background = binary_erosion(body, structure=np.ones((9, 9, 11))) & far[None, None, :]
# Each voxel's distance from the phantom's axis, through the centre of its cross-section
cx, cy = np.argwhere(background)[:, :2].mean(axis=0)
r = np.hypot((np.arange(body.shape[0])[:, None] - cx) * dx, (np.arange(body.shape[1])[None, :] - cy) * dy)
r = np.broadcast_to(r[:, :, None], body.shape)
rings = np.arange(0, r[background].max() + 10, 10)  # mm


def radial_profile(img):
    # mean of each ring 10 mm wide, relative to the mean of the whole background
    centres, values = [], []
    for lo, hi in zip(rings[:-1], rings[1:]):
        ring = background & (r >= lo) & (r < hi)
        if ring.sum() >= 500:
            centres.append((lo + hi) / 2)
            values.append(img[ring].mean() / img[background].mean())
    return np.array(centres), np.array(values)


images = {'without': recon_without_scatter.cpu().numpy(), 'with': recon_with_scatter.cpu().numpy()}
profiles = {name: radial_profile(img) for name, img in images.items()}
for name, img in images.items():
    hottest = gaussian_filter(img, 1.0)[i, j, k] / img[background].mean()
    p = profiles[name][1]
    print(f"{name} scatter correction: background from the centre to the edge {p[0]:.2f} -> {p[-1]:.2f} of its mean; "
          f"hottest sphere {hottest:.1f} times the background")

# %% 6. Save the images
from pytomography.io import save_dicom, save_nifti

for name, image in [('osem_with_scatter_correction', recon_with_scatter),
                    ('osem_without_scatter_correction', recon_without_scatter)]:
    save_dicom(image, OUTPUT / 't_PETSIRD' / name, affine=grid_affine, modality='PT',
               series_description=name.replace('_', ' '), overwrite=True)
save_nifti(mu_map, OUTPUT / 't_PETSIRD' / 'attenuation_map.nii.gz', affine=grid_affine, units='1/cm')
