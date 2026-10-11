"""Reconstruction algorithms on measured data

OSEM, OSMAPOSL, BSREM and KEM on a measured NEMA phantom, compared by sphere contrast, background noise and data fit at every iteration.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_algorithms_measured.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Lu177-NEMA-SymT2")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Lu177-NEMA-SymT2")

import time
import numpy as np
import torch
import pydicom
import matplotlib.pyplot as plt
from scipy import ndimage, optimize
import pytomography
from pytomography.io.SPECT import dicom
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.transforms.shared import GaussianFilter, KEMTransform
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.projectors.shared import KEMSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM, OSMAPOSL, BSREM, KEM
from pytomography.priors import RelativeDifferencePrior, TopNAnatomyNeighbourWeight
from pytomography.callbacks import Callback

# Memory: this tutorial peaks at about 13 GB of RAM, for a few seconds while KEM builds its kernel: ranking the 125
# neighbour weights of every voxel on the CPU takes about 7 GB more. The other reconstructions print their estimates
# before they run (estimate_memory), about 3.5 GB each, most of it PyTorch and CUDA themselves; on a machine with less
# than 16 GB, leave KEM out. RAM_GB changes nothing here: SPECT reconstructions this size never need to split their
# work. Once built, KEM's kernel stays on the GPU (1 GB); kernel_on_gpu=False keeps it in RAM, three times slower.
RAM_GB = 25
pytomography.set_memory_budget(RAM_GB)

# %% 1. The scan and the system matrix
path = DATA / 'SPECT' / 'Lu177-NEMA-SymT2'
file_NM = str(path / 'projection_data.dcm')
files_CT = sorted(str(f) for f in (path / 'CT').glob('*'))
object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=0)
photopeak = dicom.get_projections(file_NM, index_peak=0)
scatter = dicom.get_energy_window_scatter_estimate(file_NM, index_peak=0, index_lower=1, index_upper=2)
att_transform = SPECTAttenuationTransform(filepath=files_CT)
att_transform.configure(object_meta, proj_meta)
attenuation_map = att_transform.attenuation_map
psf_transform = SPECTPSFTransform(dicom.get_psfmeta_from_scanner_params('SY-ME', 208, intrinsic_resolution=0.38))
system_matrix = SPECTSystemMatrix(obj2obj_transforms=[att_transform, psf_transform], proj2proj_transforms=[],
                                  object_meta=object_meta, proj_meta=proj_meta)
likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter)

scanner = pydicom.dcmread(path / 'scanner_recon.dcm')
recon_scanner = torch.tensor(np.transpose(scanner.pixel_array, (2, 1, 0)) / proj_meta.num_projections,
                             dtype=torch.float32, device=pytomography.device)
print(f"{photopeak.sum():.3g} counts in the photopeak; the scanner's reconstruction: {scanner.ConvolutionKernel[0]}")

# %% 2. Where the spheres are
dx = 10 * object_meta.dr[0]  # voxel size, mm
mu = attenuation_map.cpu().numpy()
mu_water = float(np.median(mu[mu > 0.05]))
# the phantom: denser than 80% of water, without the thin bed (opening), holes filled slice by slice
dense = ndimage.binary_opening(mu > 0.8 * mu_water, structure=np.ones((5, 5, 1)))
pieces, _ = ndimage.label(dense)
phantom = pieces == np.argmax(np.bincount(pieces.ravel())[1:]) + 1  # the largest piece
phantom = np.stack([ndimage.binary_fill_holes(phantom[:, :, k]) for k in range(phantom.shape[2])], axis=-1)
lung = ndimage.binary_erosion(phantom, structure=np.ones((7, 7, 1))) & (mu < 0.6 * mu_water)
lung_axis = np.array(ndimage.center_of_mass(lung))[:2] * dx  # mm on the voxel grid

diameters = [37, 28, 22, 17, 13, 10]  # mm, in order around the ring
ring_radius = 57.2  # mm

def sphere_box(centre, radius, n=5):
    # The fraction of each voxel inside a sphere (n x n x n points per voxel), in the box of voxels around it
    lo = np.maximum(np.floor((centre - radius) / dx - 1).astype(int), 0)
    hi = np.minimum(np.ceil((centre + radius) / dx + 2).astype(int), object_meta.shape)
    points = (np.arange(n) + 0.5) / n - 0.5
    axes = [((np.arange(l, h)[:, None] + points) * dx).ravel() - c for l, h, c in zip(lo, hi, centre)]
    X, Y, Z = np.meshgrid(*axes, indexing='ij')
    inside = X ** 2 + Y ** 2 + Z ** 2 <= radius ** 2
    fraction = inside.reshape(hi[0] - lo[0], n, hi[1] - lo[1], n, hi[2] - lo[2], n).mean(axis=(1, 3, 5))
    return tuple(slice(l, h) for l, h in zip(lo, hi)), fraction

def sphere_centres(cx, cy, cz, angle, order):
    a = np.deg2rad(angle) + order * np.arange(6) * np.pi / 3
    return np.stack([cx + ring_radius * np.cos(a), cy + ring_radius * np.sin(a), np.full(6, cz)], axis=1)

def counts_in_sphere(centre, d, image):
    box, fraction = sphere_box(np.asarray(centre), d / 2)
    return (fraction * image[box]).sum()

image = recon_scanner.cpu().numpy()
smooth = ndimage.gaussian_filter(image, 2)  # the fits use a smoothed copy, which noise pulls less
hottest = np.array(np.unravel_index(np.argmax(smooth), image.shape)) * dx  # the 37 mm sphere
start = [*lung_axis, hottest[2], np.degrees(np.arctan2(hottest[1] - lung_axis[1], hottest[0] - lung_axis[0]))]
in_ring = lambda p, order: sum(counts_in_sphere(c, d, smooth) for c, d in zip(sphere_centres(*p, order), diameters))
fits = {order: optimize.minimize(lambda p: -in_ring(p, order), start, method='Nelder-Mead', options={'xatol': 0.05, 'fatol': 1e-3})
        for order in (1, -1)}  # both directions round the ring
order = min(fits, key=lambda o: fits[o].fun)
ring = fits[order].x
print(f"ring centre {ring[0] - lung_axis[0]:+.1f} mm, {ring[1] - lung_axis[1]:+.1f} mm from the lung insert's axis in the CT; "
      f"sphere plane at slice {ring[2] / dx:.2f}")
centres = sphere_centres(*ring, order)
for i, d in enumerate(diameters[:-1]):  # each sphere on its own, but the 10 mm one
    on_ring = centres[i].copy()
    fit = optimize.minimize(lambda c: -counts_in_sphere(c, d, smooth), on_ring, method='Nelder-Mead',
                            bounds=[(c - 10, c + 10) for c in on_ring], options={'xatol': 0.05, 'fatol': 1e-4})
    centres[i] = fit.x
    print(f"{d} mm sphere: {np.linalg.norm(fit.x - on_ring):4.1f} mm from its place on the ring")

spheres = {}
for centre, d in zip(centres, diameters):
    box, fraction = sphere_box(centre, d / 2)
    mask = np.zeros(object_meta.shape, bool)
    mask[box] = fraction > 0.5
    spheres[f'{d} mm'] = mask

ball = lambda r: np.sum((np.indices((2 * r + 1,) * 3) - r) ** 2, axis=0) <= r ** 2
any_sphere = np.any(list(spheres.values()), axis=0)
background = (ndimage.binary_erosion(phantom, structure=ball(3)) & ~ndimage.binary_dilation(lung, structure=ball(3))
              & ~ndimage.binary_dilation(any_sphere, structure=ball(4)))
for name, mask in spheres.items():
    print(f"sphere {name:6s} {int(mask.sum()):4d} voxels")
print(f"background   {int(background.sum())} voxels ({background.sum() * (dx / 10) ** 3:.0f} mL)")
spheres = {name: torch.tensor(mask, device=pytomography.device) for name, mask in spheres.items()}
background = torch.tensor(background, device=pytomography.device)

# %% 3. A callback that measures every iteration
def contrast_and_noise(image):
    b = image[background]
    return [float(image[mask].mean() / b.mean()) for mask in spheres.values()], float(b.std() / b.mean())

def log_likelihood(image):
    expected = system_matrix.forward(image) + scatter
    return float((torch.xlogy(photopeak, expected) - expected).sum())

class PhantomStatistics(Callback):
    def __init__(self, n_subsets, post_filter=None):
        self.n_subsets, self.post_filter, self.previous = n_subsets, post_filter, None
        self.contrast, self.noise, self.log_likelihood, self.change = [], [], [], []
        self.contrast_filtered, self.noise_filtered = [], []
    def run(self, object, n_iter, n_subset):
        if n_subset == self.n_subsets - 1:  # the last subset: the end of an iteration
            contrast, noise = contrast_and_noise(object)
            self.contrast.append(contrast)
            self.noise.append(noise)
            self.log_likelihood.append(log_likelihood(object))
            if self.previous is not None:
                self.change.append(float((object - self.previous).norm() / object.norm()))
            self.previous = object.clone()
            if self.post_filter is not None:
                contrast, noise = contrast_and_noise(self.post_filter(object))
                self.contrast_filtered.append(contrast)
                self.noise_filtered.append(noise)
        return object

# %% 4. Reconstruct
n_iters, n_subsets = 40, 8
results, times = {}, {}
post_filter = GaussianFilter(1.0)  # FWHM in cm, the unit of SPECT voxels
post_filter.configure(object_meta, proj_meta)

print(OSEM(likelihood).estimate_memory(n_subsets=n_subsets))
t0 = time.time()
results['OSEM'] = PhantomStatistics(n_subsets, post_filter)
recon_osem = OSEM(likelihood)(n_iters=n_iters, n_subsets=n_subsets, callback=results['OSEM'])
times['OSEM'] = time.time() - t0
recon_osem_filtered = post_filter(recon_osem)

t0 = time.time()
results['OSMAPOSL'] = PhantomStatistics(n_subsets)
prior = RelativeDifferencePrior(beta=0.3, gamma=2, weight=TopNAnatomyNeighbourWeight(attenuation_map, N_neighbours=8))
recon_osmaposl = OSMAPOSL(likelihood, prior=prior)(n_iters=n_iters, n_subsets=n_subsets, callback=results['OSMAPOSL'])
times['OSMAPOSL'] = time.time() - t0

betas = [0.01, 0.03, 0.1, 0.3]
recon_bsrem = {}
for beta in betas:
    t0 = time.time()
    results[f'BSREM {beta}'] = PhantomStatistics(n_subsets)
    bsrem = BSREM(likelihood, prior=RelativeDifferencePrior(beta=beta, gamma=2), relaxation_sequence=lambda n: 1 / (n / 50 + 1))
    recon_bsrem[beta] = bsrem(n_iters=n_iters, n_subsets=n_subsets, callback=results[f'BSREM {beta}'])
    times[f'BSREM {beta}'] = time.time() - t0

t0 = time.time()
kem_transform = KEMTransform(support_objects=[attenuation_map], support_kernels_params=[[0.005]],
                             distance_kernel_params=[0.4], top_N=40, kernel_on_gpu=True)
likelihood_kem = PoissonLogLikelihood(KEMSystemMatrix(system_matrix, kem_transform), photopeak, scatter)
results['KEM'] = PhantomStatistics(n_subsets)
recon_kem = KEM(likelihood_kem)(n_iters=n_iters, n_subsets=n_subsets, callback=results['KEM'])
times['KEM'] = time.time() - t0
for name, t in times.items():
    print(f"{name:12s} {t:5.1f} s")

# %% 6. Contrast against noise
final = {name: (r.contrast[-1], r.noise[-1]) for name, r in results.items()}
final['OSEM + 10 mm filter'] = (results['OSEM'].contrast_filtered[-1], results['OSEM'].noise_filtered[-1])
final["scanner"] = (scanner_contrast, scanner_noise)
print(f"after {n_iters} iterations     noise   " + "  ".join(f"{name:>6s}" for name in spheres))
for name, (contrast, noise) in final.items():
    print(f"{name:22s} {100 * noise:5.1f}%  " + "  ".join(f"{c:6.2f}" for c in contrast))

def at_noise(noise, contrast, level):
    # Each sphere's contrast where a curve crosses the noise `level`, by linear interpolation (NaN if it never does)
    noise, contrast = np.asarray(noise), np.asarray(contrast)
    order = np.argsort(noise)
    crosses = noise.min() <= level <= noise.max()
    return [float(np.interp(level, noise[order], contrast[order, i])) if crosses else np.nan for i in range(contrast.shape[1])]

curves = {name: (results[name].noise, results[name].contrast) for name in ['OSEM', 'OSMAPOSL', 'KEM']}
curves['OSEM + 10 mm filter'] = (results['OSEM'].noise_filtered, results['OSEM'].contrast_filtered)
curves['BSREM, over beta'] = ([results[f'BSREM {b}'].noise[-1] for b in betas], [results[f'BSREM {b}'].contrast[-1] for b in betas])
print(f"OSEM reaches the scanner's noise, {100 * scanner_noise:.1f}%, after {np.interp(scanner_noise, results['OSEM'].noise, iterations):.1f} iterations")
print(f"contrast at that noise   " + "  ".join(f"{name:>6s}" for name in list(spheres)[:5]))
for name, (noise, contrast) in curves.items():
    print(f"{name:22s} " + "  ".join(f"{c:6.2f}" for c in at_noise(noise, contrast, scanner_noise)[:5]))
print(f"{'scanner':22s} " + "  ".join(f"{c:6.2f}" for c in scanner_contrast[:5]))

# %% Save the images
from pytomography.io import save_dicom

for name, image in [('OSEM', recon_osem), ('OSEM_filtered', recon_osem_filtered), ('OSMAPOSL', recon_osmaposl),
                    ('BSREM', recon_bsrem[beta_shown]), ('KEM', recon_kem)]:
    save_dicom(image, OUTPUT / 't_algorithms_measured' / name, object_meta, reference=file_NM, units='counts',
               series_description=f'{name.replace("_", " ")} {n_iters}x{n_subsets}', overwrite=True)
