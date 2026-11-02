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
from pytomography.likelihoods import NegativeMSELikelihood, PoissonLogLikelihood, poisson_weights
from pytomography.algorithms import CGLS, OSEM

# Memory: each bed is 128 x 128 x 128 voxels and 60 views, so a reconstruction needs about 0.5 GB of GPU memory and the
# tutorial peaks at about 6 GB of RAM (measured on Windows); nothing here grows with RAM_GB. CGLS holds four images and
# two sets of projections per bed while it runs.
RAM_GB = 25
pytomography.set_memory_budget(RAM_GB)

# %% 1. The patient
path = DATA / 'SPECT' / 'Lu177-PSMA-GEDisc'
files_NM = [str(path / 'bed1_projections.dcm'), str(path / 'bed2_projections.dcm')]
files_CT = [str(path / 'CT' / f) for f in os.listdir(path / 'CT')]
file_RT = str(path / 'segmentations.dcm')  # organs drawn on the CT (RTSTRUCT)

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
def reconstruct(make, **settings):
    """Reconstruct both beds with make(bed)(**settings), and stitch them; also return the algorithms."""
    algorithms = [make(bed) for bed in beds]
    t0 = time.time()
    recons = [algorithm(**settings) for algorithm in algorithms]
    seconds = time.time() - t0
    stitched, object_meta = io.SPECT.stitch_beds(recons=torch.stack(recons[::-1]),
                                                 object_metas=[bed['object_meta'] for bed in beds[::-1]])
    return stitched, object_meta, algorithms, seconds

osem = {}
for n in (2, 4, 8, 16, 24):
    image, object_meta_stitched, _, seconds = reconstruct(
        lambda bed: OSEM(PoissonLogLikelihood(bed['system_matrix'], bed['photopeak'], additive_term=bed['scatter'])),
        n_iters=n, n_subsets=8)
    osem[n] = image
    print(f"OSEM, {n:2d} iterations of 8 subsets: {seconds:4.1f} s for both beds")

# %% 4. CGLS
cgls = {}
for n in (5, 10, 20, 40, 80, 120, 160):
    image, _, _, seconds = reconstruct(
        lambda bed: CGLS(NegativeMSELikelihood(bed['system_matrix'], bed['photopeak'], additive_term=bed['scatter'])),
        n_iters=n, rtol=0)
    cgls[n] = image
    print(f"CGLS, {n:3d} iterations: {seconds:4.1f} s for both beds")

# %% A diagonal preconditioner
def preconditioned(bed):
    H = bed['system_matrix']
    ones = torch.ones(H.object_meta.shape, device=pytomography.device)
    D = 1 / H.backward(H.forward(ones)).clamp(min=1e-6)
    return CGLS(NegativeMSELikelihood(H, bed['photopeak'], additive_term=bed['scatter']), preconditioner=D)

pcgls = {}
for n in (5, 10, 20, 40, 60):
    image, _, _, seconds = reconstruct(preconditioned, n_iters=n, rtol=0)
    pcgls[n] = image
    print(f"Preconditioned CGLS, {n:2d} iterations: {seconds:4.1f} s for both beds")

# %% Weighted least squares
def weighted(bed):
    H, g = bed['system_matrix'], bed['photopeak']
    w = poisson_weights(g, smoothing=2, min_counts=0.5)  # 1 / the counts smoothed within each view
    ones = torch.ones(H.object_meta.shape, device=pytomography.device)
    D = 1 / H.backward(w * H.forward(ones)).clamp(min=1e-6)
    return CGLS(NegativeMSELikelihood(H, g, additive_term=bed['scatter'], weights=w), preconditioner=D)

wcgls = {}
for n in (5, 10, 20, 30, 40, 60):
    image, _, _, seconds = reconstruct(weighted, n_iters=n, rtol=0)
    wcgls[n] = image
    print(f"Weighted CGLS (preconditioned), {n:2d} iterations: {seconds:4.1f} s for both beds")

# %% 5. When to stop: the kidneys
rois = ['liver', 'kidney_left', 'kidney_right', 'spleen']
masks = {roi: io.load_mask(file_RT, reference=str(path / 'CT'), rois=[roi]).on(object_meta_stitched, label=roi,
                                                                                 threshold=0.5).to(pytomography.device)
         for roi in rois}
kidneys = masks['kidney_left'] | masks['kidney_right']

def scores(image):
    """The kidneys' mean (counts per voxel) and the liver's noise (coefficient of variation, %)."""
    liver = image[masks['liver']]
    return float(image[kidneys].mean()), float(100 * liver.std() / liver.mean())

def stop(runs):
    """The fewest iterations whose kidney mean is within 2% of the longest run's."""
    final = scores(runs[max(runs)])[0]
    return min(n for n in runs if scores(runs[n])[0] >= 0.98 * final)

methods = {'OSEM': osem, 'CGLS': cgls, 'Preconditioned CGLS': pcgls, 'Weighted CGLS': wcgls}
stops = {name: stop(runs) for name, runs in methods.items()}
for name, runs in methods.items():
    kidney, noise = scores(runs[stops[name]])
    print(f"{name:20s} stops at {stops[name]:3d} iterations: kidney mean {kidney:.2f}, liver noise {noise:.0f}%")

chosen = wcgls[stops['Weighted CGLS']]
unweighted = pcgls[stops['Preconditioned CGLS']]
osem_stopped = osem[stops['OSEM']]

# %% 6. Negative values
body = io.attenuation_map(files_CT, object_meta_stitched, energy=208.0).to(pytomography.device) > 0.005
for name, image in [(f'OSEM {stops["OSEM"]} x 8', osem_stopped), (f'CGLS {stops["CGLS"]}', cgls[stops['CGLS']]),
                    (f'Preconditioned CGLS {stops["Preconditioned CGLS"]}', unweighted),
                    (f'Weighted CGLS {stops["Weighted CGLS"]}', chosen)]:
    print(f"{name:26s} negative voxels in the body: {100 * float((image[body] < 0).float().mean()):4.1f}%   "
          f"total: {float(image.sum()):9.0f} counts")

# %% 7. OSEM and CGLS side by side
print(f"{'':28s}" + ''.join(f'{roi:>14s}' for roi in rois) + f"{'liver noise':>14s}")
for name, image in [('OSEM 4 x 8', osem[4]), (f'OSEM {stops["OSEM"]} x 8', osem_stopped),
                    (f'CGLS {stops["CGLS"]}', cgls[stops['CGLS']]),
                    (f'Preconditioned CGLS {stops["Preconditioned CGLS"]}', unweighted),
                    (f'Weighted CGLS {stops["Weighted CGLS"]}', chosen)]:
    means = ''.join(f'{float(image[masks[roi]].mean()):14.2f}' for roi in rois)
    print(f"{name:28s}{means}{scores(image)[1]:13.0f}%")

# %% Save the images
saved = io.save_dicom(chosen.clamp(min=0), OUTPUT / 't_cgls' / 'weighted_cgls', object_meta_stitched,
                      reference=files_NM[1],
                      series_description=f'Weighted CGLS {stops["Weighted CGLS"]}, two beds stitched',
                      overwrite=True)
