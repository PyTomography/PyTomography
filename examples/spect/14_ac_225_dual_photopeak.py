"""Ac-225 dual photopeak

One image from the 218 and 440 keV photopeaks (ARF PSFs at 440 keV), against each peak alone.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_ac225_dualpeak.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
# %% Ac-225 Dual Photopeak Reconstruction
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Ac225-NEMA-SymT2")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Ac225-NEMA-SymT2")

import os
import time
import torch
from pytomography import io
from pytomography.projectors import ExtendedSystemMatrix
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.transforms.shared import GaussianFilter
from spectarf import Collimator, DetectorStack, DetectorResponse, EnergyResolution
from spectarf.nuclides import Nuclide
from spectarf.pytomography import generate_arfs, psf_at, psf_transform

path = DATA / 'SPECT' / 'Ac225-NEMA-SymT2'
files_CT = [os.path.join(path, 'CT', file) for file in os.listdir(path / 'CT')]
file_NM = os.path.join(path, 'projection_data.IMA')

projections, object_meta, proj_meta = io.SPECT.read_dicom(file_NM)
windows = proj_meta.energy_windows
for i, window in enumerate(windows):
    print(f'Index {i}:   Name: "{window.name}", Energies: {window.lower_keV:.1f}-{window.upper_keV:.1f} keV, photon {window.photon_keV:g} keV')

photopeak218 = projections[0]
photopeak440 = projections[3]
scatter218 = io.SPECT.energy_window_scatter(projections, proj_meta, peak=0, lower=1, upper=2,
                                            sigma_theta=2, sigma_r_mm=4.8, sigma_z_mm=4.8)
scatter440 = io.SPECT.energy_window_scatter(projections, proj_meta, peak=3, lower=4, upper=5,
                                            sigma_theta=2, sigma_r_mm=4.8, sigma_z_mm=4.8)
for name, peak, scatter in [('218 keV', photopeak218, scatter218), ('440 keV', photopeak440, scatter440)]:
    print(f'{name}: {peak.sum():,.0f} counts, of which the scatter estimate is {scatter.sum() / peak.sum():.0%}')

# %% The camera
collimator = io.SPECT.collimator('SY-HE')            # the Symbia high energy collimator
detector = io.SPECT.detector('NaI-9.5mm')            # 3/8 inch NaI(Tl), its cover and backscatter
camera = io.SPECT.camera('Siemens Symbia T')          # energy resolution model and intrinsic resolution
stack = DetectorStack(Collimator(**collimator), **detector)
response = DetectorResponse(EnergyResolution.geb(**camera['energy_resolution']),
                            intrinsic_fwhm_mm_140=camera['intrinsic_fwhm_mm_140'])

# %% 440 keV: PSFs from ARFs
arfs440 = generate_arfs(stack, response, proj_meta, peak=3)
attenuation_map440 = io.attenuation_map(files_CT, object_meta, energy_keV=windows[3].photon_keV)
system_matrix440 = SPECTSystemMatrix(
    obj2obj_transforms=[SPECTAttenuationTransform(attenuation_map=attenuation_map440),
                        psf_transform(arfs440, normalize=True)],
    proj2proj_transforms=[],
    object_meta=object_meta,
    proj_meta=proj_meta,
)

# %% 218 keV: the Gaussian PSF
psf_meta218 = io.SPECT.collimator_psf(collimator['code'], windows[0].photon_keV,
                                      intrinsic_fwhm_140keV_mm=camera['intrinsic_fwhm_mm_140'])
attenuation_map218 = io.attenuation_map(files_CT, object_meta, energy_keV=windows[0].photon_keV)
system_matrix218 = SPECTSystemMatrix(
    obj2obj_transforms=[SPECTAttenuationTransform(attenuation_map=attenuation_map218),
                        SPECTPSFTransform(psf_meta=psf_meta218)],
    proj2proj_transforms=[],
    object_meta=object_meta,
    proj_meta=proj_meta,
)

# %% Each peak's sensitivity
ac225 = Nuclide.from_ddep('Ac-225', chain='secular')   # Ac-225 and its daughters, in equilibrium

def photons_per_decay(photon_keV):
    return sum(line.yield_per_decay for line in ac225.lines if abs(line.energy_keV - photon_keV) < 0.5)

def detected_per_photon(arfs):
    return float(psf_at(arfs, 150.0, pixel_mm=float(object_meta.dx), size_mm=600.0).sum())

arfs218 = generate_arfs(stack, response, proj_meta, peak=0)
calib218 = 1e6 * photons_per_decay(windows[0].photon_keV) * detected_per_photon(arfs218)  # cps/MBq
calib440 = 1e6 * photons_per_decay(windows[3].photon_keV) * detected_per_photon(arfs440)  # cps/MBq
for name, keV, calib in [('218 keV', windows[0].photon_keV, calib218), ('440 keV', windows[3].photon_keV, calib440)]:
    print(f'{name}: {photons_per_decay(keV):.4f} photons per decay, {calib:.2f} cps/MBq')

system_matrix218 = calib218 * system_matrix218
system_matrix440 = calib440 * system_matrix440

# %% The joint system matrix
system_matrix = ExtendedSystemMatrix(system_matrices=[system_matrix218, system_matrix440])
photopeak = torch.stack([photopeak218, photopeak440])
scatter = torch.stack([scatter218, scatter440])

# %% Reconstruct
likelihoods = {
    '218 keV': PoissonLogLikelihood(system_matrix218, photopeak218, scatter218),
    '440 keV': PoissonLogLikelihood(system_matrix440, photopeak440, scatter440),
    '218 + 440 keV': PoissonLogLikelihood(system_matrix, photopeak, scatter),
}
print(OSEM(likelihoods['218 + 440 keV']).estimate_memory(n_subsets=3))

recons = {}
for name, likelihood in likelihoods.items():
    start = time.perf_counter()
    recons[name] = OSEM(likelihood)(n_iters=100, n_subsets=3)
    print(f'{name}: {time.perf_counter() - start:.0f} s')

# %% Compare
seconds = float(proj_meta.frame_duration_s[0])
voxel_mL = float(object_meta.dx * object_meta.dy * object_meta.dz) / 1000
smooth = GaussianFilter(20)  # FWHM in mm
smooth.configure(object_meta, proj_meta)
kBq_per_mL = {name: smooth(recon) / seconds / voxel_mL * 1000 for name, recon in recons.items()}
for name, recon in recons.items():
    print(f'{name}: {recon.sum() / seconds:.2f} MBq in the field of view, hottest smoothed voxel {kBq_per_mL[name].max():.1f} kBq/mL')

# %% Save the images
for (name, recon), folder in zip(recons.items(), ['osem_218keV', 'osem_440keV', 'osem_joint']):
    io.save_dicom(recon, OUTPUT / 't_ac225_dualpeak' / folder, object_meta, reference=file_NM, units='MBq.s',
                  series_description=f'OSEM 100x3, {name}', overwrite=True)
