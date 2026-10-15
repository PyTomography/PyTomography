"""Ac-225 PSF modelling (SIMIND)

Alpha-emitter imaging with PSFs from Monte Carlo ARFs (spectarf), penetration and collimator scatter included.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_ac225_simind_recon.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
# %% Ac-225 Advanced PSF Modeling (SIMIND)
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/SIMIND-Jaszak")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/SIMIND-Jaszak")

import os
from pytomography import io
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT  import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from spectarf import DetectorResponse
from spectarf.pytomography import camera_model, generate_arfs, psf_transform
from spectarf.response import FWHM_TO_SIGMA

# change to where you saved the data
PATH = DATA / 'SPECT' / 'SIMIND-Jaszak'

dT = 2.5 * 60 # seconds per projection
activity_conc = 10 # kBq/L, a very high activity conc for Ac225
CPSpMBq = 17 # an approximate calibration factor

# Specify the isotopes and equilibirum ratios based on Bateman equations
isotopes = ['ac225', 'bi213', 'fr221', 'tl209']
isotopes_ratios = [1,1,1,0.0209]
# look at the .h00 files; there are many E windows simulated. These are 5% scat win for 440peak
i_peak, i_lower, i_upper = 10, 11, 14
files_NM = [[os.path.join(PATH, f'{isotope}', f'tot_w{i}.h00') for isotope in isotopes] for i in [i_lower, i_peak, i_upper]]
activity_concs = [activity_conc * ratio for ratio in isotopes_ratios]
projections, object_meta, proj_meta = io.SPECT.read_simind(files_NM, weights=activity_concs)
projections *= dT

# Based on how they are loaded, this is idx of peak, lower upper
idx_peak, idx_lower, idx_upper = 1, 0, 2
photopeak = projections[idx_peak]
# each window's width from its header (the first isotope's; the isotopes share the windows)
ww_lower, ww_peak, ww_upper = [window.width for window in proj_meta.energy_windows]
print(f'window widths: lower {ww_lower:.0f} keV, photopeak {ww_peak:.0f} keV, upper {ww_upper:.0f} keV')
scatter_estimate_TEW = io.SPECT.energy_window_scatter(
    projections, proj_meta, peak=idx_peak, lower=idx_lower, upper=idx_upper,
    fwhm_r=11.8,   # mm
    fwhm_z=11.8,
)

path_amap = os.path.join(PATH, 'attenuation_maps', 'amap440.hct')
amap = io.attenuation_map(path_amap, object_meta, energy=proj_meta.energy_windows[idx_peak].photon_energy, layout='mu')

def perform_reconstruction(psf_transform):
    att_transform = SPECTAttenuationTransform(attenuation_map=amap)
    system_matrix = SPECTSystemMatrix(
            obj2obj_transforms = [att_transform,psf_transform],
            proj2proj_transforms = [],
            object_meta = object_meta,
            proj_meta = proj_meta)
    likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter_estimate_TEW)
    algorithm = OSEM(likelihood)
    print(algorithm.estimate_memory(n_subsets=4))
    return algorithm(n_iters=50, n_subsets=4)

# %% The camera
collimator = io.SPECT.collimator('SY-HE')            # the Symbia high energy collimator
detector = io.SPECT.detector('NaI-9.5mm')            # 3/8 inch NaI(Tl), its cover and backscatter
camera = io.SPECT.camera('Siemens Symbia T')          # its energy resolution model
# Or every value explicitly:
# collimator = dict(hole_diameter=4.0, septal_thickness=2.0, hole_length=59.7, hole_shape='hex', rotated=True,
#                   material='lead')                    # mm
# detector = dict(crystal_thickness=9.525, crystal_material='nai', cover_thickness=1.0, cover_material='aluminium',
#                 backscatter_thickness=100.0, backscatter_material='pmt')   # mm
stack, response = camera_model(collimator, detector, camera)   # spectarf's camera from these numbers
# SIMIND adds its own 3.6 mm Gaussian (FWHM at 140 keV) on top of its transport; spectarf's own arguments
response = DetectorResponse(response.energy, intrinsic_fwhm_mm_140=3.6, residual_sigma_mm_140=3.6 * FWHM_TO_SIGMA)

# %% The ARFs
arfs = generate_arfs(stack, response, proj_meta, peak=idx_peak)

# %% Reconstruct
psf_transform_arf = psf_transform(arfs, normalize=True)
recon_arf = perform_reconstruction(psf_transform_arf)

path_bi213_prim = files_NM[1][1]
psf_meta = io.SPECT.simind_psf(path_bi213_prim)
psf_transform = SPECTPSFTransform(psf_meta)
reconbad = perform_reconstruction(psf_transform)

# %% Save the images
for name, image in [('arf_psf', recon_arf), ('gaussian_psf', reconbad)]:
    io.save_dicom(image, OUTPUT / 't_ac225_simind_recon' / name, object_meta, units='counts',
                  series_description=f'OSEM 50x4, {name}', overwrite=True)
