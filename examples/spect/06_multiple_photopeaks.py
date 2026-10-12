"""Multiple photopeaks

Joint reconstruction over two photopeaks of the same isotope.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_dualpeak.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% Multi Photopeak Reconstruction
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Lu177-NEMA-SymT2")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Lu177-NEMA-SymT2")

import os
from pytomography import io
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.projectors import ExtendedSystemMatrix
import pytomography
import matplotlib.pyplot as plt
import torch

PATH = DATA / 'SPECT'

path = os.path.join(PATH, 'Lu177-NEMA-SymT2')
path_CT = os.path.join(path, 'CT')
files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
file_NM = os.path.join(path, 'projection_data.dcm')

projections, object_meta, proj_meta = io.SPECT.read_dicom(file_NM)
windows = proj_meta.energy_windows
for i, window in enumerate(windows):
    print(f'Index {i}:   Name: "{window.name}", Energies: {window.lower_keV:.1f}-{window.upper_keV:.1f} keV, photon {window.photon_keV:g} keV')

photopeak208 = projections[0]
photopeak113 = projections[3]
scatter208 = io.SPECT.energy_window_scatter(projections, proj_meta, peak=0, lower=1, upper=2)
scatter113 = io.SPECT.energy_window_scatter(projections, proj_meta, peak=3, lower=4, upper=5)
amap208 = io.attenuation_map(files_CT, object_meta, energy_keV=windows[0].photon_keV)
amap113 = io.attenuation_map(files_CT, object_meta, energy_keV=windows[3].photon_keV)
collimator_name = 'SY-ME'
intrinsic_fwhm_140keV_mm = 3.8 # mm at 140 keV
psf_meta208 = io.SPECT.collimator_psf(
    collimator_name,
    windows[0].photon_keV,
    intrinsic_fwhm_140keV_mm=intrinsic_fwhm_140keV_mm
)
psf_meta113 = io.SPECT.collimator_psf(
    collimator_name,
    windows[3].photon_keV,
    intrinsic_fwhm_140keV_mm=intrinsic_fwhm_140keV_mm
)
att_transform208 = SPECTAttenuationTransform(attenuation_map=amap208)
att_transform113 = SPECTAttenuationTransform(attenuation_map=amap113)
psf_transform208 = SPECTPSFTransform(psf_meta=psf_meta208)
psf_transform113 = SPECTPSFTransform(psf_meta=psf_meta113)
system_matrix208 = SPECTSystemMatrix(
    obj2obj_transforms=[att_transform208, psf_transform208],
    proj2proj_transforms=[],
    object_meta=object_meta,
    proj_meta=proj_meta,
)
system_matrix113 = SPECTSystemMatrix(
    obj2obj_transforms=[att_transform113, psf_transform113],
    proj2proj_transforms=[],
    object_meta=object_meta,
    proj_meta=proj_meta,
)

# These calibration factors are approximate and come from SIMIND
calib208 = 10.2419 # CPS/MBq
calib113 = 11.3 # CPS/MBq

system_matrix = ExtendedSystemMatrix(
    system_matrices= [
        calib113*system_matrix113,
        calib208*system_matrix208
    ] 
)

test_object = torch.zeros(object_meta.shape).to(pytomography.device)
test_object[50:70,50:70,50:70] = 1

test_projections = system_matrix.forward(test_object)
test_projections.shape

photopeak = torch.stack([photopeak113, photopeak208], dim=0)
scatter = torch.stack([scatter113, scatter208], dim=0)
likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter)

print(OSEM(PoissonLogLikelihood(system_matrix208, photopeak208, scatter208)).estimate_memory(n_subsets=8))
recon_algo = OSEM(likelihood)
recon_OSEM = recon_algo(n_iters = 8, n_subsets=8)

float(proj_meta.frame_duration_s[0])  # seconds

# %% Save the images
saved = io.save_dicom(recon_OSEM, OUTPUT / 't_dualpeak' / 'osem', object_meta, reference=file_NM, units='MBq.s',
                      series_description='OSEM 8x8, 113 and 208 keV', overwrite=True)
