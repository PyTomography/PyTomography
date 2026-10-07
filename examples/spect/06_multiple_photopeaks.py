"""Multiple photopeaks

Joint reconstruction over two photopeaks of the same isotope.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_dualpeak.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% Multi Photopeak Reconstruction
import os
from pathlib import Path

# Tutorial data: the folder set by the PYTOMOGRAPHY_DATA environment variable (see Tutorial data in the docs)
DATA = Path(os.environ.get("PYTOMOGRAPHY_DATA", "~/pytomography_data")).expanduser()
# Results go here, never into the data folder
OUTPUT = Path(os.environ.get("PYTOMOGRAPHY_OUTPUT", "pytomography_outputs")).expanduser() / "SPECT/Lu177-NEMA-SymT2"
OUTPUT.mkdir(parents=True, exist_ok=True)

import os
import numpy as np
from pytomography.io.SPECT import dicom
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.utils import print_collimator_parameters
from pytomography.projectors import ExtendedSystemMatrix
import pytomography
import matplotlib.pyplot as plt
import pydicom
import torch
import shutil

PATH = DATA / 'SPECT'

path = os.path.join(PATH, 'Lu177-NEMA-SymT2')
path_CT = os.path.join(path, 'CT')
files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
file_NM = os.path.join(path, 'projection_data.dcm')

dicom.print_energy_window_info(file_NM)

object_meta, proj_meta = dicom.get_metadata(file_NM)
photopeak208 = dicom.get_projections(file_NM, index_peak=0)
photopeak113 = dicom.get_projections(file_NM, index_peak=3)
scatter208 = dicom.get_energy_window_scatter_estimate(file_NM, index_peak=0, index_lower=1, index_upper=2)
scatter113 = dicom.get_energy_window_scatter_estimate(file_NM, index_peak=3, index_lower=4, index_upper=5)
amap208 = dicom.get_attenuation_map_from_CT_slices(files_CT, file_NM, index_peak=0)
amap113 = dicom.get_attenuation_map_from_CT_slices(files_CT, file_NM, index_peak=3)
collimator_name = 'SY-ME'
intrinsic_resolution=0.38 #mm at 140 keV
psf_meta208 = dicom.get_psfmeta_from_scanner_params(
    collimator_name,
    energy_keV =  208,
    intrinsic_resolution_140keV=intrinsic_resolution
)
psf_meta113 = dicom.get_psfmeta_from_scanner_params(
    collimator_name,
    energy_keV = 113,
    intrinsic_resolution_140keV=intrinsic_resolution
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

test_object = torch.zeros((128,128,128)).to(pytomography.device)
test_object[50:70,50:70,50:70] = 1

test_projections = system_matrix.forward(test_object)
test_projections.shape

photopeak = torch.stack([photopeak113, photopeak208], dim=0)
scatter = torch.stack([scatter113, scatter208], dim=0)
likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter)

recon_algo = OSEM(likelihood)
recon_OSEM = recon_algo(n_iters = 8, n_subsets=8)

pydicom.dcmread(file_NM).RotationInformationSequence[0].ActualFrameDuration / 1000
