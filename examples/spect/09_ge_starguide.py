"""GE StarGuide

A 12-head CZT system with body-contour sweeps.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_starguide.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% StarGuide Reconstruction
import pydicom
import matplotlib.pyplot as plt
import os
import numpy as np
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.io.SPECT import dicom
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.projectors.SPECT import StarGuideSystemMatrix

PATH = '/mnt/mydisk2/pytomo_tutorial_data/SPECT'

path_NM = os.path.join(PATH, 'NEMA-Starguide', 'NM_files')
files_NM = [os.path.join(path_NM, f) for f in os.listdir(path_NM)][:12]

object_meta, proj_meta = dicom.get_starguide_metadata(files_NM, nearest_theta=0.1)
projections = dicom.get_starguide_projections(files_NM)
print(projections.shape)

path_CT = os.path.join(PATH, 'NEMA-Starguide', 'CT_files')
files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
attenuation_map = dicom.get_starguide_attenuation_map_from_CT_slices(files_CT, files_NM, index_peak=0)
attenuation_transform = SPECTAttenuationTransform(attenuation_map, assume_padded = False)

psf_meta = dicom.get_psfmeta_from_scanner_params(
    'G8-LEHR', # According the the header, these are the collimator parameters
    energy_keV=140.5, # Imaging of Tc-99m
    material='tungsten', # It is known that this is the material
    shape = 'square', # collimator hole shape
)
psf_transform = SPECTPSFTransform(psf_meta, assume_padded=False)

system_matrix = StarGuideSystemMatrix(
    object_meta=object_meta,
    proj_meta=proj_meta,
    obj2obj_transforms=[attenuation_transform, psf_transform],
    proj2proj_transforms=[]
)

photopeak = projections[0]
scatter = dicom.get_energy_window_scatter_estimate_projections(
    files_NM[0],
    projections,
    index_peak=0,
    index_lower=1
)

likelihood = PoissonLogLikelihood(system_matrix, photopeak, additive_term=scatter)
reconstruction_algorithm = OSEM(likelihood)
recon_pytomography = reconstruction_algorithm(n_iters=10, n_subsets=10)

ds_recon = pydicom.dcmread(os.path.join(PATH, 'NEMA-Starguide', 'vendor_recon', 'i196884.NMDC.1'))
recon_vendor = ds_recon.pixel_array * ds_recon[0x0011,0x103b].value
recon_vendor = np.transpose(recon_vendor, (2,1,0))
