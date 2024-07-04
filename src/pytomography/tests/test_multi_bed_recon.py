"""regression test for multi bed recon"""

import os
import pathlib
from pathlib import Path
import numpy as np
from pytomography.io.SPECT import dicom
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.algorithms import OSEM
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.utils import print_collimator_parameters
import matplotlib.pyplot as plt
import pydicom
import torch


files_NM = []

def test_multi_bed_recon():
    save_path = Path.home().joinpath('Downloads')
    files_NM = [
    Path.joinpath(save_path, 'dicom_multibed_tutorial', 'bed1_projections.dcm'),
    Path.joinpath(save_path, 'dicom_multibed_tutorial', 'bed2_projections.dcm'),
    ]
    path_CT = Path.joinpath(save_path, 'dicom_multibed_tutorial', 'CT')
    files_CT = list(path_CT.glob('**/*.dcm'))
    print(pydicom.read_file(files_NM[0]).EnergyWindowInformationSequence)
    projections_upper_fov = dicom.get_projections(files_NM[0])
    projections_lower_fov = dicom.get_projections(files_NM[1])
    projections_upper, projections_lower = projectionss = dicom.load_multibed_projections(files_NM)
    attenuation_map1 = dicom.get_attenuation_map_from_CT_slices(files_CT, files_NM[0], index_peak=1)
    attenuation_map2 = dicom.get_attenuation_map_from_CT_slices(files_CT, files_NM[1], index_peak=1)
    recon_upper = reconstruct_singlebed(0, projectionss, files_NM, files_CT)
    recon_lower = reconstruct_singlebed(1, projectionss, files_NM, files_CT)
    recon_stitched = dicom.stitch_multibed(
        recons=torch.stack([recon_upper, recon_lower]),
        files_NM = files_NM)
    recon_save_path = Path.joinpath(save_path, 'dicom_multibed_tutorial', 'pytomo_recon')
    dicom.save_dcm(
        save_path = recon_save_path,
        object = recon_stitched,
        file_NM = files_NM[0],
        recon_name = 'OSEM_4it_8ss',
        scale_by_number_projections=True,
        single_dicom_file=True)
    files_recon = list (recon_save_path.glob('**/*.dcm'))
    assert(len(files_recon) == 1)
    ds = pydicom.dcmread(files_recon[0],force=True)
    assert ("RECON TOMO" in ds.ImageType)
    
def reconstruct_singlebed(i, projectionss, files_NM, files_CT):
    # Change these depending on your file:
    index_peak = 1
    index_lower = 3
    index_upper = 2
    projections = projectionss[i]
    file_NM = files_NM[i]
    object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=1)
    photopeak = projections[index_peak]
    scatter = dicom.get_energy_window_scatter_estimate_projections(file_NM, projections, index_peak, index_lower, index_upper)
    # Build system matrix
    attenuation_map = dicom.get_attenuation_map_from_CT_slices(files_CT, file_NM, index_peak=1)
    psf_meta = dicom.get_psfmeta_from_scanner_params('GI-MEGP', energy_keV=208)
    att_transform = SPECTAttenuationTransform(attenuation_map)
    psf_transform = SPECTPSFTransform(psf_meta)
    # Create system matrix
    system_matrix = SPECTSystemMatrix(
        obj2obj_transforms = [att_transform,psf_transform],
        proj2proj_transforms= [],
        object_meta = object_meta,
        proj_meta = proj_meta)
    likelihood = PoissonLogLikelihood(system_matrix, photopeak, additive_term=scatter)
    reconstruction_algorithm = OSEM(likelihood)
    return reconstruction_algorithm(n_iters=4, n_subsets=8)

if __name__ == '__main__':
    test_multi_bed_recon()
