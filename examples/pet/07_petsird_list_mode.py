"""PETSIRD list mode

Read list-mode data in the open PETSIRD format.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_PETSIRD.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% PETSIRD Data (Listmode)
import torch
import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.priors import RelativeDifferencePrior
from pytomography.algorithms import BSREM
from pytomography.io.PET import petsird, gate
import os
from pytomography.transforms.shared import GaussianFilter
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

folder = '/disk1/pytomography_tutorial_data/petsird_tutorial'
petsird_path = os.path.join(folder, 'mIEC_ETSIPETscanner_1.petsird')
eta_path = os.path.join(folder, 'eta.npy')

detector_ids, header = petsird.get_detector_ids(
    petsird_path,
    read_tof=True,
    read_energy=False,
    time_block_ids=None,
    return_header=True
)

import petsird

petsird.BinaryPETSIRDReader

scanner_LUT = petsird.get_scanner_LUT_from_header(header)
tof_meta = petsird.get_TOF_meta_from_header(header)

dr = (2.5, 2.5, 2.5)
shape = (128,128,44)
object_meta = ObjectMeta(dr, shape)

detector_ids = gate.remove_events_out_of_bounds(detector_ids, scanner_LUT, object_meta)

weights_sensitivity = torch.tensor(np.load(eta_path))
proj_meta_nonTOF = PETLMProjMeta(
    detector_ids=detector_ids[:,:2],
    scanner_LUT=scanner_LUT,
    weights_sensitivity=weights_sensitivity)
proj_meta_TOF = PETLMProjMeta(
    detector_ids=detector_ids,
    scanner_LUT=scanner_LUT,
    tof_meta=tof_meta,
    weights_sensitivity=weights_sensitivity)

amap = gate.get_aligned_attenuation_map('/home/gpuvmadm/PyTomography/notebook_testing/ETSIPET_ACmap_IEC_10cmRadius.hv', object_meta)

psf_transform = GaussianFilter(4)
system_matrix_nontof = PETLMSystemMatrix(
    object_meta,
    proj_meta_nonTOF,
    obj2obj_transforms=[psf_transform],
    attenuation_map=amap,
    N_splits=4
    )
system_matrix_tof = PETLMSystemMatrix(
    object_meta,
    proj_meta_TOF,
    obj2obj_transforms=[psf_transform],
    attenuation_map=amap,
    N_splits=4
    )

likelihood_tof = PoissonLogLikelihood(system_matrix_tof, torch.tensor([1.]).to(pytomography.device))
likelihood_nontof = PoissonLogLikelihood(system_matrix_nontof, torch.tensor([1.]).to(pytomography.device))
prior_rdp = RelativeDifferencePrior(beta=75, gamma=2)
recon_algorithm_nontof = BSREM(
    likelihood_nontof,
    prior=prior_rdp,
)
recon_algorithm_tof = BSREM(
    likelihood_tof,
    prior=prior_rdp,
)

recon_tof = recon_algorithm_tof(n_iters=100, n_subsets=1)

recon_nontof = recon_algorithm_nontof(n_iters=100, n_subsets=1)

# proj_meta_test = PETLMProjMeta(
#     detector_ids=detector_ids[:,:2][15:20],
#     scanner_LUT=scanner_LUT,
#     weights_sensitivity=weights_sensitivity)
# psf_transform = GaussianFilter(4)
# system_matrix_test = PETLMSystemMatrix(
#     object_meta,
#     proj_meta_test,
#     obj2obj_transforms=[psf_transform],
#     attenuation_map=amap,
#     N_splits=4
# )
# fhat = system_matrix_test.backward(torch.tensor([1.]).to(pytomography.device))

affine=np.eye(4); affine[-1,-1] = 0
recon_tof_nib = nib.Nifti1Image(recon_tof[0].cpu().numpy(), affine=affine)
nib.save(recon_tof_nib, os.path.join(folder, 'bsrem_nopsf_tof.nii.gz'))
recon_nontof_nib = nib.Nifti1Image(recon_nontof[0].cpu().numpy(), affine=affine)
nib.save(recon_nontof_nib, os.path.join(folder, 'bsrem_nopsf_nontof.nii.gz'))
