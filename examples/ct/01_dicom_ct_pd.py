"""DICOM-CT-PD

OS-SART on raw projections from a 3rd-generation clinical CT scanner.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_CT_GEN3.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% DICOM-CT-PD (3rd Generation CT)
from __future__ import annotations
from pytomography.metadata import ObjectMeta
from pytomography.algorithms import SART
from pytomography.io.CT import dicom_ct_pd
from pytomography.projectors.CT import CTGen3SystemMatrix
import matplotlib.pyplot as plt
import os

save_path = '/disk1/ct_proj/DICOM-CT-PD_FD' # Path where you downloaded the CT-PD data
paths = [os.path.join(save_path, file) for file in os.listdir(save_path) if '.dcm' in file]

proj, proj_meta = dicom_ct_pd.get_projections_and_metadata_gen3(paths)

dx = dy = dz =  1.0 # mm
Nx = Ny = Nz = 512
object_meta = ObjectMeta(dr=(dx,dx,dx), shape=(Nx,Ny,Nz))

system_matrix = CTGen3SystemMatrix(object_meta, proj_meta, N_splits=1, device='cpu')

recon_algorithm = SART(system_matrix, proj)
recon = recon_algorithm(n_iters=3, n_subsets=40)
