"""GE Discovery MI

Clinical list mode exported from GE Duetto, with time of flight.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_GE_HDF5.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

# %% GE Discovery MI (Listmode Reconstruction; With Time of Flight)
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("PET/GE-DMI-NEMA")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("PET/GE-DMI-NEMA")
GE_DATA = DATA / 'PET' / 'GE-DMI-NEMA'

import pytomography
from pytomography import io
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.algorithms import OSEM, BSREM
from pytomography.priors import RelativeDifferencePrior
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.shared import GaussianFilter
import matplotlib.pyplot as plt

# %% How much memory this needs
RAM_GB = 25 # memory (RAM) this tutorial may use, in GB; it needs about 9 GB however low this is
pytomography.set_memory_budget(RAM_GB)

folder = GE_DATA / 'dmi_nema_lm'
# The listmode events, the object grid and the projection metadata (the scanner's info and its TOF bins)
detector_ids, object_meta, proj_meta = io.PET.read_ge_hdf5(
    folder / 'LIST0000.BLF',
    object_meta=ObjectMeta(dr=(2.78, 2.78, 2.78), shape=(192, 192, 71)) # mm, voxels
)
info = proj_meta.info

weights, additive_counts, detector_ids_sensitivity, weights_sensitivity = io.PET.read_ge_corrections(folder / 'corrections.h5')

# The projection metadata: the events, their TOF bins, and every valid crystal pair with its weight
proj_meta = PETLMProjMeta(
    detector_ids,
    info,
    tof_meta=proj_meta.tof_meta,
    detector_ids_sensitivity=detector_ids_sensitivity,
    weights_sensitivity=weights_sensitivity
    )

psf_transform = GaussianFilter(4.5) # 4.5mm gaussian psf
system_matrix = PETLMSystemMatrix(
    object_meta,
    proj_meta,
    obj2obj_transforms = [psf_transform],
    N_splits=8,
)

likelihood = PoissonLogLikelihood(
    system_matrix,
    additive_term=additive_counts/weights,
)

recon_algorithm = OSEM(likelihood)
print(recon_algorithm.estimate_memory(n_subsets=34)) # its peak RAM and GPU memory, before it runs
recon_OSEM = recon_algorithm(n_iters=2, n_subsets=34)

# This is probably very similar to "Q.Clear" is ;)
prior = RelativeDifferencePrior(beta=25, gamma=2)
recon_algorithm = BSREM(likelihood, prior=prior)
print(recon_algorithm.estimate_memory(n_subsets=34))
recon_BSREM = recon_algorithm(n_iters=20, n_subsets=34)

# %% Save the images
for name, image in [('osem', recon_OSEM), ('bsrem', recon_BSREM)]:
    io.save_dicom(image, OUTPUT / 't_GE_HDF5' / name, object_meta, series_description=name.upper(), overwrite=True)
