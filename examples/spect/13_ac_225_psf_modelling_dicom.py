"""Ac-225 PSF modelling (DICOM)

The same ARF PSFs on a measured Symbia T2 scan with a non-circular orbit.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_ac225_dicom_recon.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
# %% Ac-225 Advanced PSF Modeling (DICOM)
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Ac225-NEMA-SymT2")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Ac225-NEMA-SymT2")

from pytomography import io
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.SPECT import SPECTAttenuationTransform
from pytomography.transforms.shared import GaussianFilter
import os
from spectarf.pytomography import camera_model, generate_arfs, psf_transform

path = DATA / 'SPECT'

pathCT = os.path.join(path, 'Ac225-NEMA-SymT2', 'CT')
files_CT = [os.path.join(pathCT, file) for file in os.listdir(pathCT)]
file_NM = os.path.join(path, 'Ac225-NEMA-SymT2', 'projection_data.IMA')

# %% The camera
collimator = io.SPECT.collimator('SY-HE')            # the Symbia high energy collimator
detector = io.SPECT.detector('NaI-9.5mm')            # 3/8 inch NaI(Tl), its cover and backscatter
camera = io.SPECT.camera('Siemens Symbia T')          # energy resolution model and intrinsic resolution
# Or every value explicitly:
# collimator = dict(hole_diameter=4.0, septal_thickness=2.0, hole_length=59.7, hole_shape='hex', rotated=True,
#                   material='lead')                    # mm
# detector = dict(crystal_thickness=9.525, crystal_material='nai', cover_thickness=1.0, cover_material='aluminium',
#                 backscatter_thickness=100.0, backscatter_material='pmt')   # mm
stack, response = camera_model(collimator, detector, camera)   # spectarf's camera from these numbers

index_peak = 3
index_lower = 4
index_upper = 5
projections, object_meta, proj_meta = io.SPECT.read_dicom(file_NM)
attenuation_map = io.attenuation_map(files_CT, object_meta, energy=proj_meta.energy_windows[index_peak].photon_energy)

photopeak = projections[index_peak]
scatter = io.SPECT.energy_window_scatter(projections, proj_meta, peak=index_peak, lower=index_lower, upper=index_upper,
                                         fwhm_theta=4.7, fwhm_r=11.3, fwhm_z=11.3)

# %% The ARFs
arfs = generate_arfs(stack, response, proj_meta, peak=index_peak)

# %% Reconstruct
att_transform = SPECTAttenuationTransform(attenuation_map=attenuation_map)
psf = psf_transform(arfs, normalize=True)
system_matrix = SPECTSystemMatrix(
        obj2obj_transforms = [att_transform, psf],
        proj2proj_transforms = [],
        object_meta = object_meta,
        proj_meta = proj_meta)

likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter)
algorithm = OSEM(likelihood)
print(algorithm.estimate_memory(n_subsets=3))

recon = algorithm(n_iters=100, n_subsets=3)

filter = GaussianFilter(20) # FWHM in mm
filter.configure(object_meta, proj_meta)
recon_smoothed = filter(recon)

# %% Save the images
for name, image in [('osem', recon), ('osem_smoothed', recon_smoothed)]:
    io.save_dicom(image, OUTPUT / 't_ac225_dicom_recon' / name, object_meta, reference=file_NM,
                  series_description=f'OSEM 100x3, {name}', overwrite=True)
