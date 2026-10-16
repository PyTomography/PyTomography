"""PSFs from angular response functions

The collimator and detector simulated once, the PSF at any distance, and what each camera parameter changes.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_arf_psf.html
It computes and saves the results; the plots, what only they need, and the explanations are in the notebook.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
# %% PSFs from angular response functions (ARFs)
from pytomography import datasets

# The tutorial data: downloaded the first time it runs (see Tutorial data in the docs)
datasets.fetch("SPECT/Ac225-NEMA-SymT2")
DATA = datasets.data_dir()  # the PYTOMOGRAPHY_DATA folder, or ~/pytomography_data
# Results go here, never into the data folder
OUTPUT = datasets.output_dir("SPECT/Ac225-NEMA-SymT2")

import os
import time
from pytomography import io
from spectarf.pytomography import camera_model, generate_arfs, psf_at, psf_transform, save_arfs, load_arfs

# %% The camera
collimator = io.SPECT.collimator('SY-HE')            # the Siemens Symbia high energy collimator
detector = io.SPECT.detector('NaI-9.5mm')            # 3/8 inch NaI(Tl), its cover and backscatter
camera = io.SPECT.camera('Siemens Symbia T')          # energy resolution model and intrinsic resolution
# Or every value explicitly:
# collimator = dict(hole_diameter=4.0, septal_thickness=2.0, hole_length=59.7, hole_shape='hex', rotated=True,
#                   material='lead')                    # mm
# detector = dict(crystal_thickness=9.525, crystal_material='nai', cover_thickness=1.0, cover_material='aluminium',
#                 backscatter_thickness=100.0, backscatter_material='pmt')   # mm
print(collimator, detector, camera, sep='\n')

stack, response = camera_model(collimator, detector, camera)   # spectarf's camera from these numbers

# %% The ARFs
file_NM = os.path.join(DATA, 'SPECT', 'Ac225-NEMA-SymT2', 'projection_data.IMA')
projections, object_meta, proj_meta = io.SPECT.read_dicom(file_NM)
index_peak, index_lower, index_upper = 3, 4, 5        # 396-484 keV, and the scatter windows below and above

t0 = time.time()
arfs = generate_arfs(stack, response, proj_meta, peak=index_peak, lower=index_lower, upper=index_upper)
print(f'ARFs in {time.time() - t0:.0f} s (read from the cache if this camera was simulated before)')
for name, arf in arfs.items():
    print(f"{name}: {arf.window[1]:.0f}-{arf.window[2]:.0f} keV, {arf.energy_keV:.1f} keV photons, "
          f"grid (theta, phi, dx, dy) {arf.data.shape}")
save_arfs(arfs, OUTPUT / 't_arf_psf' / 'SY-HE')
arfs = load_arfs(OUTPUT / 't_arf_psf' / 'SY-HE')       # read back: on any machine, no GPU needed

# %% Photopeak only, or consistent with the TEW estimate
photopeak = psf_at(arfs, 150.0, pixel_mm=1.0, size_mm=160.0)
consistent = psf_at(arfs, 150.0, pixel_mm=1.0, size_mm=160.0, tew=True)
print(f'the TEW estimate takes {1 - consistent.sum() / photopeak.sum():.1%} of the photopeak kernel at 150 mm')

# %% The transform
psf = psf_transform(arfs)
print(type(psf).__name__, '| normalize:', psf.normalize)
