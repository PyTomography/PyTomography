"""Cone-beam micro-CT

FDK and OS-SART on a laboratory micro-CT scan of glass beads, with the centre of rotation found from the data.

Script version of the tutorial at https://pytomography.readthedocs.io/en/latest/notebooks/t_CT_microct.html
It keeps the computation and leaves out the plots and explanations.
Generated from the notebook by docs/tools/export_scripts.py: edit the notebook, not this file.
"""
import matplotlib
matplotlib.use("Agg")  # no figure windows when run as a script

import os
from pathlib import Path

# Tutorial data: the folder set by the PYTOMOGRAPHY_DATA environment variable (see Tutorial data in the docs)
DATA = Path(os.environ.get("PYTOMOGRAPHY_DATA", "~/pytomography_data")).expanduser()
# Results go here, never into the data folder
OUTPUT = Path(os.environ.get("PYTOMOGRAPHY_OUTPUT", "pytomography_outputs")).expanduser() / "CT/SophiaBeads-256"
OUTPUT.mkdir(parents=True, exist_ok=True)

import configparser
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.CT import CTConeBeamFlatPanelProjMeta
from pytomography.projectors.CT import CTConeBeamFlatPanelSystemMatrix
from pytomography.algorithms import SART

# %% 1. The scan
folder = DATA / 'CT' / 'SophiaBeads-256' / 'SophiaBeads_256_averaged'
scan = configparser.ConfigParser()
scan.optionxform = str  # keep the keys' capitals
scan.read(folder / 'SophiaBeads_256_averaged.xtekct')
scan = scan['XTekCT']
source_to_axis = float(scan['SrcToObject'])      # mm
source_to_detector = float(scan['SrcToDetector'])  # mm
pixel = float(scan['DetectorPixelSizeX'])          # mm
white_level = float(scan['WhiteLevel'])            # detector reading with nothing in the beam
n_projections = int(scan['Projections'])

# One line per projection, e.g. "2:	-000.001" (degrees), after a header line
lines = (folder / 'SophiaBeads_256_averaged.ang').read_text().splitlines()[1:]
angles_deg = np.array([float(line.split(':')[1]) for line in lines if line.strip()])

magnification = source_to_detector / source_to_axis
print(f"{n_projections} projections {abs(np.diff(angles_deg).mean()):.2f}° apart, "
      f"{scan['DetectorPixelsX']} × {scan['DetectorPixelsY']} pixels of {pixel} mm, magnification {magnification:.2f}")

# %% 2. Projections
binning = 4  # 8 gives a grid with eight times fewer voxels, if your GPU has less than 8 GB of memory

t0 = time.time()
projections = []
for i in range(1, n_projections + 1):
    reading = np.array(Image.open(folder / f'SophiaBeads_256_averaged_{i:04d}.tif'), dtype=np.float32)
    reading = F.avg_pool2d(torch.from_numpy(reading)[None, None], binning)[0, 0]
    projections.append(-torch.log(reading.clamp(min=1) / white_level))
# PyTomography wants (angle, detector column, detector row)
proj = torch.stack(projections).transpose(1, 2).contiguous().to(pytomography.device)
print(f"projections {tuple(proj.shape)} read in {time.time() - t0:.0f} s")

# %% 3. Geometry
pixel_binned = pixel * binning
voxel = pixel_binned / magnification
n = 2000 // binning
thetas = -torch.tensor(angles_deg) * np.pi / 180


def system_matrix_for(cor, n_slices):
    proj_meta = CTConeBeamFlatPanelProjMeta(
        thetas, torch.zeros(n_projections),
        detector_radius=source_to_detector - source_to_axis,
        beam_radius=source_to_axis,
        shape=(n, n), dr=(pixel_binned, pixel_binned), COR=cor)
    object_meta = ObjectMeta(dr=(voxel, voxel, voxel), shape=(n, n, n_slices))
    return CTConeBeamFlatPanelSystemMatrix(object_meta, proj_meta)

# %% 4. The centre of rotation
def sharpness(image):
    s = image[:, :, image.shape[2] // 2]
    return float(((s[1:] - s[:-1]) ** 2).mean() + ((s[:, 1:] - s[:, :-1]) ** 2).mean())


cors = np.round(np.arange(0.0, 0.5, 0.02), 2)  # mm
trials = {cor: system_matrix_for(cor, 4).backward(proj, projection_type='FBP') for cor in cors}
scores = [sharpness(trials[cor]) for cor in cors]
best_cor = float(cors[int(np.argmax(scores))])
print(f"sharpest at COR = {best_cor} mm")

# %% 5. FDK
system_matrix = system_matrix_for(best_cor, n)
t0 = time.time()
recon_fdk = system_matrix.backward(proj, projection_type='FBP')
print(f"FDK: {time.time() - t0:.1f} s")

# %% 6. OS-SART
t0 = time.time()
recon_sart = SART(system_matrix, proj, object_initial=recon_fdk.clamp(min=0))(n_iters=3, n_subsets=5)
print(f"OS-SART 3 × 5: {time.time() - t0:.1f} s")

# %% 7. Comparison
air = (slice(0, 40), slice(0, 40), slice(n // 2 - 20, n // 2 + 20))  # a corner of the grid outside the tube
for recon, name in [(recon_fdk, 'FDK'), (recon_sart, 'OS-SART')]:
    a = recon[air]
    print(f"{name:8s} air: mean {float(a.mean()):+.4f} mm^-1, standard deviation {float(a.std()):.4f} mm^-1")
