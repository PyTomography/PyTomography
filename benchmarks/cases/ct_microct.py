"""CT, cone beam: a laboratory micro-CT scan of glass beads on a Nikon scanner, binned by 4 (the data of t_CT_microct).

The centre of rotation is the one t_CT_microct finds from the data (0.0-0.48 mm searched); here it is searched the
same way in the load stage, so the reconstruction is the tutorial's.

    python benchmarks/cases/ct_microct.py --recon fdk    # FDK, ram-lak, 500 x 500 x 500
    python benchmarks/cases/ct_microct.py --recon sart   # OS-SART 3 x 5 from the FDK image (t_CT_microct)
"""
import numpy as np
from _bench import Bench

import pytomography
from pytomography import datasets, io
from pytomography.algorithms import SART, FilteredBackProjection
from pytomography.metadata.CT import CTConeBeamFlatPanelProjMeta
from pytomography.projectors.CT import CTConeBeamFlatPanelSystemMatrix

bench = Bench('ct_microct', ['fdk', 'sart'], {
    'fdk': 'CT cone beam (Nikon micro-CT, binned by 4), FDK, 500^3 (SophiaBeads, t_CT_microct)',
    'sart': 'CT cone beam (Nikon micro-CT, binned by 4), OS-SART 3 x 5 from the FDK image, 500^3 (t_CT_microct)'})

datasets.fetch('CT/SophiaBeads-256')
folder = datasets.data_dir() / 'CT' / 'SophiaBeads-256' / 'SophiaBeads_256_averaged'
binning = 4

with bench.stage('load'):
    proj, object_meta, proj_meta = io.CT.read_nikon(folder, binning=binning)
    proj = proj.to(pytomography.device)
n = object_meta.shape[0]


def system_matrix_for(cor, n_slices):
    meta = CTConeBeamFlatPanelProjMeta(proj_meta.angles, proj_meta.z_locations, detector_radius=proj_meta.detector_radius,
                                       beam_radius=proj_meta.beam_radius, shape=proj_meta.shape, dr=proj_meta.dr, COR=cor)
    return CTConeBeamFlatPanelSystemMatrix(object_meta.with_grid(shape=(n, n, n_slices)), meta)


def sharpness(image):
    s = image[:, :, image.shape[2] // 2]
    return float(((s[1:] - s[:-1]) ** 2).mean() + ((s[:, 1:] - s[:, :-1]) ** 2).mean())


with bench.stage('centre of rotation (25 FDKs of 4 slices)'):
    cors = np.round(np.arange(0.0, 0.5, 0.02), 2)
    scores = [sharpness(FilteredBackProjection(proj, system_matrix_for(cor, 4), filter='ram-lak')()) for cor in cors]
    best_cor = float(cors[int(np.argmax(scores))])

with bench.stage('system matrix'):
    system_matrix = system_matrix_for(best_cor, n)

n_iters, n_subsets = (1, 5) if bench.quick else (3, 5)
bench.setting(grid=[n, n, n], binning=binning, views=int(len(proj_meta.angles)), detector=list(proj.shape[1:]),
              cor_mm=best_cor)
if bench.recon == 'fdk':
    bench.setting(algorithm='FDK', window='ram-lak')
    bench.repeat('FDK', lambda: FilteredBackProjection(proj, system_matrix, filter='ram-lak')())
else:
    bench.setting(algorithm='OS-SART', n_iters=n_iters, n_subsets=n_subsets)
    with bench.stage('FDK (the starting image)'):
        start = FilteredBackProjection(proj, system_matrix, filter='ram-lak')().clamp(min=0)
    bench.repeat(f'OS-SART {n_iters} x {n_subsets}',
                 lambda: SART(system_matrix, proj, object_initial=start)(n_iters=n_iters, n_subsets=n_subsets))
bench.save()
