"""SPECT, Lu-177: a NEMA phantom on a Siemens Symbia T2 (the data of t_dicomdata and t_algorithms).

The 208 keV photopeak with triple-energy-window scatter, attenuation from the CT, and the SY-ME collimator's
distance-dependent PSF, on the scanner's 128 x 128 x 128 grid of 4.8 mm voxels, from 120 views.

    python benchmarks/cases/spect_lu177.py --recon osem    # OSEM, 4 iterations of 8 subsets (t_dicomdata)
    python benchmarks/cases/spect_lu177.py --recon bsrem   # BSREM, 40 x 8, relative difference prior with CT-based weights (t_algorithms)
"""
import os

from _bench import Bench

from pytomography import datasets, io
from pytomography.algorithms import BSREM, OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.priors import RelativeDifferencePrior, TopNAnatomyNeighbourWeight
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform

bench = Bench('spect_lu177', ['osem', 'bsrem'], {
    'osem': 'SPECT Lu-177, OSEM 4 x 8, attenuation + PSF, 128^3 (Symbia T2 NEMA phantom, t_dicomdata)',
    'bsrem': 'SPECT Lu-177, BSREM 40 x 8 with an RDP prior (CT-based weights), attenuation + PSF, 128^3 (t_algorithms)'})

datasets.fetch('SPECT/Lu177-NEMA-SymT2')
path = datasets.data_dir() / 'SPECT' / 'Lu177-NEMA-SymT2'

with bench.stage('load'):
    files_CT = [os.path.join(path / 'CT', f) for f in os.listdir(path / 'CT')]
    projections, object_meta, proj_meta = io.SPECT.read_dicom(path / 'projection_data.dcm')
    photopeak = projections[0]
    scatter = io.SPECT.energy_window_scatter(projections, proj_meta, peak=0, lower=1, upper=2)
    amap = io.attenuation_map(files_CT, object_meta, energy=proj_meta.energy_windows[0].photon_energy)
    psf_meta = io.SPECT.collimator_psf('SY-ME', proj_meta.energy_windows[0].photon_energy, intrinsic_fwhm=3.8)
    del projections

with bench.stage('system matrix'):
    system_matrix = SPECTSystemMatrix(obj2obj_transforms=[SPECTAttenuationTransform(amap), SPECTPSFTransform(psf_meta)],
                                      proj2proj_transforms=[], object_meta=object_meta, proj_meta=proj_meta)
    likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter)

bench.setting(grid=list(object_meta.shape), voxel_mm=[round(float(d), 3) for d in object_meta.dr],
              views=int(photopeak.shape[0]))
if bench.recon == 'osem':
    n_iters, n_subsets = (1, 8) if bench.quick else (4, 8)
    bench.setting(algorithm='OSEM', n_iters=n_iters, n_subsets=n_subsets)
    bench.repeat(f'OSEM {n_iters} x {n_subsets}', lambda: OSEM(likelihood)(n_iters=n_iters, n_subsets=n_subsets))
else:
    n_iters, n_subsets = (1, 8) if bench.quick else (40, 8)
    bench.setting(algorithm='BSREM', n_iters=n_iters, n_subsets=n_subsets, prior='RDP beta 0.3 gamma 2, top-8 CT neighbours')
    with bench.stage('prior weights'):
        weight = TopNAnatomyNeighbourWeight(amap, N_neighbours=8)

    def bsrem():
        prior = RelativeDifferencePrior(beta=0.3, gamma=2, weight=weight)
        return BSREM(likelihood=likelihood, prior=prior, relaxation_sequence=lambda n: 1 / (n / 50 + 1))(n_iters, n_subsets)
    bench.repeat(f'BSREM {n_iters} x {n_subsets}', bsrem)
bench.save()
