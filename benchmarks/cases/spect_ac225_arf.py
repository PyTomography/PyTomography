"""SPECT, Ac-225: a NEMA phantom on a Siemens Symbia T2 with PSFs from Monte Carlo ARFs (the data of
t_ac225_dicom_recon).

The 218 keV photopeak with scatter from the windows beside it, attenuation from the CT, and the SY-HE collimator's
PSF from angular response functions (ARFs) that spectarf generates for this camera.

    python benchmarks/cases/spect_ac225_arf.py --recon osem   # generate the ARFs (no cache), then OSEM 100 x 3
"""
import os

from _bench import Bench

from pytomography import datasets, io
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform
from spectarf.pytomography import camera_model, generate_arfs, psf_transform

bench = Bench('spect_ac225_arf', ['osem'], {
    'osem': 'SPECT Ac-225, ARFs generated for the SY-HE collimator, then OSEM 100 x 3 with attenuation + ARF PSF '
            '(Symbia T2 NEMA phantom, t_ac225_dicom_recon)'})

datasets.fetch('SPECT/Ac225-NEMA-SymT2')
path = datasets.data_dir() / 'SPECT' / 'Ac225-NEMA-SymT2'
index_peak, index_lower, index_upper = 3, 4, 5

with bench.stage('load'):
    files_CT = [os.path.join(path / 'CT', f) for f in os.listdir(path / 'CT')]
    projections, object_meta, proj_meta = io.SPECT.read_dicom(path / 'projection_data.IMA')
    attenuation_map = io.attenuation_map(files_CT, object_meta, energy=proj_meta.energy_windows[index_peak].photon_energy)
    photopeak = projections[index_peak]
    scatter = io.SPECT.energy_window_scatter(projections, proj_meta, peak=index_peak, lower=index_lower,
                                             upper=index_upper, fwhm_theta=4.7, fwhm_r=11.3, fwhm_z=11.3)
    del projections

# The ARFs are generated here every time (cache=False), the step a new camera or isotope costs. The tutorials read
# them from spectarf's cache after their first run.
quick = dict(target_rel_error=0.2, max_passes=1) if bench.quick else {}
with bench.stage('ARFs (generated)'):
    stack, response = camera_model(io.SPECT.collimator('SY-HE'), io.SPECT.detector('NaI-9.5mm'),
                                   io.SPECT.camera('Siemens Symbia T'))
    arfs = generate_arfs(stack, response, proj_meta, peak=index_peak, cache=False, **quick)

with bench.stage('system matrix'):
    system_matrix = SPECTSystemMatrix(obj2obj_transforms=[SPECTAttenuationTransform(attenuation_map=attenuation_map),
                                                          psf_transform(arfs, normalize=True)],
                                      proj2proj_transforms=[], object_meta=object_meta, proj_meta=proj_meta)
    likelihood = PoissonLogLikelihood(system_matrix, photopeak, scatter)

n_iters, n_subsets = (1, 3) if bench.quick else (100, 3)
bench.setting(grid=list(object_meta.shape), voxel_mm=[round(float(d), 3) for d in object_meta.dr],
              views=int(photopeak.shape[0]), algorithm='OSEM', n_iters=n_iters, n_subsets=n_subsets, psf='spectarf ARF')
bench.repeat(f'OSEM {n_iters} x {n_subsets}', lambda: OSEM(likelihood)(n_iters=n_iters, n_subsets=n_subsets))
bench.save()
