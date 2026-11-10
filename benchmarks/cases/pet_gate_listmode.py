"""PET list mode: the GATE simulation of a brain phantom in a Siemens mMR (the data of t_PETGATE_scat_lm and
t_PETGATE_scat_lmTOF).

Every prompt as an event, the normalization weights of their crystal pairs, attenuation, a 3 mm Gaussian PSF, and the
smoothed delayed coincidences as the randoms estimate, on a 128 x 128 x 96 grid of 2 mm voxels.

    python benchmarks/cases/pet_gate_listmode.py --recon osem      # non-TOF list-mode OSEM 4 x 14 (t_PETGATE_scat_lm)
    python benchmarks/cases/pet_gate_listmode.py --recon osem_tof  # TOF list-mode OSEM 4 x 14, 21 bins (t_PETGATE_scat_lmTOF)
"""
import gc

from _bench import Bench

from pytomography import io
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.metadata.PET import PETLMProjMeta
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.transforms.shared import GaussianFilter

bench = Bench('pet_gate_listmode', ['osem', 'osem_tof'], {
    'osem': 'PET list mode (mMR, GATE brain phantom), non-TOF OSEM 4 x 14, normalization + attenuation + 3 mm PSF, '
            'randoms, 128 x 128 x 96 of 2 mm (t_PETGATE_scat_lm)',
    'osem_tof': 'PET TOF list mode (mMR, 21 TOF bins), OSEM 4 x 14, normalization + attenuation + 3 mm PSF, randoms, '
                '128 x 128 x 96 of 2 mm (t_PETGATE_scat_lmTOF)'})
import _gate_mmr as gate  # noqa: E402  (after Bench: --help works without the data)

tof = bench.recon == 'osem_tof'
tof_meta = gate.tof_meta() if tof else None
bench.setting(caches_present=gate.cached([gate.cache_name('prompts', tof_meta), gate.cache_name('delayeds'),
                                          gate.cache_name('trues', normalization=True)]))

with bench.stage('load (events, attenuation, normalization, randoms)'):
    weights = gate.normalization_weights()
    gc.collect()
    detector_ids, object_meta, proj_meta = gate.read_gate('prompts', tof_meta=tof_meta)
    info = proj_meta.info
    object_meta, atten_map = gate.grid_and_attenuation(object_meta, proj_meta)
    randoms = gate.smoothed_randoms(info)
    if tof:
        randoms = io.PET.randoms_sinogram_to_sinogramTOF(randoms, tof_meta=tof_meta, coincidence_timing_width=4300)
    lm_randoms = io.PET.sinogram_to_listmode(detector_ids, randoms, info)
    del randoms
    n_events = int(detector_ids.shape[0])

with bench.stage('system matrix and additive term'):
    lm_meta = PETLMProjMeta(detector_ids, info, weights_sensitivity=weights, tof_meta=tof_meta)
    system_matrix = PETLMSystemMatrix(object_meta, lm_meta, obj2obj_transforms=[GaussianFilter(3.)], N_splits=10,
                                      attenuation_map=atten_map)
    additive_term = lm_randoms / system_matrix._compute_sensitivity_projection(all_ids=False)
    additive_term[additive_term.isnan()] = 0
    del lm_randoms
    likelihood = PoissonLogLikelihood(system_matrix, additive_term=additive_term)

n_iters, n_subsets = (1, 14) if bench.quick else (4, 14)
bench.setting(grid=list(object_meta.shape), voxel_mm=[2.0, 2.0, 2.0], events=n_events, tof_bins=tof_meta.num_bins if tof else 0,
              algorithm='OSEM', n_iters=n_iters, n_subsets=n_subsets)
bench.repeat(f'OSEM {n_iters} x {n_subsets}', lambda: OSEM(likelihood)(n_iters, n_subsets))
bench.save()
