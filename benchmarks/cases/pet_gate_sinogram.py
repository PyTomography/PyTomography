"""PET sinograms: the GATE simulation of a brain phantom in a Siemens mMR (the data of t_PETGATE_scat_sino and
t_PETGATE_scat_sinoTOF).

All prompts binned into a sinogram, normalization and attenuation in the system matrix, a 3 mm Gaussian PSF, and the
smoothed delayed coincidences as the randoms estimate, on a 128 x 128 x 96 grid of 2 mm voxels.

    python benchmarks/cases/pet_gate_sinogram.py --recon osem      # non-TOF OSEM 2 x 24 (t_PETGATE_scat_sino)
    python benchmarks/cases/pet_gate_sinogram.py --recon sss       # single scatter simulation, non-TOF, steps of 4
    python benchmarks/cases/pet_gate_sinogram.py --recon osem_tof  # TOF OSEM, 21 bins (t_PETGATE_scat_sinoTOF)
"""
import gc

import torch
from _bench import Bench

import pytomography
from pytomography import io
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.metadata.PET import PETSinogramPolygonProjMeta
from pytomography.projectors.PET import PETSinogramSystemMatrix
from pytomography.transforms.shared import GaussianFilter
from pytomography.utils import sss
from pytomography.utils.memory import subsets_for_budget

bench = Bench('pet_gate_sinogram', ['osem', 'sss', 'osem_tof'], {
    'osem': 'PET sinogram (mMR, GATE brain phantom), non-TOF OSEM 2 x 24, normalization + attenuation + 3 mm PSF, '
            'randoms, 128 x 128 x 96 of 2 mm (t_PETGATE_scat_sino)',
    'sss': 'PET single scatter simulation (mMR sinogram, non-TOF), image and sinogram steps of 4 (t_PETGATE_scat_sino)',
    'osem_tof': 'PET TOF sinogram (mMR, 21 TOF bins), OSEM 2 x 28 equivalent, normalization + attenuation + 3 mm PSF, '
                'randoms, 128 x 128 x 96 of 2 mm (t_PETGATE_scat_sinoTOF)'})
import _gate_mmr as gate  # noqa: E402  (after Bench: --help works without the data)

tof = bench.recon == 'osem_tof'
tof_meta = gate.tof_meta() if tof else None
bench.setting(caches_present=gate.cached([gate.cache_name('prompts', tof_meta), gate.cache_name('delayeds'),
                                          'normalization_sinogram.pt']))

with bench.stage('load (events, attenuation, normalization, randoms)'):
    detector_ids, object_meta, proj_meta = gate.read_gate('prompts', tof_meta=tof_meta)
    info = proj_meta.info
    object_meta, atten_map = gate.grid_and_attenuation(object_meta, proj_meta)
    norm_sinogram = gate.normalization_sinogram(info)
    randoms = gate.smoothed_randoms(info)
    if tof:
        randoms = io.PET.randoms_sinogram_to_sinogramTOF(randoms, tof_meta, coincidence_timing_width=4300)
    n_events = int(detector_ids.shape[0])

with bench.stage('bin the prompts into a sinogram'):
    sinogram = io.PET.listmode_to_sinogram(detector_ids, info, tof_meta=tof_meta)
    del detector_ids
    gc.collect()

with bench.stage('system matrix and additive term'):
    sino_meta = PETSinogramPolygonProjMeta(info, tof_meta=tof_meta)
    system_matrix = PETSinogramSystemMatrix(
        object_meta, sino_meta, obj2obj_transforms=[GaussianFilter(3.)],
        sinogram_sensitivity=norm_sinogram, N_splits=10, attenuation_map=atten_map,
        device=pytomography.device if tof else 'cpu')  # as the tutorials: TOF subsets stay on the GPU
    sensitivity = system_matrix._compute_sensitivity_sinogram().cpu()
    additive_term = (randoms.unsqueeze(-1) if tof else randoms) / sensitivity
    del sensitivity
    likelihood = PoissonLogLikelihood(system_matrix, sinogram, additive_term=additive_term)

if tof:
    # the tutorial's subsets: a subset's arrays fit beside what the reconstruction holds (budget 25 GB)
    sinogram_bytes = 224 * 449 * 4096 * tof_meta.num_bins * 4
    n_subsets = subsets_for_budget(sinogram_bytes, arrays=4, held_bytes=15.5e9, minimum=28, maximum=224)
    n_iters = 1 if bench.quick else max(1, round(2 * 28 / n_subsets))
else:
    n_iters, n_subsets = (1, 24) if bench.quick else (2, 24)
bench.setting(grid=list(object_meta.shape), voxel_mm=[2.0, 2.0, 2.0], events=n_events, tof_bins=tof_meta.num_bins if tof else 0,
              algorithm='OSEM', n_iters=n_iters, n_subsets=n_subsets)

if bench.recon in ('osem', 'osem_tof'):
    bench.repeat(f'OSEM {n_iters} x {n_subsets}', lambda: OSEM(likelihood)(n_iters, n_subsets))
else:
    with bench.stage(f'OSEM {n_iters} x {n_subsets} (the image the scatter is simulated from)'):
        image = OSEM(likelihood)(n_iters, n_subsets)
    del likelihood, additive_term
    step = 8 if bench.quick else 4
    bench.setting(sss_image_step=step, sss_sinogram_steps=[step, step], attenuation_cutoff=0.004)
    bench.repeat(f'SSS (steps of {step})', lambda: sss.get_sss_scatter_estimate(
        object_meta=object_meta, proj_meta=sino_meta, pet_image=image, attenuation_image=atten_map,
        system_matrix=system_matrix, proj_data=sinogram, image_stepsize=step, attenuation_cutoff=0.004,
        sinogram_interring_stepsize=step, sinogram_intraring_stepsize=step, sinogram_random=randoms))
bench.save()
