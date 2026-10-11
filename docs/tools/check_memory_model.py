"""Proof that the memory model of the docs (``docs/source/_static/memory/model.json``) is PyTomography's own.

For every model, over a grid of configurations (every tutorial's, realistic scanners and large ones, on a CUDA device
and on the CPU, on Windows and on Linux), this compares

* the model, evaluated by ``memory_model.py``, with the library's ``estimate_memory()`` (peak RAM and GPU memory) and
  ``plan_memory()`` (the settings chosen, and whether they fit), and
* the browser's evaluator, ``pytomo-memory.js`` run with ``node``, with ``memory_model.py``,

and reports the largest relative difference of each model. It fails (exit 1) if an estimate differs by more than 1%
or a plan chooses other settings.

It runs on the CPU and allocates no full-size array: the library's estimates are called on its own classes with their
metadata and tiny or 'meta' tensors standing in for the data (only shapes, dtypes and devices are read). Estimates for a
CUDA device are made without a GPU as PyTorch's tests of them are: ``pytomography.device`` is set to CUDA,
``torch.cuda.mem_get_info`` gives the free memory of the configuration, and tensors that would be on the GPU report it
(their ``device`` is CUDA, or they are marked). A few 'mapping checks' first build the real objects at small size
(sinogram metadata, a lazy sinogram from events, a list mode system matrix, a KEM transform) to show that the
stand-ins give the library's numbers.

Usage (Windows, Git Bash)::

    E=/c/Users/lukep/anaconda3/envs/pytomo-pp2; export PATH="$E/Library/bin:$PATH"
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<PyTomography #294>/src $E/python.exe docs/tools/check_memory_model.py [--quick] [--model ID] [--no-js]
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import pytomography                                  # noqa: E402
from pytomography.metadata import ObjectMeta         # noqa: E402
from pytomography.utils import memory                # noqa: E402
from memory_model import MemoryModel, DEFAULT_MODEL  # noqa: E402

TOLERANCE = 0.01
JS = os.path.join(HERE, '..', 'source', '_static', 'memory', 'pytomo-memory.js')
REFERENCE = os.path.join(HERE, 'memory_reference_cases.json')


# ------------------------------------------------------------------------------------------------ simulated devices
class _FakeCuda(torch.Tensor):
    """A 'meta' tensor (shape and dtype only) that says it is on a CUDA device."""
    @property
    def device(self):
        return torch.device('cuda')


class _FakeCpu(torch.Tensor):
    """A 'meta' tensor (shape and dtype only) that says it is on the CPU."""
    @property
    def device(self):
        return torch.device('cpu')


def fake(shape, dtype=torch.float32, on_gpu: bool = False) -> torch.Tensor:
    """A stand-in for an array of ``shape``: no memory, but its storage size, dtype and device are those of the array."""
    base = torch.empty(tuple(int(s) for s in shape), dtype=dtype, device='meta')
    return torch.Tensor._make_subclass(_FakeCuda if on_gpu else _FakeCpu, base)


def mark(tensor: torch.Tensor, on_gpu: bool) -> torch.Tensor:
    """Marks a real (CPU) tensor as one a CUDA run keeps on the GPU, for the simulated estimates."""
    tensor._sim_cuda = bool(on_gpu)
    return tensor


_ORIGINAL_NBYTES = memory.nbytes


def _nbytes(x):
    """``memory.nbytes``, with tensors marked by :func:`mark` counted on the GPU."""
    if isinstance(x, torch.Tensor) and getattr(x, '_sim_cuda', False):
        return 0.0, float(x.untyped_storage().nbytes())
    return _ORIGINAL_NBYTES(x)


def _patch_nbytes():
    for name, module in list(sys.modules.items()):
        if name.startswith('pytomography') and getattr(module, 'nbytes', None) is _ORIGINAL_NBYTES:
            module.nbytes = _nbytes


@contextlib.contextmanager
def simulate(p: dict):
    """PyTomography as a configuration ``p`` sees it: its device, operating system, free GPU memory, memory budget and
    whether CuPy's fused kernels can run."""
    from pytomography.utils import _sss_cuda, sss
    from pytomography.projectors.CT import _wfbp_cuda
    saved = (pytomography.device, pytomography.memory_budget, memory.counts_gpu_as_ram, torch.cuda.mem_get_info,
             _sss_cuda.available, _wfbp_cuda.available)
    cuda = p.get('device', 'cuda') == 'cuda'
    cupy = bool(p.get('cupy', True))
    pytomography.device = torch.device('cuda' if cuda else 'cpu')
    budget = p.get('memory_budget_gb')
    pytomography.memory_budget = None if budget is None else float(budget) * 1e9
    windows = p.get('os', 'windows') == 'windows'
    memory.counts_gpu_as_ram = lambda: windows
    free = float(p.get('gpu_free_gb', 30)) * 1e9
    torch.cuda.mem_get_info = lambda device=None: (free, 96e9)
    _sss_cuda.available = lambda device, num_tof_bins: torch.device(device).type == 'cuda' and num_tof_bins <= _sss_cuda.MAX_TOF_BINS and cupy
    _wfbp_cuda.available = lambda device: torch.device(device).type == 'cuda' and cupy
    _patch_nbytes()
    try:
        yield
    finally:
        (pytomography.device, pytomography.memory_budget, memory.counts_gpu_as_ram, torch.cuda.mem_get_info,
         _sss_cuda.available, _wfbp_cuda.available) = saved


def _algorithm(name: str, system_matrix, projections=None, additive=None, prior: bool = False):
    """A reconstruction algorithm of the library around ``system_matrix``, without running its constructor (which makes
    an initial image): what ``estimate_memory`` and ``plan_memory`` read is its likelihood (system matrix, projections,
    additive term) and its prior."""
    from pytomography import algorithms
    from pytomography.priors import RelativeDifferencePrior
    cls = getattr(algorithms, name)
    algorithm = cls.__new__(cls)
    algorithm.likelihood = SimpleNamespace(system_matrix=system_matrix, projections=projections, additive_term=additive)
    algorithm.prior = RelativeDifferencePrior(beta=1.0) if prior else None
    return algorithm


def _anatomy_weight(p):
    """A TopNAnatomyNeighbourWeight as its prior's estimate reads it (its tensors, checked against a real one in
    :func:`mapping_checks`): the anatomical image, where the configuration keeps it, and which of the 27 neighbours
    count, a byte each on the host."""
    from pytomography.priors.nearest_neighbour import EuclideanNeighbourWeight
    shape = (int(p['Lx']), int(p['Ly']), int(p['Lz']))
    return SimpleNamespace(eucliden_neighbour_weight=EuclideanNeighbourWeight(), anatomy_image=fake(shape, on_gpu=p['device'] == 'cuda'),
                           N=8, inclusion_image=fake((3, 3, 3, *shape), torch.bool))


def _with_prior_weight(algorithm, p):
    if algorithm.prior is not None and p.get('prior_weight') == 'top_n_anatomy':
        algorithm.prior.weight = _anatomy_weight(p)
    return algorithm


def _result(estimate) -> dict:
    return {'ram_gb': estimate.ram_gb, 'gpu_gb': estimate.gpu_gb, 'settings': dict(estimate.settings),
            'parts': [(p.name, p.scope, p.ram_bytes / 1e9, p.gpu_bytes / 1e9) for p in estimate.parts]}


def _plan_result(plan) -> dict:
    return {'settings': {k: (v.item() if hasattr(v, 'item') else v) for k, v in plan.settings.items()}, 'fits': plan.fits,
            'ram_gb': plan.estimate.ram_gb, 'gpu_gb': plan.estimate.gpu_gb, 'tried': len(plan.options)}


# ------------------------------------------------------------------------------------------------------------- PET
def _pet_shape(p) -> list:
    """Sinogram shape (angles, radial bins, planes) as PETSinogramPolygonProjMeta computes it (checked in
    :func:`mapping_checks`)."""
    C, R = int(p['crystals_per_ring']), int(p['rings'])
    return [C // 2, C + 1, R * R]


def _pet_info(C: int, R: int, crystals_trans: int = 8, crystals_axial: int = 8) -> dict:
    """A scanner geometry dictionary of ``C`` crystals per ring and ``R`` rings, in blocks of ``crystals_trans`` by
    ``crystals_axial`` crystals (the mMR's layout for C=448, R=64)."""
    return dict(min_rsector_difference=0, crystal_length=20.0, radius=328.0, firstCrystalAxis=0,
                rsectorTransNr=C // crystals_trans, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=R // crystals_axial,
                moduleTransSpacing=0.0, moduleAxialSpacing=32.25, submoduleTransNr=1, submoduleAxialNr=1,
                submoduleTransSpacing=0.0, submoduleAxialSpacing=0.0, crystalTransNr=crystals_trans,
                crystalAxialNr=crystals_axial, crystalTransSpacing=4.0, crystalAxialSpacing=4.0,
                NrCrystalsPerRing=C, NrRings=R)


def _sss_rtheta(p, shape, T) -> float:
    """Host memory of a lazy SSS estimate, from the library's own estimate of it (its interpolation over angle and
    radius of the sampled ring pairs)."""
    from pytomography.utils import sss
    step = int(p.get('sss_step', 4))
    proj_meta = SimpleNamespace(info={'NrCrystalsPerRing': int(p['crystals_per_ring']), 'NrRings': int(p['rings'])})
    sizes = sss._sss_sizes(proj_meta, SimpleNamespace(num_bins=T) if T > 1 or p.get('tof') else None, step, step)
    return 4.0 * sizes['N_theta'] * sizes['N_r'] * sizes['sampled_planes'] * sizes['T']


def build_pet_sinogram(p):
    """OSEM (or the algorithm of ``p``) of a PETSinogramSystemMatrix, its data and additive term as the tutorials make
    them: a dense sinogram or a lazy one binned from the events; randoms and a lazy SSS scatter estimate divided by the
    sensitivity."""
    from pytomography.projectors.PET import PETSinogramSystemMatrix
    from pytomography.io.PET.shared import LazySinogram
    shape = _pet_shape(p)
    tof = bool(p['tof'])
    T = int(p['tof_bins']) if tof else 1
    cuda = p['device'] == 'cuda'
    sm = PETSinogramSystemMatrix.__new__(PETSinogramSystemMatrix)
    sm.object_meta = ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(int(p['Lx']), int(p['Ly']), int(p['Lz'])))
    sm.proj_meta = SimpleNamespace(N_angles=shape[0], shape=shape, tof_meta=SimpleNamespace(num_bins=T) if tof else None)
    sm.TOF = tof
    sm.output_device = p['device'] if p['proj_device'] == 'device' else 'cpu'
    out_gpu = torch.device(sm.output_device).type == 'cuda'
    sm.sinogram_sensitivity = fake(shape, on_gpu=out_gpu) if p['sensitivity'] else None
    sm.N_splits = int(p['N_splits'])
    full = shape + [T] if tof else shape
    if p['data'] == 'lazy':
        data = LazySinogram(None, full, memory_bytes=(6 if tof else 4) * int(p['events']) + 8 * (shape[0] + 1))
    else:
        data = fake(full, on_gpu=p['data'] == 'dense_gpu' and True)
    additive = None
    if p['additive'] == 'dense':
        additive = fake(full, on_gpu=p['data'] == 'dense_gpu')
    elif p['additive'] == 'lazy':
        scatter = LazySinogram(None, full, memory_bytes=_sss_rtheta(p, shape, T))
        randoms = fake(shape)
        sensitivity = fake(shape + [1] if tof else shape)
        additive = ((randoms.unsqueeze(-1) if tof else randoms) + scatter) / sensitivity
    return _with_prior_weight(_algorithm(p['algorithm'], sm, data, additive, p['prior']), p)


def _pet_sinogram_settings(p):
    shape = _pet_shape(p)
    return [{'N_splits': int(p['N_splits']), 'n_subsets': 1}, {'N_splits': 10, 'n_subsets': 28},
            {'N_splits': 1, 'n_subsets': shape[0]}, {'N_splits': 3, 'n_subsets': 7}]


def _listmode_system_matrix(p):
    """A PETLMSystemMatrix as the library's estimate reads it: the events' detector IDs (their number, columns, dtype
    and device), the sensitivity weights of every crystal pair, the scanner geometry being known, the output device
    and N_splits (checked against a real one in :func:`mapping_checks`)."""
    from pytomography.projectors.PET import PETLMSystemMatrix
    cuda = p['device'] == 'cuda'
    E = int(p['events'])
    cols = 3 if p['tof'] else 2
    dtype = torch.int64 if int(p['id_bytes']) == 8 else torch.int32
    crystals = int(p['crystals_per_ring']) * int(p['rings'])
    sm = PETLMSystemMatrix.__new__(PETLMSystemMatrix)
    sm.object_meta = ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(int(p['Lx']), int(p['Ly']), int(p['Lz'])))
    pairs = crystals * (crystals - 1) // 2 if p.get('lm_weight_pairs') is None else int(p['lm_weight_pairs'])
    weights = None if p['lm_weights'] == 'none' else fake((pairs,), on_gpu=p['lm_weights'] == 'gpu')
    kind = p.get('lm_sensitivity_ids', 'none')
    sensitivity_ids = None if kind == 'none' else fake((pairs, 2), torch.int64 if kind == 'int64' else torch.int32)   # on the host
    sm.proj_meta = SimpleNamespace(detector_ids=fake((E, cols), dtype, on_gpu=cuda and p['lor_device'] == 'device'),
                                   weights_sensitivity=weights, detector_ids_sensitivity=sensitivity_ids,
                                   info={'NrCrystalsPerRing': int(p['crystals_per_ring'])}, shape=(E,))
    sm.sort_events = bool(p['sort_events'])
    sm.output_device = p['device'] if p['proj_device'] == 'device' else 'cpu'
    sm.N_splits = int(p['N_splits'])
    return sm


def build_pet_listmode(p):
    """OSEM (or the algorithm of ``p``) of a PETLMSystemMatrix; the likelihood's projections are the one-element tensor
    a list mode likelihood makes, on the device, and its additive term one value per event, moved there too."""
    cuda = p['device'] == 'cuda'
    sm = _listmode_system_matrix(p)
    additive = fake((int(p['events']),), on_gpu=cuda) if p['lm_additive'] else None
    return _with_prior_weight(_algorithm(p['algorithm'], sm, fake((1,), on_gpu=cuda), additive, p['prior']), p)


def build_sss(p):
    """The arguments of ``sss.estimate_sss_memory`` and ``sss.plan_sss_memory`` for a configuration: the scanner
    (``proj_meta.info``), the data (a sinogram, dense or lazy, with its system matrix and sensitivity sinogram; or list
    mode events with their PETLMSystemMatrix and sensitivity weights), the randoms and the TOF metadata."""
    from pytomography.projectors.PET import PETSinogramSystemMatrix, PETLMSystemMatrix
    from pytomography.io.PET.shared import LazySinogram
    shape = _pet_shape(p)
    tof = bool(p['tof'])
    T = int(p['tof_bins']) if tof else 1
    info = {'NrCrystalsPerRing': int(p['crystals_per_ring']), 'NrRings': int(p['rings'])}
    full = shape + [T] if tof else shape
    args = dict(object_meta=ObjectMeta(dr=(2.0, 2.0, 2.0), shape=(int(p['Lx']), int(p['Ly']), int(p['Lz']))),
                tof_meta=SimpleNamespace(num_bins=T) if tof else None,
                sinogram_random=fake(shape) if p['sss_randoms'] else None,
                sinogram_interring_stepsize=int(p['sss_interring_step']), sinogram_intraring_stepsize=int(p['sss_intraring_step']),
                num_dense_tof_bins=int(p['num_dense_tof_bins']))
    if p['sss_input'] == 'listmode':
        sm = PETLMSystemMatrix.__new__(PETLMSystemMatrix)
        args['proj_meta'] = SimpleNamespace(info=info, detector_ids=fake((int(p['events']), 3 if tof else 2), torch.int32),
                                            weights_sensitivity=fake((4,)) if p['sensitivity'] else None)
        args['proj_data'] = None
    else:
        sm = PETSinogramSystemMatrix.__new__(PETSinogramSystemMatrix)
        out_gpu = p['device'] == 'cuda' and p['proj_device'] == 'device'
        sm.sinogram_sensitivity = fake(shape, on_gpu=out_gpu) if p['sensitivity'] else None
        args['proj_meta'] = SimpleNamespace(info=info)
        if p['data'] == 'lazy':
            args['proj_data'] = LazySinogram(None, full, memory_bytes=(6 if tof else 4) * int(p['events']) + 8 * (shape[0] + 1))
        else:
            args['proj_data'] = fake(full, on_gpu=p['data'] == 'dense_gpu')
    args['system_matrix'] = sm
    return args


def _estimate_sss(args, settings):
    from pytomography.utils import sss
    return _result(sss.estimate_sss_memory(**args, **settings))


def _plan_sss(args, ram_gb, gpu_gb, fixed):
    from pytomography.utils import sss
    return _plan_result(sss.plan_sss_memory(**args, ram_gb=ram_gb, gpu_gb=gpu_gb, N_splits=fixed.get('N_splits'), lazy=fixed.get('lazy')))


def _sss_settings(p):
    out = [{'lazy': None, 'N_splits': 1}, {'lazy': True, 'N_splits': 1}, {'lazy': False, 'N_splits': 1}]
    if p['tof']:
        out += [{'lazy': True, 'N_splits': 3}, {'lazy': None, 'N_splits': int(p['tof_bins'])}, {'lazy': False, 'N_splits': 100}]
    return out


# ------------------------------------------------------------------------------------------------------------ CT
def _ct_stand_in(cls, p):
    """A CT system matrix as its ``_memory_parts`` read it: the object grid, the detector shape and number of views,
    N_splits and the device its projections are output on."""
    sm = cls.__new__(cls)
    sm.object_meta = ObjectMeta(dr=(1.0, 1.0, 1.0), shape=(int(p['Lx']), int(p['Ly']), int(p['Lz'])))
    sm.proj_meta = SimpleNamespace(shape=(int(p['det_cols']), int(p['det_rows'])), N_angles=int(p['views']))
    sm.N_splits = int(p.get('N_splits', 1))
    sm.device = p['device'] if p.get('proj_device', 'device') == 'device' else 'cpu'
    return sm


def _ct_projections(p):
    return fake((int(p['views']), int(p['det_cols']), int(p['det_rows'])), on_gpu=p['ct_data'] == 'gpu' and p['device'] == 'cuda')


def build_ct_helical_sart(p):
    from pytomography.projectors.CT import CTGen3SystemMatrix
    return _algorithm('SART', _ct_stand_in(CTGen3SystemMatrix, p), _ct_projections(p))


def build_ct_cone_sart(p):
    from pytomography.projectors.CT import CTConeBeamFlatPanelSystemMatrix
    return _algorithm('SART', _ct_stand_in(CTConeBeamFlatPanelSystemMatrix, p), _ct_projections(p))


def _ct_sart_settings(p):
    return [{'N_splits': int(p['N_splits']), 'n_subsets': 1}, {'N_splits': 1, 'n_subsets': 8}, {'N_splits': 4, 'n_subsets': 5},
            {'N_splits': 1, 'n_subsets': min(256, int(p['views']))}]


def build_ct_cone_fdk(p):
    from pytomography.algorithms import FilteredBackProjection
    from pytomography.projectors.CT import CTConeBeamFlatPanelSystemMatrix
    return FilteredBackProjection(_ct_projections(p), _ct_stand_in(CTConeBeamFlatPanelSystemMatrix, p), filter='ram-lak')


def _helix_meta(p):
    """A helical scan with the geometry of ``p`` (one focal spot), laid out as the DICOM-CT-PD reader makes it."""
    from pytomography.metadata.CT import CTGen3ProjMeta
    n, V = int(p['views']), int(p['views_per_rotation'])
    phis = torch.arange(n, dtype=torch.float64) * (2 * np.pi / V)
    zs = float(p['feed']) / (2 * np.pi) * phis
    zero = torch.zeros(n)
    cols, rows = int(p['det_cols']), int(p['det_rows'])
    central = (cols + 1) / 2 if p.get('central_col') is None else float(p['central_col'])
    return CTGen3ProjMeta(phis.float(), torch.full((n,), float(p['focal_radius'])), (zs - zs.mean()).float(), zero, zero, zero,
                          torch.full((n,), central), torch.full((n,), (rows + 1) / 2), float(p['col_spacing_mm']) / float(p['dsd']),
                          float(p['row_spacing_mm']), float(p['dsd']), shape=(cols, rows))


def build_ct_helical_fbp(p):
    """Filtered back projection of a CTGen3SystemMatrix built for real (on the CPU) on the scan's metadata, with
    projections that are a stand-in of their shape and place."""
    from pytomography.projectors.CT import CTGen3SystemMatrix
    device = pytomography.device
    pytomography.device = torch.device('cpu')                 # built on the CPU; the estimate is for the configuration's device
    try:
        sm = CTGen3SystemMatrix(ObjectMeta(dr=(float(p['dx']), float(p['dy']), 1.0), shape=(int(p['Lx']), int(p['Ly']), int(p['Lz']))),
                                _helix_meta(p))
    finally:
        pytomography.device = device
    return SimpleNamespace(sm=sm, projections=_ct_projections(p), slice_thickness=float(p['slice_thickness']) or None)


def _estimate_ct_helical_fbp(obj, settings):
    from pytomography.algorithms import FilteredBackProjection
    from pytomography.utils.memory import memory_budget_set
    with memory_budget_set(settings['memory_budget_gb']):
        fbp = FilteredBackProjection(obj.projections, obj.sm, filter='hann', slice_thickness=obj.slice_thickness, gpu_budget=settings['gpu_budget'])
        return _result(fbp.estimate_memory())


def _plan_ct_helical_fbp(obj, ram_gb, gpu_gb, fixed):
    from pytomography.algorithms import FilteredBackProjection
    fbp = FilteredBackProjection(obj.projections, obj.sm, filter='hann', slice_thickness=obj.slice_thickness)
    return _plan_result(fbp.plan_memory(ram_gb=ram_gb, gpu_gb=gpu_gb, **fixed))


def _ct_helical_fbp_settings(p):
    b = p['memory_budget_gb']
    return [{'gpu_budget': 1.5e9, 'memory_budget_gb': b}, {'gpu_budget': 0.75e9, 'memory_budget_gb': 2.0},
            {'gpu_budget': 0.1875e9, 'memory_budget_gb': 0.06}, {'gpu_budget': 4e9, 'memory_budget_gb': None}]


def _estimate_fbp(fbp, settings):
    return _result(fbp.estimate_memory())


# ---------------------------------------------------------------------------------------------------------- SPECT
@contextlib.contextmanager
def _on_cpu():
    """Builds real objects on the CPU, whatever device the configuration's estimate is for."""
    device = pytomography.device
    pytomography.device = torch.device('cpu')
    try:
        yield
    finally:
        pytomography.device = device


def _mark_transforms(transforms, on_gpu: bool):
    """Marks the tensors SPECT transforms keep as on the GPU when the configuration runs there (they are built on the
    CPU here; a CUDA run builds them on the device)."""
    from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
    from pytomography.transforms.SPECT.psf import Seperable1DBlurNet, PSF2D
    for t in transforms:
        if isinstance(t, SPECTAttenuationTransform):
            mark(t.attenuation_map, on_gpu)
        elif isinstance(t, SPECTPSFTransform):
            for layer in t.layers.values():
                if isinstance(layer, Seperable1DBlurNet):
                    for conv in (layer.layer_r, layer.layer_z):
                        if conv is not None:
                            mark(conv.weight, on_gpu)
                elif isinstance(layer, PSF2D):
                    for value in vars(layer).values():
                        if isinstance(value, torch.Tensor):
                            mark(value, on_gpu)


def _psf_kernels(transforms) -> list:
    """The 1D kernel sizes of the Gaussian PSF layers, per radius: (radial, axial)."""
    from pytomography.transforms.SPECT import SPECTPSFTransform
    from pytomography.transforms.SPECT.psf import Seperable1DBlurNet
    out = []
    for t in transforms:
        if isinstance(t, SPECTPSFTransform):
            for layer in t.layers.values():
                if isinstance(layer, Seperable1DBlurNet):
                    out.append((int(layer.layer_r.weight.shape[-1]), int(layer.layer_z.weight.shape[-1]) if layer.layer_z is not None else 0))
    return out


def _spect_system_matrix(p):
    """A real SPECTSystemMatrix of the configuration (attenuation, Gaussian PSF or a PSF operator), built on the CPU;
    its Gaussian kernel sizes go into ``p['psf_kernel']`` (their mean, which is what the model's parameter is)."""
    from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta
    from pytomography.projectors.SPECT import SPECTSystemMatrix
    from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
    Lx, Ly, Lz, views = int(p['Lx']), int(p['Ly']), int(p['Lz']), int(p['views'])
    Lr = Ly if p.get('proj_r') is None else int(p['proj_r'])
    Lzp = Lz if p.get('proj_z') is None else int(p['proj_z'])
    with _on_cpu():
        object_meta = SPECTObjectMeta([0.48] * 3, (Lx, Ly, Lz))
        radii = 22.0 + (np.arange(views) % int(p['n_radii'])) * 0.5
        proj_meta = SPECTProjMeta((Lr, Lzp), [0.48, 0.48], list(np.linspace(0, 360, views, endpoint=False)), radii=radii)
        transforms = []
        if p['attenuation']:
            transforms.append(SPECTAttenuationTransform(torch.zeros(Lx, Ly, Lz), cache_probabilities=bool(p['cache_probabilities'])))
        if p['psf'] == 'gaussian':
            transforms.append(SPECTPSFTransform(SPECTPSFMeta((0.03, 0.1))))
        elif p['psf'] == 'operator':
            transforms.append(SPECTPSFTransform(psf_operator=object(), cache_kernel=bool(p['cache_kernel'])))
        sm = SPECTSystemMatrix(transforms, [], object_meta, proj_meta, object_initial_based_on_camera_path=bool(p['camera_path']))
    _mark_transforms(transforms, p['device'] == 'cuda')
    kernels = _psf_kernels(transforms)
    if kernels:
        p['psf_kernel'] = sum(r + z for r, z in kernels) / (2 * len(kernels))
    return sm


def _spect_data(p):
    Lr = int(p['Ly']) if p.get('proj_r') is None else int(p['proj_r'])
    Lzp = int(p['Lz']) if p.get('proj_z') is None else int(p['proj_z'])
    cuda = p['device'] == 'cuda'
    projections = fake((int(p['views']), Lr, Lzp), on_gpu=cuda)
    return projections, (fake((int(p['views']), Lr, Lzp), on_gpu=cuda) if p['spect_additive'] else None)


def build_spect(p):
    sm = _spect_system_matrix(p)
    projections, additive = _spect_data(p)
    return _with_prior_weight(_algorithm(p['algorithm'], sm, projections, additive, p['prior']), p)


def build_spect_fbp(p):
    from pytomography.algorithms import FilteredBackProjection
    q = dict(p, attenuation=False, psf='none', camera_path=False, n_radii=1, cache_probabilities=False, cache_kernel=False, spect_additive=False)
    return FilteredBackProjection(_spect_data(q)[0], _spect_system_matrix(q), filter='hann')


def build_kem(p):
    """KEM of a KEMSystemMatrix around a real SPECTSystemMatrix; the kernel transform is a stand-in of the arrays it
    keeps (its kernel, size**3 images; its support images), checked against a real one in :func:`mapping_checks`."""
    from pytomography.projectors.shared import KEMSystemMatrix
    sm = _spect_system_matrix(p)
    cuda = p['device'] == 'cuda'
    shape = (int(p['Lx']), int(p['Ly']), int(p['Lz']))
    s = int(p['kem_size'])
    kem = KEMSystemMatrix.__new__(KEMSystemMatrix)
    kem.system_matrix, kem.object_meta, kem.proj_meta = sm, sm.object_meta, sm.proj_meta
    kem.kem_transform = SimpleNamespace(kernel=fake((s, s, s, *shape), on_gpu=cuda and p['kem_kernel_on_gpu']),
                                        support_objects=[fake(shape, on_gpu=cuda and p['kem_support_on'] == 'device') for _ in range(int(p['kem_support']))])
    projections, additive = _spect_data(p)
    return _algorithm('KEM', kem, projections, additive)


def build_spect_multibed(p):
    from pytomography.algorithms import PGAAMultiBedSPECT
    sm = _spect_system_matrix(p)
    beds = []
    for _ in range(int(p['n_beds'])):
        projections, additive = _spect_data(p)
        beds.append(_with_prior_weight(_algorithm(p['algorithm'], sm, projections, additive, p['prior']), p))
    return PGAAMultiBedSPECT([f'bed {k}' for k in range(len(beds))], beds)


def build_starguide(p):
    """OSEM (or the algorithm of ``p``) of a real StarGuideSystemMatrix (built on the CPU): ``sg_angles`` angles of
    ``sg_views_per_angle`` views, ``n_radii`` distinct radii, attenuation and a Gaussian PSF as the tutorial makes
    them. ``p['psf_kernel']`` becomes the largest kernel, the model's parameter."""
    from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTPSFMeta, StarGuideProjMeta
    from pytomography.projectors.SPECT import StarGuideSystemMatrix
    from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
    Lx, Ly, Lz = int(p['Lx']), int(p['Ly']), int(p['Lz'])
    A, B = int(p['sg_angles']), int(p['sg_views_per_angle'])
    V = A * B if p.get('sg_views') is None else int(p['sg_views'])
    # the first angle has the most views, B; the others share the rest (as the scanner's irregular angles do)
    rest = np.full(A - 1, (V - B) // max(A - 1, 1)) + (np.arange(A - 1) < (V - B) % max(A - 1, 1))
    counts = np.concatenate([[B], rest]).astype(int) if A > 1 else np.array([V])
    assert counts.max() == B and counts.sum() == V and counts.min() >= 1, 'views per angle'
    gen = torch.Generator().manual_seed(0)
    with _on_cpu():
        object_meta = SPECTObjectMeta((0.25, 0.25, 0.25), (Lx, Ly, Lz))
        angles = torch.tensor(np.repeat(np.linspace(0, 360, A, endpoint=False), counts), dtype=torch.float32)
        offsets = (torch.rand(V, generator=gen) * 2 - 1) * 8
        radii = 15.0 + (np.arange(V) % int(p['n_radii'])) * 0.05
        proj_meta = StarGuideProjMeta((V, 16, Lz), angles, 0.5 + torch.rand(V, generator=gen), offsets, radii)
        transforms = []
        if p['attenuation']:
            transforms.append(SPECTAttenuationTransform(torch.zeros(Lx, Ly, Lz), assume_padded=False, cache_probabilities=bool(p['cache_probabilities'])))
        if p['psf'] != 'none':
            transforms.append(SPECTPSFTransform(SPECTPSFMeta((0.02, 0.05)), assume_padded=False))
        sm = StarGuideSystemMatrix(object_meta, proj_meta, obj2obj_transforms=transforms)
    _mark_transforms(transforms, p['device'] == 'cuda')
    kernels = _psf_kernels(transforms)
    if kernels:
        p['psf_kernel'] = max(max(r, z) for r, z in kernels)
        p['psf_kernel_mean'] = sum(r + z for r, z in kernels) / (2 * len(kernels))
    cuda = p['device'] == 'cuda'
    additive = fake((V, 16, Lz), on_gpu=cuda) if p['spect_additive'] else None
    return _with_prior_weight(_algorithm(p['algorithm'], sm, fake((V, 16, Lz), on_gpu=cuda), additive, p['prior']), p)


def build_spect_extended(p):
    """OSEM (or the algorithm of ``p``) of an ExtendedSystemMatrix of ``ext_n`` SPECT system matrices (each its own,
    as for a photopeak each, or the same one), or of a MotionSystemMatrix with a deformation field pair per gate; the
    projections and additive term are stacked, one set per system matrix."""
    from pytomography.projectors import ExtendedSystemMatrix
    from pytomography.projectors.shared import MotionSystemMatrix
    from pytomography.transforms.shared import DVFMotionTransform
    N = int(p['ext_n'])
    first = _spect_system_matrix(p)
    sms = [first] + [(_spect_system_matrix(p) if p['ext_distinct'] else first) for _ in range(N - 1)]
    with _on_cpu():
        if p['ext_motion']:
            shape = (int(p['Lx']), int(p['Ly']), int(p['Lz']), 3)
            transforms = [DVFMotionTransform(torch.zeros(shape), torch.zeros(shape)) for _ in range(N)]
            sm = MotionSystemMatrix(sms, transforms)
        else:
            transforms = []
            sm = ExtendedSystemMatrix(sms)
    for t in transforms:
        mark(t.dvf_forward, p['device'] == 'cuda')
        mark(t.dvf_backward, p['device'] == 'cuda')
    projections, additive = _spect_data(p)
    stack = lambda x: None if x is None else fake((N, *x.shape), on_gpu=p['device'] == 'cuda')
    return _with_prior_weight(_algorithm(p['algorithm'], sm, stack(projections), stack(additive), p['prior']), p)


def build_spect_simind(p):
    """OSEM (or the algorithm of ``p``) of a MonteCarloHybridSPECTSystemMatrix around the configuration's SPECT
    system matrix (SIMIND is not needed to estimate: its path is not looked up)."""
    from pytomography.projectors.SPECT import dualhead_system_matrix, MonteCarloHybridSPECTSystemMatrix
    p['camera_path'] = True                              # this system matrix always starts from the camera path
    base = _spect_system_matrix(p)
    which = dualhead_system_matrix.shutil.which
    dualhead_system_matrix.shutil.which = lambda name: 'simind'
    try:
        with _on_cpu():
            amap = torch.zeros(base.object_meta.shape)
            sm = MonteCarloHybridSPECTSystemMatrix(base.object_meta, base.proj_meta, n_events=10 ** 6, n_parallel=int(p['simind_parallel']),
                                                   obj2obj_transforms=base.obj2obj_transforms, proj2proj_transforms=[],
                                                   attenuation_map_140keV=amap, energy_window_params=['w'] * int(p['simind_windows']),
                                                   primary_window_idx=0, isotope_names=['lu177'], isotope_ratios=[1.0], collimator_type='SY-ME',
                                                   crystal_thickness=0.95, cover_thickness=0.1, backscatter_thickness=5.0)
    finally:
        dualhead_system_matrix.shutil.which = which
    mark(amap, p['device'] == 'cuda')
    _mark_transforms(sm.obj2obj_transforms, p['device'] == 'cuda')   # configured again by this system matrix
    projections, additive = _spect_data(p)
    return _with_prior_weight(_algorithm(p['algorithm'], sm, projections, additive, p['prior']), p)


def _spect_settings(p):
    v = int(p.get('views', p.get('sg_angles', 1)))
    return [{'n_subsets': 1}, {'n_subsets': 8}, {'n_subsets': 3}, {'n_subsets': min(256, v)}]


def _starguide_settings(p):
    return [{'n_subsets': 1}, {'n_subsets': 10}, {'n_subsets': 5}, {'n_subsets': min(256, int(p['sg_angles']))}]


def _pet_listmode_settings(p):
    return [{'N_splits': int(p['N_splits']), 'n_subsets': 1}, {'N_splits': 10, 'n_subsets': 14},
            {'N_splits': 1, 'n_subsets': 256}, {'N_splits': 3, 'n_subsets': 21}]


# --------------------------------------------------------------------------------------------------------- generic
def _estimate_algorithm(obj, settings):
    return _result(obj.estimate_memory(**settings))


def _plan_algorithm(obj, ram_gb, gpu_gb, fixed):
    return _plan_result(obj.plan_memory(ram_gb=ram_gb, gpu_gb=gpu_gb, **(fixed or {})))


class Checker:
    """How one model is checked: the library object of a configuration, its estimate and plan, the configurations, the
    settings to estimate and the budgets to plan for."""
    def __init__(self, model_id, build, grid, settings, estimate=_estimate_algorithm, plan=_plan_algorithm,
                 budgets=((16, 8), (25, 24), (64, 32), (8, 2), (1000, 1000)), fixed=({},)):
        self.model_id, self.build, self.grid, self.settings = model_id, build, grid, settings
        self.estimate, self.plan, self.budgets, self.fixed = estimate, plan, budgets, fixed


def _sample(base: dict, axes: dict, n: int, seed: int) -> list:
    """``n`` configurations: ``base`` with values of ``axes`` drawn at random (every axis value appears at least once
    when ``n`` allows)."""
    rng = random.Random(seed)
    names = list(axes)
    out = []
    for i in range(n):
        p = dict(base)
        for name in names:
            values = axes[name]
            p[name] = values[i % len(values)] if i < max(len(v) for v in axes.values()) else rng.choice(values)
        out.append(p)
    return out


# --------------------------------------------------------------------------------------------------- the grids
DEVICES = {'device': ['cuda', 'cpu'], 'os': ['windows', 'linux']}


def grid_pet_sinogram(quick: bool) -> list:
    base = dict(device='cuda', os='windows', gpu_free_gb=30, memory_budget_gb=25, Lx=128, Ly=128, Lz=96, crystals_per_ring=448,
                rings=64, tof=True, tof_bins=21, proj_device='device', sensitivity=True, data='lazy', events=50_000_000,
                additive='lazy', sss_step=6, algorithm='OSEM', prior=False, N_splits=10, n_subsets=28)
    tutorials = [dict(base),                                                        # t_PETGATE_scat_sinoTOF
                 dict(base, tof=False, proj_device='cpu', data='dense_ram', additive='dense', N_splits=10, n_subsets=24)]  # t_PETGATE_scat_sino
    axes = dict(DEVICES, gpu_free_gb=[30, 4, 11.5], memory_budget_gb=[None, 25, 8],
                scanner=[(448, 64), (504, 64), (64, 8), (760, 80), (640, 120)], image=[(128, 128, 96), (200, 200, 127), (440, 440, 159), (64, 64, 32)],
                tof=[False, True], tof_bins=[21, 13, 33], proj_device=['device', 'cpu'], sensitivity=[True, False],
                data=['dense_ram', 'dense_gpu', 'lazy'], additive=['none', 'dense', 'lazy'], sss_step=[4, 6, 2],
                algorithm=['OSEM', 'BSREM', 'MLEM', 'OSMAPOSL', 'RBIEM'], prior=[False, True], prior_weight=['euclidean', 'top_n_anatomy'], N_splits=[1, 10, 3],
                events=[50_000_000, 2_000_000, 400_000_000])
    configs = tutorials + _sample(base, axes, 12 if quick else 60, 1)
    for p in configs:
        if 'scanner' in p:
            p['crystals_per_ring'], p['rings'] = p.pop('scanner')
        if 'image' in p:
            p['Lx'], p['Ly'], p['Lz'] = p.pop('image')
    return configs


def grid_pet_listmode(quick: bool) -> list:
    base = dict(device='cuda', os='windows', gpu_free_gb=30, memory_budget_gb=25, Lx=128, Ly=128, Lz=96, crystals_per_ring=448,
                rings=64, tof=False, events=50_000_000, id_bytes=4, lor_device='device', proj_device='device', sort_events=True,
                lm_weights='ram', lm_additive=True, algorithm='OSEM', prior=False, N_splits=10, n_subsets=14)
    tutorials = [dict(base, events=97_000_000),                                         # t_PETGATE_scat_lm
                 dict(base, tof=True, events=97_000_000),                               # t_PETGATE_scat_lmTOF
                 dict(base, crystals_per_ring=504, Lx=200, Ly=200, Lz=127, events=37_709_254, id_bytes=8, N_splits=4, n_subsets=21)]  # t_PETSIRD
    axes = dict(DEVICES, gpu_free_gb=[30, 4], memory_budget_gb=[None, 25],
                scanner=[(448, 64), (504, 64), (64, 8), (760, 80), (840, 672)], image=[(128, 128, 96), (200, 200, 127), (440, 440, 159)],
                tof=[False, True], events=[50_000_000, 2_000_000, 400_000_000, 1_500_000_000], id_bytes=[4, 8],
                lor_device=['device', 'cpu'], proj_device=['device', 'cpu'], sort_events=[True, False],
                lm_weights=['ram', 'gpu', 'none'], lm_weight_pairs=[None, 146_292_480], lm_sensitivity_ids=['none', 'int32', 'int64'],
                lm_additive=[True, False], algorithm=['OSEM', 'BSREM', 'MLEM', 'OSMAPOSL'],
                prior=[False, True], prior_weight=['euclidean', 'top_n_anatomy'], N_splits=[1, 10, 3])
    configs = tutorials + _sample(base, axes, 12 if quick else 60, 2)
    for p in configs:
        if 'scanner' in p:
            p['crystals_per_ring'], p['rings'] = p.pop('scanner')
        if 'image' in p:
            p['Lx'], p['Ly'], p['Lz'] = p.pop('image')
    return configs


def grid_sss(quick: bool) -> list:
    base = dict(device='cuda', os='windows', gpu_free_gb=30, memory_budget_gb=25, cupy=True, Lx=128, Ly=128, Lz=96,
                crystals_per_ring=448, rings=64, tof=True, tof_bins=21, sss_input='sinogram', events=50_000_000, data='lazy',
                proj_device='device', sensitivity=True, sss_randoms=True, sss_interring_step=6, sss_intraring_step=6,
                num_dense_tof_bins=25, lazy=None, N_splits=1)
    tutorials = [dict(base),                                                                       # t_PETGATE_scat_sinoTOF
                 dict(base, tof=False, data='dense_ram', proj_device='cpu', sss_interring_step=4, sss_intraring_step=4),   # t_PETGATE_scat_sino
                 dict(base, tof=False, sss_input='listmode', events=97_000_000, sss_interring_step=4, sss_intraring_step=4),  # t_PETGATE_scat_lm
                 dict(base, sss_input='listmode', events=97_000_000),                             # t_PETGATE_scat_lmTOF
                 dict(base, tof=False, sss_input='listmode', crystals_per_ring=504, events=37_709_254, Lx=200, Ly=200, Lz=127,
                      sss_interring_step=4, sss_intraring_step=4)]                                # t_PETSIRD
    axes = dict(DEVICES, gpu_free_gb=[30, 4, 11.5], memory_budget_gb=[None, 25, 8, 64], cupy=[True, False],
                scanner=[(448, 64), (504, 64), (64, 8), (760, 80), (840, 672)], image=[(128, 128, 96), (200, 200, 127), (64, 64, 32)],
                tof=[False, True], tof_bins=[21, 13, 33, 80], sss_input=['sinogram', 'listmode'],
                events=[50_000_000, 2_000_000, 400_000_000], data=['dense_ram', 'dense_gpu', 'lazy'], proj_device=['device', 'cpu'],
                sensitivity=[True, False], sss_randoms=[True, False], sss_interring_step=[4, 6, 2, 3], sss_intraring_step=[4, 6, 2, 5],
                num_dense_tof_bins=[25, 10], lazy=[None, True, False], N_splits=[1, 3])
    configs = tutorials + _sample(base, axes, 12 if quick else 60, 3)
    for p in configs:
        if 'scanner' in p:
            p['crystals_per_ring'], p['rings'] = p.pop('scanner')
        if 'image' in p:
            p['Lx'], p['Ly'], p['Lz'] = p.pop('image')
    return configs


CT_BASE = dict(device='cuda', os='windows', gpu_free_gb=30, memory_budget_gb=25, cupy=True, Lx=512, Ly=512, Lz=316, dx=0.662109,
               dy=0.662109, views=9014, views_per_rotation=984, det_cols=888, det_rows=64, focal_radius=538.52, dsd=946.746,
               col_spacing_mm=1.0239, central_col=443.25, row_spacing_mm=1.0988, feed=39.45, slice_thickness=1.25, ct_data='ram',
               proj_device='device', gpu_budget=1.5e9, N_splits=1, n_subsets=8)
CT_MICRO = dict(CT_BASE, Lx=500, Ly=500, Lz=500, views=255, det_cols=500, det_rows=500, ct_data='gpu', n_subsets=5)   # t_CT_microct


def _ct_images(configs):
    for p in configs:
        if 'image' in p:
            p['Lx'], p['Ly'], p['Lz'] = p.pop('image')
        if 'detector' in p:
            p['det_cols'], p['det_rows'] = p.pop('detector')
    return configs


def grid_ct_sart(quick: bool) -> list:
    axes = dict(DEVICES, gpu_free_gb=[30, 4], memory_budget_gb=[None, 25], image=[(512, 512, 316), (256, 256, 128), (500, 500, 500), (1024, 1024, 600)],
                views=[9014, 255, 720, 30000], detector=[(888, 64), (500, 500), (736, 16), (1024, 1024)], proj_device=['device', 'cpu'],
                ct_data=['ram', 'gpu'], N_splits=[1, 4, 3])
    return _ct_images([dict(CT_BASE), dict(CT_MICRO)] + _sample(CT_BASE, axes, 10 if quick else 40, 4))


def grid_ct_helical_fbp(quick: bool) -> list:
    axes = dict(DEVICES, gpu_free_gb=[30, 4, 2], memory_budget_gb=[None, 25, 4, 0.5], cupy=[True, False],
                image=[(512, 512, 316), (256, 256, 128), (512, 512, 100), (768, 768, 400)],
                views=[9014, 4000, 23000, 2000], views_per_rotation=[984, 1152, 720, 2304], detector=[(888, 64), (736, 16), (920, 96), (672, 32)],
                focal_radius=[538.52, 595.0, 541.0], dsd=[946.746, 1085.6, 949.0], col_spacing_mm=[1.0239, 1.2858, 1.0],
                central_col=[None, 443.25, 368.625], row_spacing_mm=[1.0988, 1.0, 0.6], feed=[39.45, 20.0, 55.0, 8.0],
                slice_thickness=[1.25, 0.0, 3.0], ct_data=['ram', 'gpu'], dx=[0.662109, 0.977, 0.5])
    configs = _ct_images([dict(CT_BASE)] + _sample(CT_BASE, axes, 10 if quick else 40, 5))
    for p in configs:                                    # a central column inside the detector
        if p.get('central_col') is not None and p['central_col'] > p['det_cols']:
            p['central_col'] = None
        p['dy'] = p['dx']
    return configs


def grid_ct_cone_fdk(quick: bool) -> list:
    axes = dict(DEVICES, gpu_free_gb=[30, 4], image=[(500, 500, 500), (256, 256, 256), (1000, 1000, 1000), (512, 512, 4)],
                views=[255, 720, 1800, 64], detector=[(500, 500), (1000, 1000), (2000, 2000), (64, 24)], ct_data=['gpu', 'ram'])
    return _ct_images([dict(CT_MICRO)] + _sample(CT_MICRO, axes, 10 if quick else 30, 6))


SPECT_BASE = dict(device='cuda', os='windows', gpu_free_gb=30, memory_budget_gb=None, Lx=128, Ly=128, Lz=128, views=120, proj_r=None,
                  proj_z=None, attenuation=True, cache_probabilities=False, psf='gaussian', psf_kernel=31, n_radii=1, cache_kernel=False,
                  camera_path=False, spect_additive=True, algorithm='OSEM', prior=False, n_subsets=8)


def _spect_axes(**more):
    axes = dict(DEVICES, image=[(128, 128, 128), (64, 64, 64), (128, 128, 100), (256, 256, 200)], views=[120, 96, 60, 128],
                attenuation=[True, False], cache_probabilities=[False, True], psf=['gaussian', 'none', 'operator'],
                n_radii=[1, 2, 120], cache_kernel=[False, True], camera_path=[False, True], spect_additive=[True, False],
                algorithm=['OSEM', 'BSREM', 'OSMAPOSL', 'MLEM', 'RBIEM'], prior=[False, True], prior_weight=['euclidean', 'top_n_anatomy'],
                proj_r=[None, 128], proj_z=[None, 100])
    axes.update(more)
    return axes


def _spect_grid(seed, base=SPECT_BASE, tutorials=(), **more):
    def grid(quick: bool) -> list:
        configs = [dict(t) for t in tutorials] + _sample(base, _spect_axes(**more), 8 if quick else 30, seed)
        for p in configs:
            if 'image' in p:
                p['Lx'], p['Ly'], p['Lz'] = p.pop('image')
            p['n_radii'] = min(p['n_radii'], p['views'])
        return configs
    return grid


SPECT_TUTORIALS = [dict(SPECT_BASE, views=96, n_subsets=8),                                  # t_dicomdata (Lu-177 SymT2, 96 views)
                   dict(SPECT_BASE, views=120, n_subsets=8, algorithm='BSREM', prior=True),  # t_algorithms
                   dict(SPECT_BASE, views=120, psf='operator', n_subsets=4)]                 # t_ac225_simind_recon (PSF operator)
SG_BASE = dict(SPECT_BASE, Lx=196, Ly=196, Lz=112, sg_angles=360, sg_views_per_angle=13, sg_views=2575, n_radii=2575, psf_kernel=21, n_subsets=10)   # t_starguide


def grid_starguide(quick: bool) -> list:
    axes = dict(DEVICES, image=[(196, 196, 112), (128, 128, 100), (96, 96, 64)], sg_angles=[360, 30, 60], sg_views_per_angle=[13, 4, 12],
                sg_views=[None, 2575, 300],
                attenuation=[True, False], cache_probabilities=[False, True], psf=['gaussian', 'none'], n_radii=[2575, 1, 40],
                spect_additive=[True, False], algorithm=['OSEM', 'BSREM', 'MLEM'], prior=[False, True], prior_weight=['euclidean', 'top_n_anatomy'])
    configs = [dict(SG_BASE)] + _sample(SG_BASE, axes, 6 if quick else 20, 8)
    for p in configs:
        if 'image' in p:
            p['Lx'], p['Ly'], p['Lz'] = p.pop('image')
        A, B = p['sg_angles'], p['sg_views_per_angle']
        if p.get('sg_views') is not None and not (B + A - 1 <= p['sg_views'] <= A * B):
            p['sg_views'] = None                           # views that cannot make these angles
        p['n_radii'] = min(p['n_radii'], A * B if p.get('sg_views') is None else p['sg_views'])
    return configs


CHECKERS = {
    'pet_sinogram': Checker('pet_sinogram', build_pet_sinogram, grid_pet_sinogram, _pet_sinogram_settings),
    'pet_listmode': Checker('pet_listmode', build_pet_listmode, grid_pet_listmode, _pet_listmode_settings),
    'sss': Checker('sss', build_sss, grid_sss, _sss_settings, estimate=_estimate_sss, plan=_plan_sss,
                   fixed=lambda p: [{}, {'lazy': True}] + ([{'N_splits': 2}] if p['tof'] else [])),
    'ct_helical_sart': Checker('ct_helical_sart', build_ct_helical_sart, grid_ct_sart, _ct_sart_settings,
                               fixed=({}, {'n_subsets': 8})),
    'ct_helical_fbp': Checker('ct_helical_fbp', build_ct_helical_fbp, grid_ct_helical_fbp, _ct_helical_fbp_settings,
                              estimate=_estimate_ct_helical_fbp, plan=_plan_ct_helical_fbp,
                              budgets=((16, 8), (12, 2), (8, 1), (6, 0.5), (1000, 1000)), fixed=({}, {'gpu_budget': 0.75e9})),
    'ct_cone_sart': Checker('ct_cone_sart', build_ct_cone_sart, grid_ct_sart, _ct_sart_settings),
    'ct_cone_fdk': Checker('ct_cone_fdk', build_ct_cone_fdk, grid_ct_cone_fdk, lambda p: [{}], estimate=_estimate_fbp),
    'spect': Checker('spect', build_spect, _spect_grid(7, tutorials=SPECT_TUTORIALS), _spect_settings,
                     budgets=((16, 8), (6, 1), (5, 0.5), (4.5, 0.3), (1000, 1000))),
    'spect_fbp': Checker('spect_fbp', build_spect_fbp, _spect_grid(9), lambda p: [{}], estimate=_estimate_fbp,
                         budgets=((16, 8), (4, 0.3), (1000, 1000))),
    'starguide': Checker('starguide', build_starguide, grid_starguide, _starguide_settings,
                         budgets=((16, 8), (8, 2), (6, 1), (1000, 1000))),
    'kem': Checker('kem', build_kem, _spect_grid(10, kem_size=[3, 5], kem_support=[1, 2, 0], kem_kernel_on_gpu=[False, True],
                                                   kem_support_on=['device', 'ram']),
                   _spect_settings, budgets=((16, 8), (8, 2), (6, 1), (1000, 1000))),
    'spect_multibed': Checker('spect_multibed', build_spect_multibed, _spect_grid(11, n_beds=[2, 3, 1, 4]), _spect_settings,
                              budgets=((16, 8), (8, 2), (6, 1), (1000, 1000))),
    'spect_extended': Checker('spect_extended', build_spect_extended,
                              _spect_grid(12, tutorials=[dict(SPECT_BASE, views=96, n_radii=71, ext_n=2, ext_distinct=True, ext_motion=False)],   # t_dualpeak
                                          ext_n=[2, 3, 1, 4], ext_distinct=[True, False], ext_motion=[False, True]),
                              _spect_settings, budgets=((16, 8), (8, 2), (6, 1), (1000, 1000))),
    'spect_simind': Checker('spect_simind', build_spect_simind,
                            _spect_grid(13, tutorials=[dict(SPECT_BASE, views=96, n_radii=1, spect_additive=False, simind_windows=1, simind_parallel=24, n_subsets=16)],  # t_spect_mc
                                        simind_windows=[1, 3, 6], simind_parallel=[24, 6, 90], camera_path=[True]),
                            _spect_settings, budgets=((16, 8), (8, 2), (6, 1), (1000, 1000))),
}


# ------------------------------------------------------------------------------------------------- mapping checks
def mapping_checks() -> list:
    """The assumptions the stand-ins make, checked against the library's real objects at small size."""
    from pytomography.metadata.PET import PETSinogramPolygonProjMeta, PETTOFMeta
    from pytomography.io.PET.shared import listmode_to_sinogram
    lines = []
    ok = True
    # 1. the sinogram shape of a scanner, as the metadata computes it (loops over every crystal pair: small scanners)
    for C, R, ct, ca in ((64, 8, 4, 4), (448, 64, 8, 8), (128, 16, 8, 4)):
        meta = PETSinogramPolygonProjMeta(_pet_info(C, R, ct, ca))
        expected = _pet_shape({'crystals_per_ring': C, 'rings': R})
        same = list(meta.shape) == expected and meta.N_angles == expected[0]
        ok &= same
        lines.append(f"sinogram shape of {C} crystals x {R} rings: library {list(meta.shape)}, model {expected}: {'same' if same else 'DIFFERENT'}")
    # 2. what a lazy sinogram binned from events keeps: 4 bytes an event (6 with TOF) and the angle offsets
    info = _pet_info(64, 8, 4, 4)
    gen = torch.Generator().manual_seed(0)
    n = 20000
    for tof in (False, True):
        c1 = torch.randint(0, 64 * 8, (n,), generator=gen)
        c2 = (c1 + 32 + torch.randint(-8, 8, (n,), generator=gen)) % (64 * 8)
        ids = torch.stack([c1, c2] + ([torch.randint(0, 5, (n,), generator=gen)] if tof else []), dim=1).to(torch.int32)
        tof_meta = PETTOFMeta(5, 300.0, 60.0) if tof else None
        lazy = listmode_to_sinogram(ids, info, tof_meta=tof_meta, lazy=True)
        kept = int(lazy.description.split()[2].replace(',', ''))           # events inside the sinogram
        formula = (6 if tof else 4) * kept + 8 * (64 // 2 + 1)
        same = lazy.memory_bytes == formula
        ok &= same
        lines.append(f"lazy sinogram of {kept} events ({'TOF' if tof else 'non-TOF'}): library keeps {lazy.memory_bytes:.0f} bytes, model {formula}: {'same' if same else 'DIFFERENT'}")
    # 3. real sinogram and list mode system matrices (a small scanner, on the CPU) against their stand-ins
    from pytomography.metadata.PET import PETLMProjMeta
    from pytomography.projectors.PET import PETSinogramSystemMatrix, PETLMSystemMatrix
    object_meta = ObjectMeta(dr=(4.0, 4.0, 4.0), shape=(32, 32, 6))
    base = dict(device='cpu', os='windows', gpu_free_gb=30, memory_budget_gb=None, Lx=32, Ly=32, Lz=6, crystals_per_ring=64,
                rings=8, tof=True, tof_bins=5, events=n, id_bytes=4, lor_device='device', proj_device='device', sort_events=True,
                lm_weights='ram', lm_sensitivity_ids='int32', lm_additive=False, algorithm='OSEM', prior=False, N_splits=3, n_subsets=4,
                sensitivity=True, data='dense_ram', additive='none')
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3)
    with simulate(base):
        proj_meta = PETSinogramPolygonProjMeta(info, tof_meta)
        real = PETSinogramSystemMatrix(object_meta, proj_meta, sinogram_sensitivity=torch.rand(proj_meta.shape), N_splits=3, device='cpu')
        stand_in = build_pet_sinogram(base).likelihood.system_matrix
        a, b = real.estimate_memory(4), stand_in.estimate_memory(4)
    same = a.ram_gb == b.ram_gb and a.gpu_gb == b.gpu_gb and [(p.name, p.ram_bytes, p.gpu_bytes) for p in a.parts] == [(p.name, p.ram_bytes, p.gpu_bytes) for p in b.parts]
    ok &= same
    lines.append(f"PETSinogramSystemMatrix (real, 64 x 8 crystals, TOF) and its stand-in: {a.ram_gb:.6f} / {b.ram_gb:.6f} GB RAM, {a.gpu_gb:.6f} / {b.gpu_gb:.6f} GB GPU: {'same' if same else 'DIFFERENT'}")
    saved = PETLMSystemMatrix._backward_full
    PETLMSystemMatrix._backward_full = lambda self, N_splits=20: torch.zeros(self.object_meta.shape)   # its sensitivity image: a projector call
    try:
        c1 = torch.randint(0, 64 * 8, (n,), generator=gen)
        ids = torch.stack([c1, (c1 + 32) % (64 * 8), torch.randint(0, 5, (n,), generator=gen)], dim=1).to(torch.int32)
        weights = torch.rand(64 * 8 * (64 * 8 - 1) // 2)
        pairs = torch.combinations(torch.arange(64 * 8, dtype=torch.int32))
        with simulate(base):
            real = PETLMSystemMatrix(object_meta, PETLMProjMeta(ids, info, tof_meta=tof_meta, weights_sensitivity=weights,
                                                                detector_ids_sensitivity=pairs),
                                     N_splits=3, device='cpu', lor_device='cpu')
            stand_in = _listmode_system_matrix(base)
            a, b = real.estimate_memory(4), stand_in.estimate_memory(4)
    finally:
        PETLMSystemMatrix._backward_full = saved
    same = a.ram_gb == b.ram_gb and a.gpu_gb == b.gpu_gb and [(p.name, p.ram_bytes, p.gpu_bytes) for p in a.parts] == [(p.name, p.ram_bytes, p.gpu_bytes) for p in b.parts]
    ok &= same
    lines.append(f"PETLMSystemMatrix (real, {n} events, TOF, sensitivity weights and their detector IDs) and its stand-in: {a.ram_gb:.6f} / {b.ram_gb:.6f} GB RAM, {a.gpu_gb:.6f} / {b.gpu_gb:.6f} GB GPU: {'same' if same else 'DIFFERENT'}")
    # 4. a real KEM transform (32 x 32 x 16, size 3, one support image) and its stand-in
    from pytomography.projectors.shared import KEMSystemMatrix
    from pytomography.transforms.shared import KEMTransform
    q = dict(SPECT_BASE, device='cpu', Lx=32, Ly=32, Lz=16, views=60, psf='none', kem_size=3, kem_support=1,
             kem_kernel_on_gpu=False, kem_support_on='device')
    with simulate(q):
        sm = _spect_system_matrix(q)
        real = KEMSystemMatrix(sm, KEMTransform([torch.rand(32, 32, 16)], size=3))
        stand_in = build_kem(q).likelihood.system_matrix
        a, b = real.estimate_memory(4), stand_in.estimate_memory(4)
    same = a.ram_gb == b.ram_gb and a.gpu_gb == b.gpu_gb and [(p.name, p.ram_bytes, p.gpu_bytes) for p in a.parts] == [(p.name, p.ram_bytes, p.gpu_bytes) for p in b.parts]
    ok &= same
    lines.append(f"KEMSystemMatrix (real KEMTransform, 32 x 32 x 16, size 3) and its stand-in: {a.ram_gb:.6f} / {b.ram_gb:.6f} GB RAM, {a.gpu_gb:.6f} / {b.gpu_gb:.6f} GB GPU: {'same' if same else 'DIFFERENT'}")
    # 5. a real TopNAnatomyNeighbourWeight (after set_object_meta, as an algorithm's constructor calls it) and its stand-in
    from pytomography.priors import RelativeDifferencePrior
    from pytomography.priors.nearest_neighbour import TopNAnatomyNeighbourWeight
    q = dict(q, prior=True, prior_weight='top_n_anatomy', algorithm='BSREM')
    with simulate(q):
        sm = _spect_system_matrix(q)
        real = _algorithm('BSREM', sm, *_spect_data(q))
        real.prior = RelativeDifferencePrior(beta=1.0, weight=TopNAnatomyNeighbourWeight(torch.rand(32, 32, 16), N_neighbours=8))
        real.prior.set_object_meta(sm.object_meta)
        a, b = real.estimate_memory(4), build_spect(q).estimate_memory(4)
    same = a.ram_gb == b.ram_gb and a.gpu_gb == b.gpu_gb
    ok &= same
    lines.append(f"BSREM with a RelativeDifferencePrior of TopNAnatomyNeighbourWeight (real) and its stand-in: {a.ram_gb:.6f} / {b.ram_gb:.6f} GB RAM, {a.gpu_gb:.6f} / {b.gpu_gb:.6f} GB GPU: {'same' if same else 'DIFFERENT'}")
    return ok, lines


# ---------------------------------------------------------------------------------------------------------- running
def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def _settings_equal(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool):
            if not math.isclose(float(x), float(y), rel_tol=1e-12, abs_tol=0):
                return False
        elif x != y:
            return False
    return True


def check_model(model: MemoryModel, checker: Checker, quick: bool, js_cases: list) -> dict:
    """Compares the model with the library over the checker's grid; returns the summary and adds the cases for the
    JavaScript comparison to ``js_cases``."""
    worst = {'ram': 0.0, 'gpu': 0.0}
    worst_case = None
    estimates = plans = plan_mismatches = searched = unfit = names_differ = 0
    failures = []
    configs = checker.grid(quick)
    names = {v['name'] for v in model.data['variables']}
    for p in configs:
        with simulate(p):
            obj = checker.build(p)                       # may set parameters it measures (e.g. the PSF kernel size)
            params = {k: v for k, v in p.items() if k in names}
            adaptive = []                                # budgets just above the estimates of other settings: the plan must search
            for i, s in enumerate(checker.settings(p)):
                lib = checker.estimate(obj, s)
                if 1 <= i <= 2:
                    adaptive.append((lib['ram_gb'] + 1e-6, lib['gpu_gb'] + 1e-6))
                mine = model.estimate(checker.model_id, params, s)
                js_cases.append({'kind': 'estimate', 'model': checker.model_id, 'params': params, 'settings': s})
                estimates += 1
                for key in ('ram', 'gpu'):
                    a, b = mine[f'{key}_gb'], lib[f'{key}_gb']
                    d = _rel(a, b) if b > 1e-6 else abs(a - b)
                    if d > worst[key]:
                        worst[key] = d
                        worst_case = (p, s, key, a, b)
                    if d > TOLERANCE:
                        failures.append(f"estimate {key}: model {a:.4f} GB, library {b:.4f} GB for {s} in {params}\n"
                                        f"    library parts {[(n, round(r, 4), round(g, 4)) for n, _, r, g in lib['parts']]}\n"
                                        f"    model parts   {[(q['name'], round(q['ram_gb'], 4), round(q['gpu_gb'], 4)) for q in mine['parts']]}")
                if [q['name'] for q in mine['parts']] != [q[0] for q in lib['parts']]:
                    names_differ += 1
                    if names_differ <= 2:
                        failures.append(f"part names: model {[q['name'] for q in mine['parts']]}, library {[q[0] for q in lib['parts']]}")
                if not _settings_equal(mine['settings'], lib['settings']):
                    failures.append(f"estimate settings: model {mine['settings']}, library {lib['settings']} for {s} in {params}")
            fixed_list = checker.fixed(p) if callable(checker.fixed) else checker.fixed
            budgets = list(checker.budgets[:3] if quick else checker.budgets) + adaptive
            for (ram_gb, gpu_gb), fixed in itertools.product(budgets, fixed_list):
                lib = checker.plan(obj, ram_gb, gpu_gb, fixed)
                mine = model.plan(checker.model_id, params, ram_gb, gpu_gb, fixed)
                js_cases.append({'kind': 'plan', 'model': checker.model_id, 'params': params, 'ram_gb': ram_gb, 'gpu_gb': gpu_gb, 'fixed': fixed})
                plans += 1
                searched += lib['tried'] > 1                 # the first settings tried did not fit
                unfit += not lib['fits']
                if not _settings_equal(mine['settings'], lib['settings']) or mine['fits'] != lib['fits']:
                    plan_mismatches += 1
                    failures.append(f"plan for {ram_gb} GB RAM, {gpu_gb} GB GPU (fixed {fixed}): model {mine['settings']} "
                                    f"fits={mine['fits']} ({mine['estimate']['ram_gb']:.3f}, {mine['estimate']['gpu_gb']:.3f} GB), "
                                    f"library {lib['settings']} fits={lib['fits']} ({lib['ram_gb']:.3f}, {lib['gpu_gb']:.3f} GB) in {params}")
    return {'model': checker.model_id, 'configurations': len(configs), 'estimates': estimates, 'plans': plans,
            'max_rel_ram': worst['ram'], 'max_rel_gpu': worst['gpu'], 'plan_mismatches': plan_mismatches,
            'plans_searched': searched, 'plans_unfit': unfit, 'part_names_differ': names_differ,
            'failures': failures, 'worst_case': worst_case}


def check_reference_cases(model: MemoryModel, js_cases: list) -> tuple:
    """The tutorials' configurations (``memory_reference_cases.json``): the model's estimate against the library's for
    each, and both against what the tutorial measured."""
    with open(REFERENCE, encoding='utf-8') as f:
        reference = json.load(f)
    names = {v['name'] for v in model.data['variables']}
    rows, worst, failures = [], 0.0, []
    for case in reference['cases']:
        if not case.get('model'):
            continue
        m = model.by_id[case['model']]
        p = {q['name']: q.get('default') for q in model._params_of(m)}
        p.update(reference.get('defaults', {}))
        p.update(case['params'])
        checker = CHECKERS[case['model']]
        with simulate(p):
            lib = checker.estimate(checker.build(p), case['settings'])
        params = {k: v for k, v in p.items() if k in names}
        mine = model.estimate(case['model'], params, case['settings'])
        js_cases.append({'kind': 'estimate', 'model': case['model'], 'params': params, 'settings': case['settings']})
        d = max(_rel(mine['ram_gb'], lib['ram_gb']), _rel(mine['gpu_gb'], lib['gpu_gb']) if lib['gpu_gb'] > 1e-6 else abs(mine['gpu_gb'] - lib['gpu_gb']))
        worst = max(worst, d)
        if d > TOLERANCE:
            failures.append(f"{case['id']}: model {mine['ram_gb']:.3f} / {mine['gpu_gb']:.3f} GB, library {lib['ram_gb']:.3f} / {lib['gpu_gb']:.3f} GB")
        rows.append((case['id'], mine['ram_gb'], mine['gpu_gb'], lib['ram_gb'], lib['gpu_gb'], case.get('measured', {})))
    return rows, worst, failures


def run_js(cases: list) -> list:
    """The JavaScript evaluator's results for ``cases``, run with node."""
    driver = r"""
const fs = require('fs');
const M = require(process.argv[2]);
M.load(JSON.parse(fs.readFileSync(process.argv[3], 'utf8')));
const cases = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
const out = cases.map(c => {
  if (c.kind === 'estimate') { const e = M.estimate(c.model, c.params, c.settings); return {ram_gb: e.ram_gb, gpu_gb: e.gpu_gb, settings: e.settings, parts: e.parts.map(p => p.name), code: M.code(c.model, c.params, c.settings)}; }
  const p = M.plan(c.model, c.params, c.ram_gb, c.gpu_gb, c.fixed);
  return {settings: p.settings, fits: p.fits, ram_gb: p.estimate.ram_gb, gpu_gb: p.estimate.gpu_gb, tried: p.options.length};
});
fs.writeFileSync(process.argv[5], JSON.stringify(out));
"""
    with tempfile.TemporaryDirectory() as tmp:
        paths = [os.path.join(tmp, name) for name in ('driver.js', 'cases.json', 'out.json')]
        with open(paths[0], 'w') as f:
            f.write(driver)
        with open(paths[1], 'w') as f:
            json.dump(cases, f)
        subprocess.run(['node', paths[0], os.path.abspath(JS), os.path.abspath(DEFAULT_MODEL), paths[1], paths[2]], check=True)
        with open(paths[2]) as f:
            return json.load(f)


def compare_js(model: MemoryModel, cases: list) -> dict:
    """Largest difference between the JavaScript and the Python evaluators, per model."""
    results = run_js(cases)
    summary = {}
    for c, js in zip(cases, results):
        s = summary.setdefault(c['model'], {'cases': 0, 'max_rel': 0.0, 'mismatches': []})
        s['cases'] += 1
        if c['kind'] == 'estimate':
            py = model.estimate(c['model'], c['params'], c['settings'])
            d = max(_rel(js['ram_gb'], py['ram_gb']), _rel(js['gpu_gb'], py['gpu_gb']) if py['gpu_gb'] else abs(js['gpu_gb']))
            s['max_rel'] = max(s['max_rel'], d)
            names = [q['name'] for q in py['parts']]
            code = model.code(c['model'], c['params'], c['settings'])
            if d > 1e-9 or not _settings_equal(js['settings'], py['settings']) or js['parts'] != names or js['code'] != code:
                s['mismatches'].append(f"estimate {c}: js {js}, python {py['ram_gb']}, {py['gpu_gb']}, {py['settings']}, {names}, {code!r}")
        else:
            py = model.plan(c['model'], c['params'], c['ram_gb'], c['gpu_gb'], c['fixed'])
            same = _settings_equal(js['settings'], py['settings']) and js['fits'] == py['fits'] and js['tried'] == len(py['options'])
            if not same:
                s['mismatches'].append(f"plan {c}: js {js}, python {py['settings']} fits={py['fits']} tried={len(py['options'])}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--quick', action='store_true', help='a small grid')
    parser.add_argument('--model', action='append', help='check only this model (repeatable)')
    parser.add_argument('--no-js', action='store_true', help='skip the JavaScript comparison')
    parser.add_argument('--json', help='write the summary to this file')
    args = parser.parse_args(argv)
    model = MemoryModel.load()
    t0 = time.time()
    ok, lines = mapping_checks()
    print('Mapping checks (stand-ins against the real objects):')
    for line in lines:
        print('  ' + line)
    failed = not ok
    js_cases = []
    rows = []
    # every model evaluates and plans with its own defaults alone (no parameter missing from its list)
    for m in model.models():
        model.estimate(m['id'])
        model.plan(m['id'], {}, 16, 8)
        js_cases += [{'kind': 'estimate', 'model': m['id'], 'params': {}, 'settings': {}},
                     {'kind': 'plan', 'model': m['id'], 'params': {}, 'ram_gb': 16, 'gpu_gb': 8, 'fixed': {}}]
    print(f"Every model evaluates and plans with its defaults ({len(model.models())} models).")
    for model_id, checker in CHECKERS.items():
        if args.model and model_id not in args.model:
            continue
        t = time.time()
        r = check_model(model, checker, args.quick, js_cases)
        r['seconds'] = time.time() - t
        rows.append(r)
        bad = max(r['max_rel_ram'], r['max_rel_gpu']) > TOLERANCE or r['plan_mismatches'] or r['failures']
        failed |= bool(bad)
        print(f"{model_id:22s} {r['configurations']:4d} configurations, {r['estimates']:5d} estimates: max difference "
              f"RAM {100 * r['max_rel_ram']:.3f}%, GPU {100 * r['max_rel_gpu']:.3f}%; {r['plans']:4d} plans "
              f"({r['plans_searched']} beyond the first settings, {r['plans_unfit']} fitting nowhere), "
              f"{r['plan_mismatches']} different; part names as the library's in {r['estimates'] - r['part_names_differ']} ({r['seconds']:.0f} s)")
        for f in r['failures'][:8]:
            print('    ' + f)
        if len(r['failures']) > 8:
            print(f"    ... {len(r['failures']) - 8} more")
    if not args.model:
        rows_ref, worst_ref, failures_ref = check_reference_cases(model, js_cases)
        print(f"Reference cases (the tutorials): {len(rows_ref)}, max difference model vs library {100 * worst_ref:.3f}%")
        print(f"  {'case':34s} {'model RAM / GPU':>16s}  {'library':>13s}  {'tutorial peak':>13s}  {'cell peak (RAM above start / GPU)':>34s}")
        for case_id, mr, mg, lr, lg, measured in rows_ref:
            t = measured.get('tutorial', {})
            c = measured.get('cell')
            cell = f"{c['ram_peak_gb'] - c['ram_before_gb']:.2f} / {c['gpu_peak_torch_gb']:.2f}" if c else ''
            print(f"  {case_id:34s} {mr:7.2f} / {mg:5.2f}  {lr:6.2f} / {lg:5.2f}  {t.get('ram_gb', 0):6.1f} / {t.get('gpu_gb', 0):5.2f}  {cell:>34s}")
        for f in failures_ref:
            print('    ' + f)
        failed |= bool(failures_ref)
    js = {}
    if not args.no_js and js_cases:
        js = compare_js(model, js_cases)
        for model_id, s in js.items():
            print(f"{model_id:22s} JavaScript vs Python: {s['cases']} cases, max difference {s['max_rel']:.2e}, {len(s['mismatches'])} different")
            for m in s['mismatches'][:5]:
                print('    ' + m)
            failed |= bool(s['mismatches']) or s['max_rel'] > 1e-9
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'rows': [{k: v for k, v in r.items() if k != 'worst_case'} for r in rows], 'js': js,
                       'mapping': lines}, f, indent=1, default=str)
    print(f"{'FAILED' if failed else 'PASSED'} in {time.time() - t0:.0f} s")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
