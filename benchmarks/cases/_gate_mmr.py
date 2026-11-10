"""The GATE mMR brain simulation that the PET cases share (the data of the t_PETGATE tutorials).

Reading the 54 ROOT files takes a few minutes, so, as in the tutorials, each kind of coincidence is kept in the output
folder (datasets.output_dir) the first time and loaded from there afterwards; the tutorials and the benchmarks share
those files when PYTOMOGRAPHY_OUTPUT is the same.
"""
import torch

from pytomography import datasets, io
from pytomography.metadata.PET import PETTOFMeta

datasets.fetch('PET/GATE-mMR-BrainInWater')
PATH = datasets.data_dir() / 'PET' / 'GATE-mMR-BrainInWater'
OUTPUT = datasets.output_dir('PET/GATE-mMR-BrainInWater')
MACRO = PATH / 'mMR_Geometry.mac'
ROOT_FILES = [PATH / 'simple_phantom' / f'mMR_voxBrain_withSimplePhantom_{i}.root' for i in range(1, 55)]


def cache_name(events, tof_meta=None, normalization=None):
    return f'gate_{events}' + ('' if tof_meta is None else f'_tof{tof_meta.num_bins}') + \
        ('' if normalization is None else '_norm') + '.pt'


def read_gate(events, tof_meta=None, normalization=None):
    """io.PET.read_gate for one kind of coincidence, cached as in the tutorials: (events, object_meta, proj_meta)."""
    cache = OUTPUT / cache_name(events, tof_meta, normalization)
    if not cache.exists():
        torch.save(io.PET.read_gate(ROOT_FILES, MACRO, events=events, tof_meta=tof_meta, normalization=normalization,
                                    doi=9), cache)
    return torch.load(cache, weights_only=False)


def tof_meta():
    """The tutorials' TOF: 550 ps resolution, 21 bins over 1000 ps."""
    c = 0.3  # mm/ps
    return PETTOFMeta(21, 1000 * c, 550 * c / 2, n_sigmas=3)


def grid_and_attenuation(object_meta, proj_meta):
    """The tutorials' grid (128 x 128 x 96 voxels of 2 mm) and the phantom's attenuation map on it (1/mm)."""
    object_meta = object_meta.with_grid(dr=(2, 2, 2), shape=(128, 128, 96))
    atten_map = io.attenuation_map(io.PET.gate_image(PATH / 'umap_mMR_brainSimplePhantom.hv', proj_meta, units='1/cm'),
                                   object_meta)
    return object_meta, atten_map


def smoothed_randoms(info):
    """The delayed coincidences binned into a sinogram and smoothed: the randoms estimate."""
    delays, _, _ = read_gate('delayeds')
    sinogram = io.PET.listmode_to_sinogram(delays, info)
    del delays
    return io.PET.smooth_randoms_sinogram(sinogram, info, fwhm_r=20, fwhm_theta=7, fwhm_z=40)


def normalization_weights():
    """The normalization weight of every crystal pair (in torch.combinations order), as read_gate gives them."""
    _, _, proj_meta = read_gate('trues', normalization=PATH / 'normalization_weights.npy')
    return proj_meta.weights_sensitivity


def normalization_sinogram(info):
    """The normalization weights as a sinogram, cached as in the tutorials."""
    cache = OUTPUT / 'normalization_sinogram.pt'
    if not cache.exists():
        torch.save(io.PET.normalization_sinogram(normalization_weights(), info), cache)
    return torch.load(cache)


def cached(names):
    """Which of the tutorials' cache files exist already (reading the ROOT files instead is much slower)."""
    return {n: (OUTPUT / n).exists() for n in names}
