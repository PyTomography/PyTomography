"""The SIMIND forward projection of the hybrid Monte Carlo system matrix (``pytomography.utils.simind_mc``): the command
it runs, and, where SIMIND is installed, that the voxel phantom attenuates the photons on their way to the detector."""
import math
import shutil

import numpy as np
import pytest
import torch

from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.utils import simind_mc


def test_command_puts_every_voxel_in_the_second_phantom_material():
    """simind.ini entry 21 (the bone density threshold) is set to 0 by a switch of its own: SIMIND V8.0 does not
    attenuate photons through voxels of the first material (see ``simind_mc.SIMIND_INI_CHANGES``)."""
    switches = simind_mc.create_simind_command({'29': 2, 'fi': 'lu177'}, 3).split('/')
    assert switches[0] == 'temp_output3'
    assert 'RR:3' in switches
    assert 'in:x2,6x' in switches
    assert 'in:x21,0x' in switches
    assert switches[-2:] == ['29:2', 'fi:lu177']


def _total_and_scatter(attenuation_map, source, object_meta, proj_meta, index_dict):
    """Photopeak window projections (total, scatter) of SIMIND for the given maps, 1e6 photons per view."""
    window = ['187.2,228.8,0']
    args = (source, attenuation_map, object_meta, proj_meta, window, index_dict, 1e6, 2)
    total = simind_mc.run_scatter_simulation(*args, return_total=True)[0]
    scatter = simind_mc.run_scatter_simulation(*args, return_total=False)[0]  # same seeds, so the same photons
    return total.sum().item(), scatter.sum().item()


@pytest.mark.skipif(shutil.which('simind') is None, reason='needs SIMIND on the PATH')
def test_voxel_phantom_attenuates_photons():
    """A Lu-177 point source in the centre of a water cylinder of radius 10 cm: the unscattered photons in the 208 keV
    window are exp(-mu * 10 cm) of those in air, mu = 0.1365 /cm (water at 208 keV). Before entry 21 was set to 0,
    SIMIND V8.0 gave as many as in air."""
    N, dx = 128, 0.48
    object_meta = SPECTObjectMeta((dx, dx, dx), (N, N, N))
    proj_meta = SPECTProjMeta((N, N), (dx, dx), np.array([0.0, 90.0]), radii=np.full(2, 25.0))
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    r = np.hypot(i - (N - 1) / 2, j - (N - 1) / 2) * dx
    water = torch.zeros((N, N, N))
    water[torch.from_numpy(r < 10)] = 0.1538  # 1/cm, water at 140.5 keV: SIMIND's maps are at 140.5 keV
    source = torch.zeros((N, N, N))
    source[63:65, 63:65, 63:65] = 1
    index_dict = simind_mc.get_simind_params_from_metadata(object_meta, proj_meta)
    index_dict.update(simind_mc.get_simind_isotope_detector_params(
        isotope_name='lu177', collimator_type='SY-ME', cover_thickness=0.1, backscatter_thickness=6.6,
        crystal_thickness=0.9525, energy_resolution_140keV=10))
    total_air, scatter_air = _total_and_scatter(torch.zeros((N, N, N)), source, object_meta, proj_meta, index_dict)
    total_water, scatter_water = _total_and_scatter(water, source, object_meta, proj_meta, index_dict)
    assert scatter_air == 0
    primary_ratio = (total_water - scatter_water) / total_air
    expected = math.exp(-0.1365 * 10)
    assert abs(primary_ratio - expected) < 0.2 * expected, (primary_ratio, expected)
    assert 0 < scatter_water < total_water
