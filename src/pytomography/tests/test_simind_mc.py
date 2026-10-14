"""The SIMIND forward projection of the hybrid Monte Carlo system matrix (``pytomography.utils.simind_mc``): the command
it runs, and, where SIMIND is installed, that the voxel phantom attenuates the photons on their way to the detector."""
import math
import shutil

import numpy as np
import pytest
import torch

import pytomography
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


def test_detector_thicknesses_are_mm_and_reach_simind_in_cm(monkeypatch):
    """The detector thicknesses are in mm, like every length in PyTomography 4, and become SIMIND's cm (indices 09, 11
    and 13) only in the parameters SIMIND is called with; the hybrid system matrix passes them through unchanged. Their
    names before PyTomography 4 (in cm) raise a TypeError that gives the value in mm."""
    params = simind_mc.get_simind_isotope_detector_params('lu177', 'SY-ME', cover_thickness_mm=1.0,
                                                          backscatter_thickness_mm=66.0, crystal_thickness_mm=9.525)
    assert (params['09'], params['11'], params['13']) == (0.9525, 6.6, 0.1)
    command = simind_mc.create_simind_command(params, 0)
    assert '/09:0.9525' in command and '/11:6.6/' in command and '/13:0.1/' in command
    with pytest.raises(TypeError, match='crystal_thickness_mm=9.525'):
        simind_mc.get_simind_isotope_detector_params('lu177', 'SY-ME', cover_thickness_mm=1.0,
                                                     backscatter_thickness_mm=66.0, crystal_thickness=0.9525)
    from pytomography.projectors.SPECT import dualhead_system_matrix, MonteCarloHybridSPECTSystemMatrix
    monkeypatch.setattr(dualhead_system_matrix.shutil, 'which', lambda name: 'simind')
    N, dx = 8, 4.8
    object_meta = SPECTObjectMeta((dx, dx, dx), (N, N, N))
    proj_meta = SPECTProjMeta((N, N), (dx, dx), np.array([0.0, 90.0]), radii=np.full(2, 250.0))
    common = dict(n_events=10, n_parallel=1, obj2obj_transforms=[], proj2proj_transforms=[],
                  attenuation_map_140keV=torch.zeros((N, N, N)), energy_window_params=['187.2,228.8,0'],
                  primary_window_idx=0, isotope_names=['lu177'], isotope_ratios=[1.0], collimator_type='SY-ME')
    with pytest.raises(TypeError, match='cover_thickness_mm=1'):
        MonteCarloHybridSPECTSystemMatrix(object_meta, proj_meta, crystal_thickness_mm=9.525, cover_thickness=0.1,
                                          backscatter_thickness_mm=66.0, **common)
    sm = MonteCarloHybridSPECTSystemMatrix(object_meta, proj_meta, crystal_thickness_mm=9.525, cover_thickness_mm=1.0,
                                           backscatter_thickness_mm=66.0, **common)
    seen = []

    def fake_simulation(object, attenuation_map, object_meta, proj_meta, windows, index_dict, *args, **kwargs):
        seen.append(dict(index_dict))
        return torch.zeros((len(windows), proj_meta.num_projections, N, N), device=pytomography.device)
    monkeypatch.setattr(simind_mc, 'run_scatter_simulation', fake_simulation)
    sm.forward(torch.ones((N, N, N)).to(pytomography.device))
    assert seen and all((d['09'], d['11'], d['13']) == (0.9525, 6.6, 0.1) for d in seen)


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
    N, dx = 128, 4.8
    object_meta = SPECTObjectMeta((dx, dx, dx), (N, N, N))
    proj_meta = SPECTProjMeta((N, N), (dx, dx), np.array([0.0, 90.0]), radii=np.full(2, 250.0))
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    r = np.hypot(i - (N - 1) / 2, j - (N - 1) / 2) * dx
    water = torch.zeros((N, N, N))
    water[torch.from_numpy(r < 100)] = 0.01538  # 1/mm, water at 140.5 keV: SIMIND's maps are at 140.5 keV
    source = torch.zeros((N, N, N))
    source[63:65, 63:65, 63:65] = 1
    index_dict = simind_mc.get_simind_params_from_metadata(object_meta, proj_meta)
    index_dict.update(simind_mc.get_simind_isotope_detector_params(
        isotope_name='lu177', collimator_type='SY-ME', cover_thickness_mm=1.0, backscatter_thickness_mm=66.0,
        crystal_thickness_mm=9.525, energy_resolution_140keV=10))
    total_air, scatter_air = _total_and_scatter(torch.zeros((N, N, N)), source, object_meta, proj_meta, index_dict)
    total_water, scatter_water = _total_and_scatter(water, source, object_meta, proj_meta, index_dict)
    assert scatter_air == 0
    primary_ratio = (total_water - scatter_water) / total_air
    expected = math.exp(-0.1365 * 10)
    assert abs(primary_ratio - expected) < 0.2 * expected, (primary_ratio, expected)
    assert 0 < scatter_water < total_water
