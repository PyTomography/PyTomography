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
    attenuate photons through voxels of the first material (see ``simind_mc.SIMIND_INI_CHANGES``). The scattwin
    images are asked for with /CA:2, the setting with which SIMIND V9 writes the scatter images (and V8 too)."""
    switches = simind_mc.create_simind_command({'29': 2, 'fi': 'lu177'}, 3).split('/')
    assert switches[0] == 'temp_output3'
    assert 'RR:3' in switches
    assert 'CA:2' in switches
    assert 'in:x2,6x' in switches
    assert 'in:x21,0x' in switches
    assert switches[-2:] == ['29:2', 'fi:lu177']


def _staged(tmp_path, n_parallel, windows):
    """The files SIMIND leaves for ``add_together``: process i's images hold i + 1 (total) and (i + 1) / 2 (scatter)."""
    for i in range(n_parallel):
        for w in range(1, windows + 1):
            for kind, value in (('tot', i + 1.0), ('sca', (i + 1.0) / 2)):
                name = f'temp_output{i}_{kind}_w{w}'
                np.full(6, value, np.float32).tofile(tmp_path / f'{name}.a00')
                (tmp_path / f'{name}.h00').write_text(f'!INTERFILE :=\n!name of data file := {name}.a00\n'
                                                      f'patient name := SMC_{name}.a00\n!END OF INTERFILE :=\n')


def test_projections_are_added_up_without_shell_tools(tmp_path, monkeypatch):
    """The images of the parallel SIMIND processes are averaged and given headers in Python: no cp, mv or sed, which
    a Windows computer without Git for Windows does not have."""
    _staged(tmp_path, n_parallel=3, windows=2)
    monkeypatch.setattr(simind_mc.subprocess, 'run', None)       # any shell call would fail
    monkeypatch.setattr(simind_mc.subprocess, 'Popen', None)
    simind_mc.add_together(3, 2, str(tmp_path))
    for w in (1, 2):
        assert np.allclose(np.fromfile(tmp_path / f'tot_w{w}.a00', np.float32), 2.0)
        assert np.allclose(np.fromfile(tmp_path / f'sca_w{w}.a00', np.float32), 1.0)
        for kind in ('tot', 'sca'):
            header = (tmp_path / f'{kind}_w{w}.h00').read_text()
            assert f'!name of data file := {kind}_w{w}.a00' in header and 'temp_output' not in header


class _FailedSimind:
    """A SIMIND process that stops with an access violation."""
    def __init__(self, args, **kwargs):
        self.args, self.returncode = args, 157

    def communicate(self):
        return b'', b'forrtl: severe (157): Program Exception - access violation'


def test_a_failed_simind_raises(monkeypatch):
    """A SIMIND process that fails stops the forward projection with its error, instead of a missing-file error later.
    The maps and simind.smc are staged without shell tools first."""
    monkeypatch.setattr(simind_mc.subprocess, 'Popen', _FailedSimind)
    N, dx = 4, 4.8
    object_meta = SPECTObjectMeta((dx, dx, dx), (N, N, N))
    proj_meta = SPECTProjMeta((N, N), (dx, dx), np.array([0.0, 90.0]), radii=np.full(2, 250.0))
    with pytest.raises(RuntimeError, match='2 of 2 SIMIND processes failed(.|\n)*access violation'):
        simind_mc.run_scatter_simulation(torch.ones((N, N, N)), torch.zeros((N, N, N)), object_meta, proj_meta,
                                         ['187.2,228.8,0'], {'29': 2}, 1e3, 2)


def test_detector_thicknesses_are_mm_and_reach_simind_in_cm(monkeypatch):
    """The detector thicknesses are in mm, like every length in PyTomography 4, and become SIMIND's cm (indices 09, 11
    and 13) only in the parameters SIMIND is called with; the hybrid system matrix passes them through unchanged. The
    same names were in cm before PyTomography 4: a crystal thinner than 3 mm, a thickness in cm, raises a ValueError that
    gives it in mm."""
    params = simind_mc.get_simind_isotope_detector_params('lu177', 'SY-ME', cover_thickness=1.0,
                                                          backscatter_thickness=66.0, crystal_thickness=9.525)
    assert (params['09'], params['11'], params['13']) == (0.9525, 6.6, 0.1)
    command = simind_mc.create_simind_command(params, 0)
    assert '/09:0.9525' in command and '/11:6.6/' in command and '/13:0.1/' in command
    with pytest.raises(ValueError, match='crystal_thickness=9.525'):
        simind_mc.get_simind_isotope_detector_params('lu177', 'SY-ME', cover_thickness=0.1,
                                                     backscatter_thickness=6.6, crystal_thickness=0.9525)
    from pytomography.projectors.SPECT import dualhead_system_matrix, MonteCarloHybridSPECTSystemMatrix
    monkeypatch.setattr(dualhead_system_matrix.shutil, 'which', lambda name: 'simind')
    N, dx = 8, 4.8
    object_meta = SPECTObjectMeta((dx, dx, dx), (N, N, N))
    proj_meta = SPECTProjMeta((N, N), (dx, dx), np.array([0.0, 90.0]), radii=np.full(2, 250.0))
    common = dict(n_events=10, n_parallel=1, obj2obj_transforms=[], proj2proj_transforms=[],
                  attenuation_map_140keV=torch.zeros((N, N, N)), energy_window_params=['187.2,228.8,0'],
                  primary_window_idx=0, isotope_names=['lu177'], isotope_ratios=[1.0], collimator_type='SY-ME')
    with pytest.raises(ValueError, match='crystal_thickness=9.525'):
        MonteCarloHybridSPECTSystemMatrix(object_meta, proj_meta, crystal_thickness=0.9525, cover_thickness=0.1,
                                          backscatter_thickness=6.6, **common)
    sm = MonteCarloHybridSPECTSystemMatrix(object_meta, proj_meta, crystal_thickness=9.525, cover_thickness=1.0,
                                           backscatter_thickness=66.0, **common)
    seen = []

    def fake_simulation(object, attenuation_map, object_meta, proj_meta, windows, index_dict, *args, **kwargs):
        seen.append(dict(index_dict))
        return torch.zeros((len(windows), proj_meta.num_projections, N, N), device=pytomography.device)
    monkeypatch.setattr(simind_mc, 'run_scatter_simulation', fake_simulation)
    sm.forward(torch.ones((N, N, N)).to(pytomography.device))
    assert seen and all((d['09'], d['11'], d['13']) == (0.9525, 6.6, 0.1) for d in seen)


def test_only_the_reconstructed_window_is_simulated(monkeypatch):
    """The hybrid system matrix asks SIMIND for the window(s) of primary_window_idx only, and returns those
    projections; the other windows of energy_window_params would be thrown away."""
    from pytomography.projectors.SPECT import dualhead_system_matrix, MonteCarloHybridSPECTSystemMatrix
    monkeypatch.setattr(dualhead_system_matrix.shutil, 'which', lambda name: 'simind')
    N, dx = 8, 4.8
    object_meta = SPECTObjectMeta((dx, dx, dx), (N, N, N))
    proj_meta = SPECTProjMeta((N, N), (dx, dx), np.array([0.0, 90.0]), radii=np.full(2, 250.0))
    windows = ['187.2,228.8,0', '166.4,187.2,0', '228.8,249.6,0']
    asked = []

    def fake_simulation(object, attenuation_map, object_meta, proj_meta, windows, *args, **kwargs):
        asked.append(list(windows))           # window w's projections hold its lower bound
        return torch.stack([torch.full((proj_meta.num_projections, N, N), float(w.split(',')[0]))
                            for w in windows]).to(pytomography.device)
    monkeypatch.setattr(simind_mc, 'run_scatter_simulation', fake_simulation)
    common = dict(n_events=10, n_parallel=1, obj2obj_transforms=[], proj2proj_transforms=[],
                  attenuation_map_140keV=torch.zeros((N, N, N)), energy_window_params=windows,
                  isotope_names=['lu177'], isotope_ratios=[1.0], collimator_type='SY-ME', crystal_thickness=9.525,
                  cover_thickness=1.0, backscatter_thickness=66.0)
    obj = torch.ones((N, N, N)).to(pytomography.device)
    one = MonteCarloHybridSPECTSystemMatrix(object_meta, proj_meta, primary_window_idx=2, **common).forward(obj).cpu()
    assert asked[-1] == [windows[2]] and one.shape == (2, N, N)
    assert torch.allclose(one, torch.full((2, N, N), 228.8 * N ** 3))
    two = MonteCarloHybridSPECTSystemMatrix(object_meta, proj_meta, primary_window_idx=[0, 2], **common).forward(obj).cpu()
    assert asked[-1] == [windows[0], windows[2]] and two.shape == (2, 2, N, N)
    assert torch.allclose(two[1], one)


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
    SIMIND V8.0 gave as many as in air; V9.0 attenuates them with or without it."""
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
        isotope_name='lu177', collimator_type='SY-ME', cover_thickness=1.0, backscatter_thickness=66.0,
        crystal_thickness=9.525, energy_resolution_140keV=10))
    total_air, scatter_air = _total_and_scatter(torch.zeros((N, N, N)), source, object_meta, proj_meta, index_dict)
    total_water, scatter_water = _total_and_scatter(water, source, object_meta, proj_meta, index_dict)
    assert scatter_air == 0
    primary_ratio = (total_water - scatter_water) / total_air
    expected = math.exp(-0.1365 * 10)
    assert abs(primary_ratio - expected) < 0.2 * expected, (primary_ratio, expected)
    assert 0 < scatter_water < total_water
