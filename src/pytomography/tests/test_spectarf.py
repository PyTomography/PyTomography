"""spectarf, a dependency since PyTomography 4: its adapter reads PyTomography's camera tables and projection metadata,
and its kernels make a SPECT system matrix (a small ARF made on the CPU, in seconds)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pytomography
from pytomography.io import SPECT
from pytomography.metadata.SPECT import EnergyWindow, SPECTObjectMeta, SPECTProjMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix

spectarf_pytomography = pytest.importorskip('spectarf.pytomography')


def _translates_hole_shapes() -> bool:
    """Whether camera_model takes PyTomography's 'square' and 'circular' holes (spectarf calls them 'rect' and 'circ';
    spectarf 0.1.0 passes the name through and refuses them)."""
    try:
        spectarf_pytomography.camera_model(dict(SPECT.collimator('SY-HE'), hole_shape='square'),
                                           SPECT.detector('NaI-9.5mm'), SPECT.camera('Siemens Symbia T'))
    except ValueError:
        return False
    return True


@pytest.mark.parametrize('label', [r['label'] for r in SPECT.detectors()])
@pytest.mark.parametrize('shape', ['hex', 'square', 'circular'])
def test_every_parallel_collimator_and_detector_of_the_tables_makes_a_camera(shape, label):
    if shape != 'hex' and not _translates_hole_shapes():
        pytest.xfail(f"this spectarf does not translate {shape!r} holes into its own names")
    camera = SPECT.camera('Siemens Symbia T')
    rows = [r for r in SPECT.collimators() if r['kind'] == 'parallel' and r['hole_shape'] == shape]
    assert rows
    for row in rows:
        stack, response = spectarf_pytomography.camera_model(SPECT.collimator(row['code']), SPECT.detector(label),
                                                             camera)
        assert stack.collimator.hole_diameter_mm == pytest.approx(row['hole_diameter'])
        assert stack.collimator.hole_length_mm == pytest.approx(row['hole_length'])
        assert stack.crystal_mm == pytest.approx(SPECT.detector(label)['crystal_thickness'])
        assert response.intrinsic_fwhm_mm_140 == pytest.approx(camera['intrinsic_fwhm_140keV'])


def test_kernels_from_a_small_arf_make_a_system_matrix_in_counts_per_photon():
    from spectarf.grid import ArfGrid
    stack, response = spectarf_pytomography.camera_model(SPECT.collimator('SY-HE'), SPECT.detector('NaI-9.5mm'),
                                                         SPECT.camera('Siemens Symbia T'))
    # views 90 degrees apart, which the system matrix turns the object to exactly (no interpolation)
    proj_meta = SPECTProjMeta((16, 16), (4.8, 4.8), np.arange(0, 360, 90.0), radii=np.full(4, 200.0),
                              energy_windows=[EnergyWindow('peak', [(187.2, 228.8)], 208.0)])
    object_meta = SPECTObjectMeta((4.8, 4.8, 4.8), (16, 16, 16))
    # a coarse grid of incidence angles and 20,000 photons per angle, on the CPU: about 3 s
    grid = ArfGrid(theta_deg=np.linspace(0.0, 8.0, 9), phi_deg=np.array([0.0, 15.0, 30.0]),
                   dx_edges_mm=np.linspace(-30.0, 30.0, 41), dy_edges_mm=np.linspace(-30.0, 30.0, 41))
    arfs = spectarf_pytomography.generate_arfs(stack, response, proj_meta, peak=0, cache=False, engine='torch',
                                               device='cpu', photons_per_node=20_000, grid=grid)
    system_matrix = SPECTSystemMatrix(obj2obj_transforms=[spectarf_pytomography.psf_transform(arfs)],
                                      proj2proj_transforms=[], object_meta=object_meta, proj_meta=proj_meta)
    point = torch.zeros(object_meta.shape, device=pytomography.device)
    point[8, 8, 8] = 1.0
    per_view = system_matrix.forward(point).sum(dim=(1, 2)).cpu()
    # absolute kernels: counts per emitted photon, about 1e-4 per view for a high-energy collimator at 208 keV
    assert torch.all((per_view > 3e-5) & (per_view < 3e-4)), per_view
    assert float(per_view.max() / per_view.min()) < 1.05
    # and the back projection is its transpose
    g = torch.Generator().manual_seed(0)
    x = torch.rand(object_meta.shape, generator=g).to(pytomography.device)
    y = torch.rand(proj_meta.shape, generator=g).to(pytomography.device)
    lhs = float((system_matrix.forward(x) * y).sum())
    rhs = float((x * system_matrix.backward(y)).sum())
    assert lhs == pytest.approx(rhs, rel=1e-5)
