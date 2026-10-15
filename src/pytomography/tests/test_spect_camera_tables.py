"""PyTomography's SPECT camera tables (Luke, 14 Oct 2026: PyTomography keeps the libraries of names and gives spectarf
numbers): collimators by code, detectors by label, cameras by name, and collimator_psf reading the same table."""
from __future__ import annotations

import numpy as np
import pytest

from pytomography.io import SPECT


def test_a_collimator_code_gives_spectarfs_collimator_arguments():
    he = SPECT.collimator('SY-HE')
    assert he == dict(hole_diameter=4.0, septal_thickness=2.0, hole_length=59.7, hole_shape='hex', rotated=True,
                      material='lead', code='SY-HE')
    assert SPECT.collimator('sy-he') == he                                   # any case
    assert SPECT.collimator('SY-HE', material='tungsten')['material'] == 'tungsten'
    assert not SPECT.collimator('GI-MEGP')['rotated']                        # SIMIND's positive hole-shape code


def test_the_table_is_simind_9s_dimensions_and_keeps_the_older_codes():
    rows = {r['code']: r for r in SPECT.collimators()}
    assert len(rows) == 124 and sum(r['kind'] == 'parallel' for r in rows.values()) == 120
    assert rows['SI-HE']['hole_diameter'] == 4.0 and rows['SI-HE']['source'] == 'SIMIND 9.0'   # 4.06 before 9.0
    assert rows['G8-LUHR']['hole_diameter'] == pytest.approx(1.22)                           # 2.2 before 9.0
    assert rows['SI-HEGP']['source'] == 'SIMIND before 9.0'                   # no longer in SIMIND 9.0, still here
    assert rows['SY-HE']['family'] == 'Siemens Symbia' and rows['SY-HE']['type'] == 'high energy'
    assert all(r['hole_diameter'] > 0 and r['hole_length'] > r['septal_thickness'] for r in rows.values())


def test_unknown_and_non_parallel_collimators_are_refused():
    with pytest.raises(KeyError, match='SY-HE'):                              # the closest codes are suggested
        SPECT.collimator('SY-HEE')
    with pytest.raises(KeyError, match='spect_parameters'):
        SPECT.collimator('XX-NONE')
    with pytest.raises(ValueError, match='fan'):
        SPECT.collimator('GV-LEUR')


def test_detector_labels_give_the_crystal_cover_and_backscatter():
    thin = SPECT.detector('NaI-9.5mm')
    assert thin == dict(crystal_thickness=9.525, crystal_material='nai', cover_thickness=1.0, cover_material='aluminium',
                        backscatter_thickness=100.0, backscatter_material='pmt')
    assert SPECT.detector('nai-15.9MM')['crystal_thickness'] == 15.875
    assert [r['label'] for r in SPECT.detectors()] == ['NaI-9.5mm', 'NaI-15.9mm']
    with pytest.raises(KeyError, match='NaI-9.5mm'):
        SPECT.detector('NaI-9mm')


def test_a_camera_gives_its_energy_resolution_model_and_intrinsic_resolution():
    symbia = SPECT.camera('Siemens Symbia T')
    assert symbia == dict(name='Siemens Symbia T', energy_resolution_model='Morphis',
                          energy_resolution=dict(a=-0.534, b=0.946, c=0.006), intrinsic_fwhm_140keV=3.8)
    assert SPECT.camera('symbia t2') == SPECT.camera('Symbia T16') == symbia
    # Morphis et al.'s model: about 10.5 % at 140 keV and 37 keV FWHM at Ac-225's 440 keV
    from pytomography.io.SPECT._camera import energy_resolution_fwhm
    fwhm = energy_resolution_fwhm('Morphis', np.array([140.0, 440.0]), **symbia['energy_resolution'])
    np.testing.assert_allclose(fwhm, [14.65, 37.33], atol=0.01)
    with pytest.raises(KeyError):
        SPECT.camera('Symbia Z')


def test_collimator_psf_reads_the_same_table():
    psf = SPECT.collimator_psf('SY-ME', 208.4)
    me = SPECT.collimator('SY-ME')
    assert psf.sigma_fit_params[1] == pytest.approx(me['hole_diameter'] / (2 * np.sqrt(2 * np.log(2))))
    assert SPECT.collimator_psf('sy-me', 208.4).sigma_fit_params == psf.sigma_fit_params
    with pytest.raises(ValueError):
        SPECT.collimator_psf('XX-NONE', 140.5)


def test_the_tables_print():
    from pytomography.utils import print_collimator_parameters
    print_collimator_parameters()
