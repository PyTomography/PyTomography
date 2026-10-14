"""SPECT camera parameters by name: collimators by code, detectors by label, cameras by name.

PyTomography keeps the tables (``pytomography/data``: ``collimators.csv``, ``detectors.csv``, ``cameras.json``; the
docs' "SPECT parameters" page lists them all) and returns plain numbers, named as spectarf's classes take them, so a
camera for angular response functions reads::

    collimator = io.SPECT.collimator('SY-HE')
    detector = io.SPECT.detector('NaI-9.5mm')
    camera = io.SPECT.camera('Siemens Symbia T')
    stack = spectarf.DetectorStack(spectarf.Collimator(**collimator), **detector)
    response = spectarf.DetectorResponse(spectarf.EnergyResolution.geb(**camera['energy_resolution']),
                                         intrinsic_fwhm_mm_140=camera['intrinsic_fwhm_mm_140'])
"""
from __future__ import annotations
import csv
import difflib
import functools
import json
import os

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
#: The docs page that lists every collimator, detector and camera.
PARAMETERS_PAGE = 'https://pytomography.readthedocs.io/en/latest/spect_parameters.html'


def _rows(name: str) -> list[dict]:
    with open(os.path.join(_DATA, name), newline='', encoding='utf-8') as f:
        return list(csv.DictReader(line for line in f if not line.startswith('#')))


def _number(value: str) -> float | None:
    return float(value) if value not in ('', None) else None


@functools.lru_cache(maxsize=None)
def collimators() -> tuple[dict, ...]:
    """Every collimator of PyTomography's table, from SIMIND's collimator database: code, camera family, type,
    kind (``'parallel'``, ``'fan'``), hole shape (``'hex'``, ``'square'``, ``'circular'``), whether the hexagonal
    lattice is rotated by 90 degrees, hole diameter (across the flats for hexagons), septal thickness and hole length
    in mm, focal length in mm (fan beams), and source.

    Returns:
        tuple[dict, ...]: The rows, sorted by family and code.
    """
    out = []
    for r in _rows('collimators.csv'):
        out.append(dict(code=r['code'], family=r['family'], type=r['type'], kind=r['kind'], hole_shape=r['hole_shape'],
                        rotated=r['rotated'] == 'true', hole_diameter_mm=_number(r['hole_diameter_mm']),
                        septal_thickness_mm=_number(r['septal_thickness_mm']), hole_length_mm=_number(r['hole_length_mm']),
                        focal_length_mm=_number(r['focal_length_mm']), source=r['source']))
    return tuple(out)


@functools.lru_cache(maxsize=None)
def detectors() -> tuple[dict, ...]:
    """Every detector label of PyTomography's table: the crystal, its thickness and material, the cover and the
    backscatter behind the crystal (thicknesses in mm, materials as spectarf names them), and source.

    Returns:
        tuple[dict, ...]: The rows.
    """
    out = []
    for r in _rows('detectors.csv'):
        out.append(dict(label=r['label'], crystal=r['crystal'], crystal_mm=float(r['crystal_mm']),
                        crystal_material=r['crystal_material'], cover_mm=float(r['cover_mm']),
                        cover_material=r['cover_material'], backscatter_mm=float(r['backscatter_mm']),
                        backscatter_material=r['backscatter_material'], source=r['source']))
    return tuple(out)


@functools.lru_cache(maxsize=None)
def cameras() -> tuple[dict, ...]:
    """Every camera of PyTomography's table: its name and other names, its energy resolution model (``model``, the
    model's parameters and the source) and its intrinsic spatial resolution (FWHM at 140 keV, mm) with source.

    Returns:
        tuple[dict, ...]: The cameras.
    """
    with open(os.path.join(_DATA, 'cameras.json'), encoding='utf-8') as f:
        return tuple(json.load(f)['cameras'])


def _key(name: str) -> str:
    return ''.join(name.split()).upper()


def _not_found(what: str, name: str, names) -> KeyError:
    by_key = {n.upper(): n for n in names}
    close = [by_key[c] for c in difflib.get_close_matches(name.upper(), list(by_key), n=3, cutoff=0.5)]
    hint = f'; did you mean {", ".join(close)}?' if close else ''
    return KeyError(f'no {what} {name!r} in PyTomography\'s table{hint} Every {what} is listed at {PARAMETERS_PAGE}')


def collimator(code: str, material: str = 'lead') -> dict:
    """A parallel-hole collimator's dimensions, from its code (SIMIND's; e.g. ``'SY-HE'``, the Siemens Symbia high
    energy collimator).

    Args:
        code (str): The collimator's code (any case). Every code is listed on the docs' "SPECT parameters" page.
        material (str, optional): The septa's material, as spectarf names it. Defaults to ``'lead'``.

    Returns:
        dict: ``hole_diameter_mm`` (across the flats for hexagons), ``septal_thickness_mm``, ``hole_length_mm``,
        ``hole_shape`` (``'hex'``, ``'square'`` or ``'circular'``), ``rotated`` (the hexagonal lattice turned 90
        degrees), ``material`` and ``code``: the arguments of ``spectarf.Collimator``.

    Raises:
        KeyError: The code is not in the table (the closest codes are suggested).
        ValueError: The collimator is not a parallel-hole collimator (e.g. a fan beam).
    """
    rows = {r['code'].upper(): r for r in collimators()}
    row = rows.get(code.strip().upper())
    if row is None:
        raise _not_found('collimator', code, [r['code'] for r in collimators()])
    if row['kind'] != 'parallel':
        raise ValueError(f'{row["code"]} is a {row["kind"]} collimator; PyTomography models parallel-hole collimators only')
    return dict(hole_diameter_mm=row['hole_diameter_mm'], septal_thickness_mm=row['septal_thickness_mm'],
                hole_length_mm=row['hole_length_mm'], hole_shape=row['hole_shape'], rotated=row['rotated'],
                material=material, code=row['code'])


def detector(label: str) -> dict:
    """A detector's crystal, cover and backscatter, from its label: ``'NaI-9.5mm'`` (the 3/8 inch NaI(Tl) crystal of
    most general purpose cameras) or ``'NaI-15.9mm'`` (5/8 inch).

    Args:
        label (str): The detector's label (any case).

    Returns:
        dict: ``crystal_mm``, ``crystal_material``, ``cover_mm``, ``cover_material``, ``backscatter_mm`` and
        ``backscatter_material``: the detector arguments of ``spectarf.DetectorStack``.

    Raises:
        KeyError: The label is not in the table.
    """
    rows = {_key(r['label']): r for r in detectors()}
    row = rows.get(_key(label))
    if row is None:
        raise _not_found('detector', label, [r['label'] for r in detectors()])
    return {k: row[k] for k in ('crystal_mm', 'crystal_material', 'cover_mm', 'cover_material', 'backscatter_mm',
                                'backscatter_material')}


def camera(name: str) -> dict:
    """A camera's energy and intrinsic spatial resolution, from its name (e.g. ``'Siemens Symbia T'``).

    The energy resolution is the parameters of its model; the only model so far is Morphis et al.'s,
    :math:`\\text{FWHM} = a + b\\sqrt{E + cE^2}` (keV), whose parameters ``a_keV``, ``b`` and ``c_per_keV`` are the
    arguments of ``spectarf.EnergyResolution.geb``.

    Args:
        name (str): The camera's name or one of its other names (any case and spacing).

    Returns:
        dict: ``name``, ``energy_resolution_model``, ``energy_resolution`` (the model's parameters) and
        ``intrinsic_fwhm_mm_140`` (the intrinsic spatial resolution, FWHM at 140 keV, mm).

    Raises:
        KeyError: The camera is not in the table.
    """
    for row in cameras():
        if _key(name) in {_key(n) for n in [row['name'], *row.get('aliases', [])]}:
            model = dict(row['energy_resolution'])
            kind = model.pop('model')
            model.pop('source', None)
            return dict(name=row['name'], energy_resolution_model=kind, energy_resolution=model,
                        intrinsic_fwhm_mm_140=float(row['intrinsic_fwhm_mm_140']))
    raise _not_found('camera', name, [n for r in cameras() for n in [r['name'], *r.get('aliases', [])]])


def energy_resolution_fwhm_keV(model: str, energy_keV, **parameters):
    """The FWHM (keV) of an energy resolution model at ``energy_keV``: ``'Morphis'``, :math:`a + b\\sqrt{E + cE^2}`.

    Args:
        model (str): The model's name (``camera(...)['energy_resolution_model']``).
        energy_keV (float | np.ndarray): The energy (keV).
        **parameters: The model's parameters (``camera(...)['energy_resolution']``).

    Returns:
        float | np.ndarray: The FWHM (keV).
    """
    import numpy as np
    if model != 'Morphis':
        raise ValueError(f'unknown energy resolution model {model!r}')
    e = np.asarray(energy_keV, dtype=np.float64)
    return parameters['a_keV'] + parameters['b'] * np.sqrt(e + parameters['c_per_keV'] * e ** 2)
