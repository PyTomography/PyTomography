"""Reading SIMIND Monte Carlo projections: Interfile ``.h00`` headers and ``.a00`` data, one pair per energy window.

When the source was simulated organ by organ there is one pair per organ and window. :func:`read_simind` reads them
into PyTomography's standard form, and :func:`simind_psf` gives the collimator's PSF from a header.
"""
from __future__ import annotations

import os
import re
from typing import Sequence

import numpy as np
import torch

import pytomography
from pytomography.metadata.SPECT import EnergyWindow, SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta
from .. import _vendors
from .._formats.interfile import Interfile
from ._energy import _mu_per_mm, _psf_meta, photon_energy


def simind_version(header: Interfile) -> int:
    """The major version of the SIMIND that wrote a header: the number after 'V' in 'program version'.

    Args:
        header (Interfile): The header, as PyTomography's Interfile reader reads it.

    Returns:
        int: The major version, e.g. 8.

    Raises:
        ValueError: If 'program version' has no version number.
    """
    text = header.get('program version')
    found = re.search(r'[vV]\s*(\d+)', text)
    if not found:
        raise ValueError(f"{header.path.name}: cannot read the SIMIND version from 'program version := {text}'")
    return int(found.group(1))


def _mm_per_unit(header: Interfile) -> float:
    """SIMIND's Radius and collimator keys are in cm before version 8 and in mm from version 8 on."""
    _vendors.use('simind-units')
    return 10.0 if simind_version(header) < 8 else 1.0


def _header(path) -> Interfile:
    header = Interfile(path)
    if header.producer != 'simind':
        raise ValueError(f'{header.path.name} was not written by SIMIND')
    if 'number of projections' not in header:
        raise ValueError(f'{header.path.name} holds no SIMIND projections (a density map is read with io.load_image)')
    return header


def _energy_window(header: Interfile) -> EnergyWindow:
    """The window of a SIMIND projection file: its bounds (comment keys) and its photon energy (the nuclide's line in
    it, else SIMIND's photon energy when inside it, else its centre)."""
    lower = header.comment('energy window lower level', float)
    upper = header.comment('energy window upper level', float)
    nuclide = header.comment('radionuclide', default=None)
    simulated = header.comment('simind: photon energy', float, default=None)
    return EnergyWindow(header.path.stem, [(lower, upper)], photon_energy([(lower, upper)], nuclide, simulated))


def _projections(header: Interfile) -> np.ndarray:
    """The views of one file, [view, r, z] in acquisition order, z rows flipped to run foot to head."""
    n = header.get('total number of images', int)
    shape = (n, header.get('matrix size [2]', int), header.get('matrix size [1]', int))
    data = header.read_data(int(np.prod(shape))).astype(np.float32).reshape(shape)
    return np.transpose(data[:, ::-1], (0, 2, 1))          # row 0 is at +z


def _radii(radii_file, n_views: int, Lr: int) -> np.ndarray:
    """The detector distances (mm) of a non-circular orbit from a SIMIND ``.cor`` file, in acquisition order.

    The file has one line per view, in acquisition order. SIMIND writes two columns (index 42 negative, the
    auto-contour orbit): the distance from the rotation axis to the detector surface in cm, and the centre of
    rotation in pixels from the edge of the projection (half the matrix size, the centre of the projections). The
    files SIMIND reads (its third argument, as :mod:`pytomography.utils.simind_mc` writes them) may hold the distances
    alone.
    """
    name = os.path.basename(radii_file)
    table = np.loadtxt(radii_file, ndmin=2)
    if table.shape[1] > 2:
        raise ValueError(f'{name}: {table.shape[1]} columns; a SIMIND .cor file has the distance (cm) and the centre '
                         'of rotation (pixels)')
    if table.shape[0] != n_views:
        raise ValueError(f'{name}: {table.shape[0]} radii for {n_views} views')
    if table.shape[1] == 2 and not np.allclose(table[:, 1], Lr / 2):
        raise ValueError(f'{name}: the centre of rotation (second column, {np.unique(table[:, 1])} pixels) is not the '
                         f'centre of the {Lr} pixel wide projections ({Lr / 2}); PyTomography puts the rotation axis at '
                         'the centre of the projections')
    return table[:, 0] * 10                                  # SIMIND's .cor files are in cm


def _geometry(header: Interfile, radii_file=None) -> dict:
    _vendors.use('simind-projection-frame', header.path.name)
    n = header.get('number of projections', int)
    extent = header.get('extent of rotation', float)
    start = header.get('start angle', float)
    direction = header.get('direction of rotation').upper()
    angles = np.linspace(-start, -start + extent, n, endpoint=False)
    if direction == 'CW':
        if start != 0:
            _vendors.use('simind-clockwise-start', header.path.name)
        angles = -angles
    angles = angles % 360
    orbit_file = header.comment('non-uniform orbit file', default=None)
    if radii_file is None and orbit_file is not None:       # a non-circular orbit: the header has no radius
        radii_file = header.path.parent / orbit_file
        if not radii_file.exists():
            raise FileNotFoundError(f'{header.path.name}: its orbit file {orbit_file} is not next to it; pass radii_file')
    if radii_file is not None:
        radii = _radii(radii_file, n, header.get('matrix size [1]', int))
    else:
        radius = header.get('radius', float, default=None)
        if radius is None:
            radius = header.comment('radius', float)
        radii = np.full(n, radius * _mm_per_unit(header))
    if len(radii) != n:
        raise ValueError(f'{len(radii)} radii for {n} views')
    order = np.argsort(angles, kind='stable')
    duration = header.get('time per projection (sec)', float, default=None)
    return dict(order=order, angles=angles[order], radii=radii[order],
                dr=header.get('scaling factor (mm/pixel) [1]', float), dz=header.get('scaling factor (mm/pixel) [2]', float),
                Lr=header.get('matrix size [1]', int), Lz=header.get('matrix size [2]', int), duration=duration,
                nuclide=header.comment('radionuclide', default=None))


def read_simind(headers, energy_window: int | None = None, *, weights: Sequence[float] | None = None,
                radii_file: str | None = None):
    """Reads SIMIND projections (one ``.h00`` header per energy window) into PyTomography's standard form.

    * The version is read as an integer (V7 keys in cm, V8 and later mm); lengths become mm.
    * Views are sorted by angle (β as for DICOM); rows are flipped so that z runs foot to head.
    * The radii are the header's radius for every view, or, for a non-circular orbit, those of SIMIND's ``.cor``
      orbit file (``radii_file``, or the file the header names as its non-uniform orbit file).
    * Every window's bounds and photon energy go into ``proj_meta.energy_windows`` (the window's nuclide line, not the
      simulation's main photon energy); the values are counts per second per MBq (``proj_meta.units``), and
      ``proj_meta.frame_duration`` is the header's time per projection.
    * The grid is the projections' (Lr x Lr x Lz), centred on the rotation axis (frame ``'scanner'``).

    Args:
        headers (str | Sequence[str] | Sequence[Sequence[str]]): The ``.h00`` file of one window, of every window, or
            for each window the files of every organ (added up with ``weights``).
        energy_window (int, optional): Index of the window to return. Defaults to None: all of them.
        weights (Sequence[float], optional): For organ-by-organ simulations, each organ's weight (e.g. its activity).
            Defaults to None (organs added up with weight 1).
        radii_file (str, optional): SIMIND's ``.cor`` file of a non-circular orbit, as SIMIND writes it: one line per
            view, in acquisition order, with the distance from the rotation axis to the detector surface in cm (read
            as mm), and the centre of rotation in pixels, which must be the centre of the projections (files with the
            distances alone, as :mod:`pytomography.utils.simind_mc` writes them for SIMIND, are read too). Defaults
            to None: the orbit file the header names, next to it, if it names one; else the header's radius for every
            view.

    Returns:
        tuple[torch.Tensor, SPECTObjectMeta, SPECTProjMeta]: The projections [W, θ, r, z] (or [θ, r, z] for one
        ``energy_window``) on ``pytomography.device``, the grid (mm) and the projection metadata (radii in mm).

    Raises:
        ValueError: If a file was not written by SIMIND or holds no projections, the weights are not one per organ
            file, the files' shapes differ, or the orbit file does not have one line (and at most two columns) per view
            or puts the centre of rotation elsewhere than the centre of the projections.
        FileNotFoundError: If the header names an orbit file that is not next to it and ``radii_file`` is not given.
    """
    if isinstance(headers, (str, os.PathLike)):
        headers = [headers]
    windows = []
    for entry in headers:
        organs = [entry] if isinstance(entry, (str, os.PathLike)) else list(entry)
        if weights is not None and len(weights) != len(organs):
            raise ValueError(f'{len(weights)} weights for {len(organs)} organ files')
        windows.append([_header(h) for h in organs])
    first = windows[0][0]
    g = _geometry(first, radii_file)
    stacks = []
    for organs in windows:
        total = 0
        for j, header in enumerate(organs):
            views = _projections(header)
            if views.shape != (len(g['angles']), g['Lr'], g['Lz']):
                raise ValueError(f'{header.path.name}: shape {views.shape} differs from {first.path.name}')
            total = total + views * (1.0 if weights is None else float(weights[j]))
        stacks.append(total[g['order']])
    projections = torch.tensor(np.stack(stacks)).to(pytomography.dtype).to(pytomography.device)
    object_meta = SPECTObjectMeta((g['dr'], g['dr'], g['dz']), (g['Lr'], g['Lr'], g['Lz']), frame='scanner', units=None)
    proj_meta = SPECTProjMeta((g['Lr'], g['Lz']), (g['dr'], g['dz']), g['angles'], g['radii'],
                              energy_windows=[_energy_window(organs[0]) for organs in windows],
                              frame_duration=g['duration'], nuclide=g['nuclide'], units='counts/s/MBq')
    return (projections if energy_window is None else projections[energy_window]), object_meta, proj_meta


def simind_psf(header: str, photon_energy: float | None = None, min_sigmas: float = 3) -> SPECTPSFMeta:
    """The Gaussian collimator PSF of a SIMIND simulation, from the collimator and camera keys of its header.

    The hole diameter, the collimator thickness and the camera's intrinsic FWHM (at 140 keV, scaled as
    :math:`E^{-1/2}`) come from the header; the PSF is that of a lead parallel-hole collimator, as for
    :func:`~pytomography.io.SPECT.collimator_psf`, at the photon energy of the header's window (or ``photon_energy``).

    Args:
        header (str): A ``.h00`` file of the simulation (its window sets the photon energy).
        photon_energy (float, optional): The photon energy in keV, instead of the window's. Defaults to None.
        min_sigmas (float, optional): Kernel half-width in sigmas. Defaults to 3.

    Returns:
        SPECTPSFMeta: The PSF, its width as a function of the distance in mm.

    Raises:
        ValueError: If the file was not written by SIMIND or holds no projections.
    """
    h = _header(header)
    unit = _mm_per_unit(h)
    if photon_energy is None:
        photon_energy = _energy_window(h).photon_energy
    hole_diameter = h.comment('collimator hole diameter', float) * unit
    hole_length = h.comment('collimator thickness', float) * unit
    intrinsic_140 = h.comment('intrinsic fwhm for the camera', float, default=0.0) * unit
    intrinsic = intrinsic_140 * (photon_energy / 140) ** -0.5
    return _psf_meta(hole_diameter, hole_length, _mu_per_mm('lead', photon_energy), intrinsic, 'gaussian', min_sigmas)
