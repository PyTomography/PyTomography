"""Interfile 3.3 headers and images, from any producer (SIMIND, GATE/STIR, Siemens JSRecon).

Keys are matched exactly after normalisation (case, leading '!' or '%', spaces and the producers' spelling variants),
and a missing key raises. Comment lines (';' or ';#') are not keys; SIMIND's comment keys, which hold facts found
nowhere else (energy windows, the photon energy, the collimator), are kept apart in ``Interfile.comments``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# spellings of the same key by different producers -> one spelling
_ALIASES = [
    (re.compile(r'^scale factor \(mm/pixel\)'), 'scaling factor (mm/pixel)'),
    (re.compile(r'^image data byte order$'), 'imagedata byte order'),
]
_MISSING = object()


def normalise_key(key: str) -> str:
    """A key in one spelling: lower case, no leading '!' or '%', single spaces, a space before '[n]'."""
    key = key.strip().lstrip('!%').strip().lower()
    key = re.sub(r'\s+', ' ', key)
    key = re.sub(r'\s*\[\s*(\d+)\s*\]', r' [\1]', key)
    for pattern, replacement in _ALIASES:
        key = pattern.sub(replacement, key)
    return key


class Interfile():
    """The keys of an Interfile header.

    Args:
        path (str | Path): the header file.

    Attributes:
        keys (dict): every key (normalised) and its value as written (the first one when a key repeats).
        comments (dict): the keys of comment lines (starting ';' or ';#'), normalised the same way.
    """
    def __init__(self, path) -> None:
        self.path = Path(path)
        self.keys, self.comments, self._all = {}, {}, {}
        text = self.path.read_text(errors='replace')
        if not text.lstrip().lstrip('!').upper().startswith('INTERFILE'):
            raise ValueError(f'{self.path} is not an Interfile header (no "!INTERFILE" first line)')
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or ':=' not in stripped:
                continue
            target = self.keys
            if stripped.startswith(';'):                 # a comment (SIMIND keeps facts in ';' and ';#' lines)
                target, stripped = self.comments, stripped.lstrip(';#')
            key, value = stripped.split(':=', 1)
            key = normalise_key(key)
            value = value.strip()
            if target is self.keys:
                self._all.setdefault(key, []).append(value)
            target.setdefault(key, value)

    def __contains__(self, key: str) -> bool:
        return normalise_key(key) in self.keys

    def get(self, key: str, kind=str, default=_MISSING):
        """The value of ``key`` as ``kind`` (str, int or float); ``default`` when the header has no such key, which
        otherwise raises a KeyError."""
        return self._value(self.keys, key, kind, default, 'key')

    def comment(self, key: str, kind=str, default=_MISSING):
        """The value of the comment key ``key`` (a ';' or ';#' line), like :meth:`get`."""
        return self._value(self.comments, key, kind, default, 'comment key')

    def get_all(self, key: str) -> list:
        """Every value of a key that repeats (as strings)."""
        return list(self._all.get(normalise_key(key), []))

    def _value(self, table: dict, key: str, kind, default, what: str):
        name = normalise_key(key)
        if name not in table or table[name] == '':
            if default is _MISSING:
                raise KeyError(f"{self.path.name}: no {what} '{key}'")
            return default
        value = table[name]
        if kind is str:
            return value
        try:
            if kind is int:
                return int(float(value))
            return kind(value)
        except ValueError as e:
            raise ValueError(f"{self.path.name}: '{key}' is {value!r}, not a {kind.__name__}") from e

    @property
    def producer(self) -> str:
        """'simind', 'siemens' or 'stir' (GATE and STIR images, and anything else)."""
        origin = self.get('originating system', default='').lower()
        if 'simind' in origin or 'ljungberg' in self.get('program author', default='').lower():
            return 'simind'
        if 'sms-mi header name space' in self.keys or 'jsrecon' in self.get('comment', default='').lower():
            return 'siemens'
        return 'stir'

    @property
    def data_file(self) -> Path:
        return self.path.parent / self.get('name of data file')

    def dtype(self) -> np.dtype:
        """The data's numpy dtype, from 'number format', 'number of bytes per pixel' and the byte order."""
        fmt = self.get('number format', default='float').lower().replace(' ', '')
        size = self.get('number of bytes per pixel', int, default=4)
        order = self.get('imagedata byte order', default='LITTLEENDIAN').upper()
        endian = '>' if order.startswith('BIG') else '<'
        if 'float' in fmt:
            kind = 'f'
            if fmt == 'longfloat':
                size = 8
        elif fmt.startswith('signed') or fmt in ('int', 'integer'):
            kind = 'i'
        elif fmt.startswith('unsigned'):
            kind = 'u'
        else:
            raise ValueError(f"{self.path.name}: number format '{fmt}' is not supported")
        return np.dtype(f'{endian}{kind}{size}')

    def read_data(self, count: int) -> np.ndarray:
        """``count`` values from the data file, after 'data offset in bytes'."""
        offset = self.get('data offset in bytes', int, default=0)
        data = np.fromfile(self.data_file, dtype=self.dtype(), count=count, offset=offset)
        if data.size != count:
            raise ValueError(f'{self.data_file.name}: expected {count} values, found {data.size}')
        return data


#: SIMIND density-map layouts (simind.ini index 22): the axes flipped after reading x fastest, to reach +x, +y, +z.
SIMIND_LAYOUTS = {'mu': (False, False, True),          # index 22 = 3: first slice superior
                  'mu-castor': (False, True, True)}    # index 22 = 5: y runs the other way too


def _centred(dr, shape) -> np.ndarray:
    A = np.diag([*dr, 1.0])
    A[:3, 3] = [-(n - 1) / 2 * d for n, d in zip(shape, dr)]
    return A


def read_image(path, layout: str | None = None) -> dict:
    """An image stored as Interfile, as a dict of the :class:`~pytomography.io.Image` fields.

    The data are read x fastest, then y and z. The image is centred on the scanner (frame ``'scanner'``): none of
    the producers PyTomography reads states a patient position. Producer rules (the vendor table):

    * SIMIND density and attenuation maps (``.hct``): ``layout`` (``'mu'`` or ``'mu-castor'``, simind.ini index 22)
      is required, because the header does not record it; values in 1/cm for attenuation maps.
    * Siemens JSRecon (``.v.hdr``): 'quantification units' gives the units.
    * GATE/STIR (``.hv``): 'first pixel offset' is kept in ``meta`` only; the orientation is unverified.
    """
    from .. import _vendors
    header = Interfile(path)
    producer = header.producer
    if producer == 'simind' and ('number of projections' in header.keys or 'extent of rotation' in header.keys):
        raise ValueError(f'{header.path.name} holds SIMIND projections: read them with io.SPECT.read_simind')
    if layout is not None and producer != 'simind':
        raise ValueError(f'layout= is for SIMIND density maps; {header.path.name} is not one')
    shape = tuple(header.get(f'matrix size [{n}]', int) for n in (1, 2, 3))
    dr = [header.get(f'scaling factor (mm/pixel) [{n}]', float, default=None) for n in (1, 2, 3)]
    if dr[2] is None and producer == 'simind':      # SIMIND V7 writes the slice spacing only as a comment
        dr[2] = header.comment('scaling factor (mm/pixel) [3]', float,
                               default=header.comment('ct spacing between slices', float, default=None))
    if None in dr:
        raise KeyError(f"{header.path.name}: no 'scaling factor (mm/pixel) [n]' for every axis")
    data = header.read_data(int(np.prod(shape))).reshape(shape[::-1]).transpose(2, 1, 0)
    units, meta = None, {'producer': producer, 'header': str(header.path)}
    if producer == 'simind':
        rule = _vendors.use('simind-density-layout', header.path.name)
        if layout is None:
            raise ValueError(f'{header.path.name} is a SIMIND density map, whose layout the header does not record: '
                             "pass layout='mu' (simind.ini index 22 = 3) or layout='mu-castor' (index 22 = 5). "
                             f'({rule.rule})')
        if layout not in SIMIND_LAYOUTS:
            raise ValueError(f'layout must be one of {sorted(SIMIND_LAYOUTS)}, not {layout!r}')
        flips = [axis for axis, flip in enumerate(SIMIND_LAYOUTS[layout]) if flip]
        data = np.flip(data, axis=flips)
        if header.comment('units of data (ect)', default='').lower() == 'mu':
            units = '1/cm'
        meta.update(layout=layout, program_version=header.get('program version', default=None))
    elif producer == 'siemens':
        _vendors.use('siemens-mmr-mu-interfile', header.path.name)
        units = header.get('quantification units', default=None)
        meta['patient_orientation'] = header.get('patient orientation', default=None)
    else:
        _vendors.use('gate-stir-interfile-image', header.path.name)
        offsets = [header.get(f'first pixel offset (mm) [{n}]', float, default=None) for n in (1, 2, 3)]
        if any(o is not None for o in offsets):
            meta['first_pixel_offset_mm'] = offsets
    return dict(data=np.ascontiguousarray(data), affine=_centred(dr, shape), units=units, modality=None, meta=meta,
                frame='scanner')
