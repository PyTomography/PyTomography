"""NRRD images and 3D Slicer segmentations (.nrrd, .seg.nrrd, .nhdr): the header's space, directions and origin.

The image is in the patient's frame unless the header has the key-value field ``pytomography_frame:=scanner`` (an
image placed only relative to the scanner's centre, such as a mask drawn on a reconstruction with no patient frame).

A segmentation whose segments overlap is stored by 3D Slicer in layers: a 4D file whose first axis is the layer, each
segment a value in one layer (``Segment{n}_Layer``, ``Segment{n}_LabelValue``). It is read as one label image in which
each set of overlapping segments has a value of its own, so that ``labels[name]`` (a tuple of values) gives back each
segment exactly, as for an RTSTRUCT's overlapping ROIs."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np

from pytomography.metadata.metadata import FRAMES

_TYPES = {'signed char': 'i1', 'int8': 'i1', 'int8_t': 'i1', 'uchar': 'u1', 'unsigned char': 'u1', 'uint8': 'u1',
          'uint8_t': 'u1', 'short': 'i2', 'short int': 'i2', 'signed short': 'i2', 'signed short int': 'i2',
          'int16': 'i2', 'int16_t': 'i2', 'ushort': 'u2', 'unsigned short': 'u2', 'unsigned short int': 'u2',
          'uint16': 'u2', 'uint16_t': 'u2', 'int': 'i4', 'signed int': 'i4', 'int32': 'i4', 'int32_t': 'i4',
          'uint': 'u4', 'unsigned int': 'u4', 'uint32': 'u4', 'uint32_t': 'u4', 'longlong': 'i8', 'long long': 'i8',
          'int64': 'i8', 'int64_t': 'i8', 'ulonglong': 'u8', 'unsigned long long': 'u8', 'uint64': 'u8',
          'uint64_t': 'u8', 'float': 'f4', 'double': 'f8'}
# each word of a NRRD space name, and the LPS axis and sign it stands for
_SPACE_WORDS = {'left': (0, 1), 'right': (0, -1), 'posterior': (1, 1), 'anterior': (1, -1), 'superior': (2, 1),
                'inferior': (2, -1)}
_SPACE_SHORT = {'RAS': 'right-anterior-superior', 'LAS': 'left-anterior-superior', 'LPS': 'left-posterior-superior'}


def _vector(text: str) -> np.ndarray | None:
    text = text.strip()
    if text == 'none':
        return None
    return np.array([float(v) for v in text.strip('()').split(',')])


def _to_lps(space: str) -> np.ndarray:
    """The 3 x 3 matrix taking coordinates in the NRRD space to LPS."""
    space = _SPACE_SHORT.get(space.upper(), space.lower())
    words = space.split('-')
    if len(words) != 3 or any(w not in _SPACE_WORDS for w in words):
        raise ValueError(f"NRRD space '{space}' is not a patient space (left/right, anterior/posterior, "
                         'superior/inferior)')
    M = np.zeros((3, 3))
    for column, word in enumerate(words):
        axis, sign = _SPACE_WORDS[word]
        M[axis, column] = sign
    return M


def read_header(path) -> tuple[dict, dict, bytes, int]:
    """The fields (``key: value``) and key/value pairs (``key:=value``) of a NRRD header, the file's bytes and where
    the data start."""
    raw = Path(path).read_bytes()
    if not raw.startswith(b'NRRD'):
        raise ValueError(f'{path} is not a NRRD file')
    end = raw.find(b'\n\n')
    end = len(raw) if end < 0 else end
    fields, pairs = {}, {}
    for line in raw[:end].decode('latin-1').splitlines()[1:]:
        if not line or line.startswith('#'):
            continue
        if ':=' in line:
            key, value = line.split(':=', 1)
            pairs[key] = value
        elif ': ' in line:
            key, value = line.split(': ', 1)
            fields[key.strip().lower()] = value.strip()
    return fields, pairs, raw, end + 2


def read_image(path) -> dict:
    """A NRRD image as a dict of the :class:`~pytomography.io.Image` fields; a 3D Slicer segmentation gets its
    segment names as ``labels``."""
    path = Path(path)
    fields, pairs, raw, start = read_header(path)
    dimension = int(fields['dimension'])
    sizes = [int(v) for v in fields['sizes'].split()]
    if dimension not in (3, 4):
        raise ValueError(f'{path.name}: a 3D image (or a 3D Slicer segmentation in layers) is expected, the file has '
                         f'{dimension} dimensions')
    dtype = np.dtype(_TYPES[fields['type'].lower()])
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder('>' if fields.get('endian', 'little') == 'big' else '<')
    encoding = fields.get('encoding', 'raw').lower()
    if 'data file' in fields or 'datafile' in fields:
        data_bytes = (path.parent / (fields.get('data file') or fields['datafile'])).read_bytes()
    else:
        data_bytes = raw[start:]
    if encoding in ('gzip', 'gz'):
        data_bytes = gzip.decompress(data_bytes)
    elif encoding != 'raw':
        raise ValueError(f"{path.name}: NRRD encoding '{encoding}' is not supported")
    count = int(np.prod(sizes))
    data = np.frombuffer(data_bytes, dtype=dtype, count=count).reshape(sizes[::-1]).transpose(*range(dimension)[::-1])
    to_lps = _to_lps(fields.get('space', 'left-posterior-superior'))
    directions = [_vector(v) for v in re.findall(r'\([^)]*\)|none', fields['space directions'])]
    layered = dimension == 4
    if layered:  # 3D Slicer's segmentation in layers: the first axis is the layer, with no direction in space
        if directions[0] is not None or any(d is None for d in directions[1:]) or 'Segment0_LabelValue' not in pairs:
            raise ValueError(f'{path.name}: a 4D NRRD is read only as a 3D Slicer segmentation in layers (the first '
                             'axis the layer, with no space direction, and Segment fields)')
        directions = directions[1:]
    A = np.eye(4)
    A[:3, :3] = to_lps @ np.stack(directions, axis=1)
    A[:3, 3] = to_lps @ _vector(fields.get('space origin', '(0,0,0)'))
    labels, layers, n = {}, {}, 0
    while f'Segment{n}_LabelValue' in pairs:
        name = pairs.get(f'Segment{n}_Name', f'Segment_{n + 1}')
        labels[name] = int(pairs[f'Segment{n}_LabelValue'])
        layers[name] = int(pairs.get(f'Segment{n}_Layer', 0))
        n += 1
    if layered:
        from pytomography.io._mask import _labels_from_masks
        layer_data = data
        data, labels = _labels_from_masks(((name, layer_data[layers[name]] == value) for name, value in labels.items()),
                                          layer_data.shape[1:])
    meta = {'nrrd_space': fields.get('space'), 'kinds': fields.get('kinds')}
    # NRRD has no field for it: PyTomography's own key, for an image placed only relative to the scanner's centre
    frame = pairs.get('pytomography_frame', 'patient').strip()
    if frame not in FRAMES:
        raise ValueError(f"{path.name}: pytomography_frame:={frame}; it must be one of {FRAMES}")
    return dict(data=np.array(data), affine=A, units='label' if labels else None, modality=None, meta=meta,
                frame=frame, labels=labels or None)
