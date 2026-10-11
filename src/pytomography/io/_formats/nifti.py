"""NIfTI-1 and NIfTI-2 images: the file's own affine (sform, else qform), from RAS to LPS (#38)."""
from __future__ import annotations

import re
import warnings

import numpy as np

RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
_TO_MM = {'mm': 1.0, 'meter': 1000.0, 'micron': 1e-3, 'unknown': 1.0}


def read_image(path) -> dict:
    """A NIfTI image as a dict of the :class:`~pytomography.io.Image` fields. The affine is the sform when its code is
    set, else the qform (qfac included), else the voxel sizes alone (with a warning). A sform or qform of code 1
    (scanner anatomy) puts the image in the patient's frame; other codes (aligned, a template) in frame
    ``'scanner'``."""
    import nibabel as nib
    img = nib.load(str(path))
    sform, scode = img.get_sform(coded=True)
    qform, qcode = img.get_qform(coded=True)
    if scode:
        A, code = sform, int(scode)
    elif qcode:
        A, code = qform, int(qcode)
    else:
        warnings.warn(f'{path}: neither a sform nor a qform is set, so the image is placed from its voxel sizes only, '
                      'centred on the scanner')
        A, code = img.header.get_base_affine(), 0
    unit = img.header.get_xyzt_units()[0] or 'unknown'
    A = np.asarray(A, dtype=np.float64).copy()
    A[:3, :] *= _TO_MM.get(unit, 1.0)
    data = np.asanyarray(img.dataobj)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f'{path}: a 3D image is expected, the file holds shape {data.shape}')
    descrip = bytes(img.header['descrip']).split(b'\0')[0].decode('ascii', 'replace').strip()
    units = None
    found = re.search(r'\[([^\]]+)\]$', descrip)
    if found:                                          # save_nifti keeps the units at the end of the description
        units = found.group(1)
    meta = {'description': descrip, 'xform_code': code}
    return dict(data=data, affine=RAS_TO_LPS @ A, units=units, modality=None, meta=meta,
                frame='patient' if code == 1 else 'scanner')
