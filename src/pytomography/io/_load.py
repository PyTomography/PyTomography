"""``load_image``: any image file into an :class:`~pytomography.io.Image`."""
from __future__ import annotations

import os
from pathlib import Path

from ._image import Image, reorient_to_lps


def _kind(path: Path) -> str:
    name = path.name.lower()
    if path.is_dir():
        return 'dicom'
    if name.endswith(('.nii', '.nii.gz')):
        return 'nifti'
    if name.endswith(('.nrrd', '.nhdr')):
        return 'nrrd'
    if name.endswith(('.mhd', '.mha')):
        return 'metaimage'
    with open(path, 'rb') as f:
        head = f.read(132)
    if head.startswith(b'NRRD'):
        return 'nrrd'
    if head.lstrip().lstrip(b'!').upper().startswith(b'INTERFILE'):
        return 'interfile'
    if name.endswith(('.ict', '.a00', '.v', '.raw', '.img')):
        for suffix in ('.hct', '.h00', '.hv', '.v.hdr', '.hdr', '.h33'):
            header = path.with_name(path.name.rsplit('.', 1)[0] + suffix)
            if header.exists():
                raise ValueError(f'{path.name} is the data file of an Interfile image: pass its header, {header.name}')
    return 'dicom'


def load_image(source, *, layout: str | None = None, units: str | None = None, reorient: bool = True) -> Image:
    """Read an image file into an :class:`~pytomography.io.Image`: its values (x, y, z, on the CPU), its affine
    (voxel index to LPS mm), its units, modality and what the source says about itself.

    Args:
        source: one of
            * a DICOM series: a folder, or a list of files (CT, MR, PT... one slice per file, of one series);
            * a single DICOM file: an NM reconstruction (PyTomography's own NM output included), or an enhanced or
              classic multi-frame image;
            * a NIfTI file (``.nii``, ``.nii.gz``): its sform, else its qform;
            * an Interfile header (SIMIND ``.hct``, GATE/STIR ``.hv``, Siemens ``.v.hdr``...);
            * a NRRD file (``.nrrd``, ``.seg.nrrd``; 3D Slicer segment names become ``labels``).
        layout (str, optional): for a SIMIND density map (``.hct``), and required for one: ``'mu'`` or
            ``'mu-castor'``, as simind.ini index 22 was set (3 or 5) when the map was made. The header does not
            record it.
        units (str, optional): the units of the values when the file does not say (e.g. ``'1/cm'`` for a GATE mu
            map), or to override what it says.
        reorient (bool): permute and flip the axes (never interpolating) so that i, j and k run as close as possible
            to +x, +y and +z (patient left, posterior, superior). Defaults to True.

    Returns:
        Image: the image.
    """
    from ._formats import dicom, interfile, nifti, nrrd
    if isinstance(source, Image):
        fields = dict(data=source.data, affine=source.affine, units=source.units, modality=source.modality,
                      meta=source.meta, frame=source.frame, labels=source.labels)
    elif isinstance(source, (list, tuple)):
        fields = dicom.read_series(source)
    else:
        path = Path(os.fspath(source))
        if not path.exists():
            raise FileNotFoundError(path)
        kind = _kind(path)
        if kind == 'metaimage':
            raise NotImplementedError('MetaImage (.mhd, .mha) arrives in PyTomography 4.1')
        if layout is not None and kind != 'interfile':
            raise ValueError(f'layout= is for SIMIND density maps (Interfile); {path.name} is not one')
        if kind == 'nifti':
            fields = nifti.read_image(path)
        elif kind == 'nrrd':
            fields = nrrd.read_image(path)
        elif kind == 'interfile':
            fields = interfile.read_image(path, layout=layout)
        else:
            fields = dicom.read_series(path)
    image = Image(**fields)
    if units is not None:
        image.units = units
    return reorient_to_lps(image) if reorient else image
