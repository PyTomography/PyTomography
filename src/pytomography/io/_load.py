"""``load_image``: any image file into an :class:`~pytomography.io.Image`."""
from __future__ import annotations

import os
from pathlib import Path

from pytomography.metadata.metadata import FRAMES
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


def load_image(source, *, layout: str | None = None, units: str | None = None, reorient: bool = True,
               frame: str | None = None) -> Image:
    """Reads an image file into an :class:`~pytomography.io.Image`.

    The image holds the values (axes x, y, z, on the CPU), the affine (voxel index to LPS mm), the units, the modality
    and what the source says about itself. Put it on a reconstruction grid with :meth:`~pytomography.io.Image.on`.

    Args:
        source (str | os.PathLike | list | Image): What to read:

            * a DICOM series: a folder, or a list of files (CT, MR, PT... one slice per file, of one series);
            * a single DICOM file: an NM reconstruction (PyTomography's own NM output included), or an enhanced or
              classic multi-frame image;
            * a NIfTI file (``.nii``, ``.nii.gz``): its sform, else its qform;
            * an Interfile header (SIMIND ``.hct``, GATE/STIR ``.hv``, Siemens ``.v.hdr``...);
            * a NRRD file (``.nrrd``, ``.nhdr``, ``.seg.nrrd``; 3D Slicer segment names become ``labels``);
            * an :class:`~pytomography.io.Image`, which is copied (and reoriented, with ``reorient``).

        layout (str, optional): For a SIMIND density map (``.hct``), and required for one: ``'mu'`` or
            ``'mu-castor'``, as simind.ini index 22 was set (3 or 5) when the map was made. The header does not
            record it. Defaults to None.
        units (str, optional): The units of the values when the file does not say (e.g. ``'1/cm'`` for a GATE mu
            map), or to override what it says. Defaults to None (the file's).
        reorient (bool, optional): Permute and flip the axes (never interpolating) so that i, j and k run as close as
            possible to +x, +y and +z (patient left, posterior, superior). Defaults to True.
        frame (str, optional): ``'patient'`` or ``'scanner'``, to override the frame the file implies. Files say it
            where their format allows: a NIfTI's sform (or qform) code, 1 for ``'patient'`` and any other for
            ``'scanner'`` (what :func:`~pytomography.io.save_nifti` writes); a NRRD's ``pytomography_frame:=`` field
            (``'patient'`` without one); DICOM is always ``'patient'``; Interfile ``'scanner'``. Defaults to None:
            the file's.

    Returns:
        Image: The image.

    Raises:
        FileNotFoundError: If ``source`` does not exist.
        ValueError: If ``layout`` is given for a file that is not Interfile, ``frame`` is not ``'patient'`` or
            ``'scanner'``, or ``source`` is the data file of an Interfile image instead of its header.
        NotImplementedError: For MetaImage files (``.mhd``, ``.mha``).

    Example:
        .. code-block:: python

            ct = io.load_image('CT_folder')                   # a DICOM series
            mu = io.attenuation_map(ct, object_meta, energy=208)
            mask = io.load_mask('liver.seg.nrrd').on(object_meta, label='liver', threshold=0.5)
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
    if frame is not None:
        if frame not in FRAMES:
            raise ValueError(f"frame must be one of {FRAMES}, not {frame!r}")
        fields['frame'] = frame
    image = Image(**fields)
    if units is not None:
        image.units = units
    return reorient_to_lps(image) if reorient else image
