"""``load_mask``: regions of interest (RTSTRUCT contours, NIfTI and NRRD label images) as a label image."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

from pytomography.metadata.metadata import FRAMES
from ._image import Image, reorient_to_lps


def _is_rtstruct(path: Path) -> bool:
    if path.is_dir() or not path.exists():
        return False
    try:
        import pydicom
        return str(pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=["Modality"]).get("Modality", "")) == "RTSTRUCT"
    except Exception:
        return False


def _series_folder(reference) -> str:
    """The folder of the CT series an RTSTRUCT was drawn on (rt_utils reads a series from its folder)."""
    if isinstance(reference, (list, tuple)):
        folders = {os.path.dirname(os.path.abspath(f)) for f in reference}
        if len(folders) != 1:
            raise ValueError("the reference CT's files must be in one folder (rt_utils reads the series from it)")
        return folders.pop()
    return os.fspath(reference)


#: the first value of a set of overlapping masks (the masks themselves are 1, 2, 3...)
_OVERLAP_VALUES = 1_000_000


def _labels_from_masks(masks, shape) -> tuple[np.ndarray, dict]:
    """One integer label image from boolean masks ``(name, mask)``: mask n has the value n (from 1), and each set of
    masks that overlap has a value of its own (from 1,000,000), so that ``labels[name]`` (an int, or a tuple of the
    values whose voxels are in the mask) gives back every mask exactly."""
    data = np.zeros(shape, dtype=np.int32)
    members: dict[int, frozenset] = {}        # value -> the masks (by number) its voxels are in
    overlaps: dict[frozenset, int] = {}
    names = []
    for name, mask in masks:
        number = len(names) + 1
        names.append(name)
        members[number] = frozenset([number])
        overlap = mask & (data != 0)
        for value in np.unique(data[overlap]):
            key = members[int(value)] | {number}
            if key not in overlaps:
                overlaps[key] = _OVERLAP_VALUES + len(overlaps)
                members[overlaps[key]] = key
            data[overlap & (data == value)] = overlaps[key]
        data[mask & (data == 0)] = number
    present = set(np.unique(data).tolist())
    labels = {}
    for number, name in enumerate(names, start=1):
        values = [number] + sorted(v for v, m in members.items() if v != number and number in m and v in present)
        labels[name] = values[0] if len(values) == 1 else tuple(values)
    return data, labels


def _rtstruct(path: Path, reference, rois) -> Image:
    if reference is None:
        raise ValueError("an RTSTRUCT is drawn on a CT: give the CT series as reference= (its folder or its files)")
    from rt_utils import RTStructBuilder
    from ._load import load_image
    folder = _series_folder(reference)
    rtstruct = RTStructBuilder.create_from(dicom_series_path=folder, rt_struct_path=str(path))
    names = rtstruct.get_roi_names() if rois is None else list(rois)
    # rt_utils fills the CT's pixels as written, slices in ascending order along their normal: the CT's own affine
    # (with the vendor table's rules), its slices in that order (reorient=False)
    ct = load_image(reference if isinstance(reference, (list, tuple)) else folder, reorient=False)

    def masks():
        for name in names:
            try:
                mask = rtstruct.get_roi_mask_by_name(name)
            except Exception as error:   # an ROI without contours
                warnings.warn(f"the RTSTRUCT's ROI {name!r} could not be drawn ({error}); it is left out", stacklevel=4)
                continue
            mask = np.ascontiguousarray(mask.transpose(1, 0, 2))            # (rows, columns, slices) to (x, y, z)
            if mask.shape != ct.shape:
                raise ValueError(f"the ROI {name!r} has shape {mask.shape}, the CT {ct.shape}")
            yield name, mask
    data, labels = _labels_from_masks(masks(), ct.shape)
    meta = {"rtstruct": str(path), "reference": folder}
    return reorient_to_lps(Image(data, ct.affine, units="label", modality="RTSTRUCT", meta=meta, frame=ct.frame,
                                 labels=labels))


def load_mask(source, *, reference=None, rois=None, frame: str | None = None) -> Image:
    """Regions of interest as a label image, to put on any grid with ``.on(object_meta, label=...)``::

        liver = io.load_mask(file_RT, reference=files_CT).on(object_meta, label='liver', threshold=0.5)
        spheres = io.load_mask('fdg_spheres.seg.nrrd').on(object_meta, label=3)

    * An RTSTRUCT is drawn on its CT (``reference``) with rt_utils, so the mask has the CT's voxels and geometry. Each
      ROI has its own value; voxels of ROIs that overlap get further values, and ``labels[name]`` is then a tuple of
      every value in the ROI.
    * A NIfTI or NRRD label image keeps its own geometry; 3D Slicer's segment names become ``labels``, and a label
      image without names gets ``labels`` ``{'1': 1, ...}`` from its values.

    Args:
        source: an RTSTRUCT file, or a NIfTI or NRRD label image (anything :func:`~pytomography.io.load_image` reads).
        reference: for an RTSTRUCT, the CT series it was drawn on: its folder, or its files (in one folder).
        rois (Sequence[str], optional): for an RTSTRUCT, the ROIs to draw (each takes a few seconds). Defaults to None:
            all of them.
        frame (str, optional): ``'patient'`` or ``'scanner'``, to override the frame the file implies (see
            :func:`~pytomography.io.load_image`; an RTSTRUCT has its CT's). Defaults to None: the file's.

    Returns:
        Image: integer labels (``units='label'``), with ``labels``: name -> value (or values).
    """
    if frame is not None and frame not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}, not {frame!r}")
    if not isinstance(source, Image) and not isinstance(source, (list, tuple)):
        path = Path(os.fspath(source))
        if _is_rtstruct(path):
            mask = _rtstruct(path, reference, rois)
            if frame is not None:
                mask.frame = frame
            return mask
    if rois is not None:
        raise ValueError("rois= is for RTSTRUCT files; use label= of Image.on to pick a label")
    from ._load import load_image
    image = source if isinstance(source, Image) else load_image(source)
    data = np.asarray(image.data)
    if not np.all(np.isfinite(data)) or not np.all(data == np.round(data)):
        raise ValueError("a mask holds integer labels; this image does not")
    labels = image.labels
    if not labels:
        labels = {str(int(v)): int(v) for v in np.unique(data) if v != 0}
    return Image(data.astype(np.int32), image.affine, units="label", modality=image.modality, meta=image.meta,
                 frame=image.frame if frame is None else frame, labels=labels)
