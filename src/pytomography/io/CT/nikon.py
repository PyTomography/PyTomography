"""Reader for Nikon XTek cone-beam micro-CT scans.

A scan folder holds the geometry (an ``.xtekct`` file), the stage angle of every projection (an ``.ang`` file) and
one TIFF per projection. :func:`read_nikon` returns the scan in PyTomography's standard form::

    projections, object_meta, proj_meta = io.CT.read_nikon(folder, binning=4, cor_mm=0.24)

The stage turns the sample between a fixed source and flat panel, so in the sample's frame the source and detector
turn the other way: the angles of the ``.ang`` file are negated. The centre of rotation is not in the files; find it
from the data (the sharpest reconstruction) and pass it as ``cor_mm``.
"""
from __future__ import annotations

import configparser
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pytomography.io import _vendors
from pytomography.metadata import ObjectMeta
from pytomography.metadata.CT import CTConeBeamFlatPanelProjMeta


def _scan_files(folder: Path) -> tuple[Path, Path]:
    xtekct = sorted(folder.glob("*.xtekct"))
    if len(xtekct) != 1:
        raise FileNotFoundError(f"{folder} must hold one .xtekct file, not {len(xtekct)}")
    ang = sorted(folder.glob("*.ang"))
    if len(ang) != 1:
        raise FileNotFoundError(f"{folder} must hold one .ang file (the stage angle of every projection), not {len(ang)}")
    return xtekct[0], ang[0]


def _angles(path: Path) -> dict:
    """The stage angle (degrees) of every projection, by its number, from the lines ``number: angle`` of a ``.ang``
    file (after its header line)."""
    angles = {}
    for line in path.read_text(errors="replace").splitlines()[1:]:
        if ":" in line:
            number, angle = line.split(":", 1)
            angles[int(number)] = float(angle)
    if not angles:
        raise ValueError(f"{path} lists no projections")
    return angles


def read_nikon(folder, *, binning: int = 1, cor_mm: float = 0.0, object_meta: ObjectMeta | None = None
               ) -> tuple[torch.Tensor, ObjectMeta, CTConeBeamFlatPanelProjMeta]:
    r"""Read a Nikon XTek cone-beam scan into PyTomography's standard form.

    * The geometry: ``SrcToObject`` (the source's distance from the rotation axis), ``SrcToDetector``,
      ``DetectorPixelSizeX`` and ``DetectorPixelSizeY`` (in mm), and ``WhiteLevel`` (:math:`I_0`, the reading with
      nothing in the beam) of the ``.xtekct`` file's ``[XTekCT]`` section. There is no dark field.
    * The projections: those listed in the ``.ang`` file, in the order of their numbers, read from
      ``<Name><InputSeparator><number:04d>.tif`` (``Name`` and ``InputSeparator`` from the ``.xtekct`` file; they
      default to the file's stem and ``_``). Each is averaged over ``binning`` x ``binning`` blocks of pixels (a
      remainder at the edge is dropped), then turned into line integrals :math:`-\ln(I / I_0)`, with :math:`I` at
      least 1. u runs along the TIFF's columns and v along its rows (v = 0 is the top of the radiograph, +z).
    * The angles: the ``.ang`` file's, negated (the stage turns the sample), in radians.
    * The rotation axis lies ``cor_mm`` from the central ray, at (0, -cor_mm) in the object frame. The grid is centred
      on the central ray of the first view, at the origin of the scanner frame.

    Args:
        folder (str | os.PathLike): The scan's folder, with one ``.xtekct`` file, one ``.ang`` file and the TIFFs.
        binning (int, optional): Detector pixels averaged along each axis of the detector. Defaults to 1.
        cor_mm (float, optional): Offset of the rotation axis from the central ray, in mm at the axis. Defaults to 0.
        object_meta (ObjectMeta | None, optional): Grid to reconstruct on. Only its ``dr`` and ``shape`` are used.
            Defaults to None: voxels of the binned detector pixel size divided by the magnification
            (``SrcToDetector / SrcToObject``), with shape (u pixels, u pixels, v pixels), so that the grid spans the
            detector's field of view at the axis.

    Returns:
        tuple[torch.Tensor, ObjectMeta, CTConeBeamFlatPanelProjMeta]: The projections (line integrals, [views, u, v],
        float32 on the CPU), the object grid (frame ``'scanner'``, units ``'1/mm'``, modality ``'CT'``, centred on
        (0, 0, 0)) and the projection metadata.

    Raises:
        FileNotFoundError: The folder does not hold exactly one ``.xtekct`` file and one ``.ang`` file, or a projection
            listed in the ``.ang`` file has no TIFF.
        ValueError: The ``.xtekct`` file has no ``[XTekCT]`` section, the ``.ang`` file lists no projections, or
            ``binning`` is less than 1.
    """
    from PIL import Image as PILImage
    folder = Path(os.fspath(folder))
    xtekct, ang = _scan_files(folder)
    _vendors.use("nikon-xtek", str(xtekct))
    parser = configparser.ConfigParser()
    parser.optionxform = str                      # the keys keep their capitals
    parser.read(xtekct)
    if "XTekCT" not in parser:
        raise ValueError(f"{xtekct} has no [XTekCT] section")
    scan = parser["XTekCT"]
    source_to_axis = float(scan["SrcToObject"])
    source_to_detector = float(scan["SrcToDetector"])
    pixel = (float(scan["DetectorPixelSizeX"]), float(scan.get("DetectorPixelSizeY", scan["DetectorPixelSizeX"])))
    white_level = float(scan["WhiteLevel"])
    name = scan.get("Name", xtekct.stem)
    separator = scan.get("InputSeparator", "_")
    angles = _angles(ang)
    numbers = sorted(angles)
    binning = int(binning)
    if binning < 1:
        raise ValueError(f"binning must be at least 1, not {binning}")
    views = []
    for number in numbers:
        path = folder / f"{name}{separator}{number:04d}.tif"
        if not path.exists():
            raise FileNotFoundError(f"projection {number} of {ang.name} is not in the folder ({path.name})")
        reading = torch.from_numpy(np.asarray(PILImage.open(path), dtype=np.float32))     # (rows, columns)
        if binning > 1:
            reading = F.avg_pool2d(reading[None, None], binning)[0, 0]
        views.append(-torch.log(reading.clamp(min=1) / white_level))
    projections = torch.stack(views).transpose(1, 2).contiguous()                      # (views, u, v)
    thetas = -torch.tensor([angles[n] for n in numbers], dtype=torch.float64) * np.pi / 180
    n_u, n_v = projections.shape[1:]
    du, dv = pixel[0] * binning, pixel[1] * binning
    proj_meta = CTConeBeamFlatPanelProjMeta(thetas.float(), torch.zeros(len(numbers)),
                                            detector_radius=source_to_detector - source_to_axis,
                                            beam_radius=source_to_axis, shape=(n_u, n_v), dr=(du, dv), COR=cor_mm)
    if object_meta is None:
        magnification = source_to_detector / source_to_axis
        voxel = du / magnification
        dr, shape = (voxel, voxel, dv / magnification), (n_u, n_u, n_v)
    else:
        dr, shape = object_meta.dr, object_meta.shape
    object_meta = ObjectMeta(dr, shape, centre=(0.0, 0.0, 0.0), frame="scanner", units="1/mm", modality="CT")
    return projections, object_meta, proj_meta
