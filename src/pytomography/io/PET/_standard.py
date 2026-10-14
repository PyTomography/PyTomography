"""What the PET readers share: their default grid, the object_meta they return, and the normalisation weights' checks."""
from __future__ import annotations

import math
import os
import warnings
from collections.abc import Sequence

import numpy as np
import torch

from pytomography.metadata import ObjectMeta


def default_grid(info: dict) -> tuple[tuple, tuple]:
    """The grid a PET reader uses when it is not given one: voxels half the crystal pitch (around the ring and along
    the axis), a square covering 80 % of the ring's diameter, and every ring. Returns ``(dr, shape)``."""
    dx = info["crystalTransSpacing"] / 2
    dz = info["crystalAxialSpacing"] / 2 if info["crystalAxialSpacing"] > 0 else dx
    n_xy = max(1, math.ceil(0.8 * 2 * info["radius"] / dx))
    n_z = max(1, 2 * info["NrRings"])
    if info["NrRings"] > 1 and info["crystalAxialSpacing"] > 0:
        # every ring: the rings' extent (with gaps between modules) at half the crystal pitch
        from .scanner import scanner_lut
        z = scanner_lut(info)[:, 2]
        n_z = max(1, math.ceil((float(z.max() - z.min()) + info["crystalAxialSpacing"]) / dz))
    return (dx, dx, dz), (n_xy, n_xy, n_z)


def reader_object_meta(object_meta: ObjectMeta | None, default: tuple, *, centre, frame: str, modality: str,
                       units: str) -> ObjectMeta:
    """The ``object_meta`` a reader returns: the grid (``dr`` and ``shape``) of ``object_meta`` when one is given, else
    ``default`` (``(dr, shape)``), with its centre, frame, modality and units set by the reader."""
    dr, shape = default if object_meta is None else (object_meta.dr, object_meta.shape)
    return ObjectMeta(dr, shape, centre=tuple(float(c) for c in np.asarray(centre, dtype=np.float64)), frame=frame,
                      units=units, modality=modality)


def normalization_weights(source, info: dict, *, in_place: bool = False) -> torch.Tensor:
    """Normalisation weights of every crystal pair, in ``torch.combinations`` order, from a ``.npy`` file, an array or
    a tensor: their number is checked against the scanner's pairs, and NaN weights (pairs without calibration data)
    become 0, with a warning that says how many (in ``source`` itself if ``in_place``)."""
    from_file = isinstance(source, (str, os.PathLike))
    own = in_place or from_file
    if from_file:   # 1.6 GB for the mMR: read once, and its NaNs replaced in place
        weights = torch.from_numpy(np.load(source, mmap_mode="r").astype(np.float32))
    else:
        weights = torch.as_tensor(source).to(torch.float32)
    n = int(info["NrCrystalsPerRing"] * info["NrRings"])
    if weights.ndim != 1 or weights.shape[0] != n * (n - 1) // 2:
        raise ValueError(f"normalisation weights must have one weight per crystal pair of this scanner "
                         f"({n * (n - 1) // 2:,} for {n:,} crystals), not shape {tuple(weights.shape)}")
    nan = torch.isnan(weights)
    count = int(nan.sum())
    if count:
        if own:
            weights[nan] = 0
        else:   # the caller's weights are not changed
            weights = torch.where(nan, torch.zeros((), dtype=weights.dtype), weights)
        warnings.warn(f"{count:,} of {weights.shape[0]:,} normalisation weights ({100 * count / weights.shape[0]:.1f} %) "
                      "are NaN (pairs without calibration data, such as LORs outside a calibration cylinder); they are "
                      "set to 0, so those pairs add nothing to the sensitivity", stacklevel=3)
    return weights
