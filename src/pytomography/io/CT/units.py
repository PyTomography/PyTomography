"""Hounsfield units and linear attenuation coefficients of a CT reconstruction."""
from __future__ import annotations

import numpy as np
import torch


def _water(proj_meta_or_water) -> float:
    """Water's linear attenuation coefficient (1/mm): a number, or ``water_attenuation`` of a CT's projection metadata
    (DICOM-CT-PD's WaterAttenuationCoefficient, (7041,1001))."""
    if isinstance(proj_meta_or_water, (int, float, np.floating)):
        water = float(proj_meta_or_water)
    else:
        water = getattr(proj_meta_or_water, "water_attenuation", None)
        if water is None:
            raise ValueError("this projection metadata carries no water_attenuation: pass water's attenuation (1/mm) "
                             "at the scan's effective energy instead")
    if not water > 0:
        raise ValueError(f"water's attenuation must be positive, not {water}")
    return float(water)


def to_hu(mu, proj_meta_or_water):
    """Hounsfield units of a reconstruction in 1/mm: 1000 (mu / mu_water - 1).

    Args:
        mu (torch.Tensor | np.ndarray): linear attenuation coefficients (1/mm).
        proj_meta_or_water: the scan's projection metadata (its ``water_attenuation``, 1/mm), or water's attenuation.

    Returns:
        the image in HU, of the type of ``mu``.
    """
    water = _water(proj_meta_or_water)
    return 1000 * (mu / water - 1)


def from_hu(hu, proj_meta_or_water):
    """Linear attenuation coefficients (1/mm) of an image in Hounsfield units: mu_water (1 + HU / 1000).

    Args:
        hu (torch.Tensor | np.ndarray): the image in HU.
        proj_meta_or_water: the scan's projection metadata (its ``water_attenuation``, 1/mm), or water's attenuation.

    Returns:
        the image in 1/mm, of the type of ``hu``.
    """
    water = _water(proj_meta_or_water)
    return water * (1 + hu / 1000)
