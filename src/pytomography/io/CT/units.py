"""Conversions of a CT image between linear attenuation coefficients (1/mm) and Hounsfield units."""
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
    r"""Convert an image from linear attenuation coefficients (1/mm) to Hounsfield units.

    :math:`\mathrm{HU} = 1000\,(\mu / \mu_\mathrm{water} - 1)`.

    Args:
        mu (torch.Tensor | np.ndarray): Linear attenuation coefficients, in 1/mm.
        proj_meta_or_water (CTGen3ProjMeta | float): The scan's projection metadata, whose ``water_attenuation`` (set
            by :func:`~pytomography.io.CT.read_dicom_ct_pd`) is used, or water's attenuation coefficient at the scan's
            effective energy, in 1/mm.

    Returns:
        torch.Tensor | np.ndarray: The image in HU, of the type of ``mu``.

    Raises:
        ValueError: The projection metadata has no ``water_attenuation``, or water's attenuation is not positive.

    Example:
        >>> projections, object_meta, proj_meta = io.CT.read_dicom_ct_pd(folder)
        >>> hu = io.CT.to_hu(recon, proj_meta)
    """
    water = _water(proj_meta_or_water)
    return 1000 * (mu / water - 1)


def from_hu(hu, proj_meta_or_water):
    r"""Convert an image from Hounsfield units to linear attenuation coefficients (1/mm).

    :math:`\mu = \mu_\mathrm{water}\,(1 + \mathrm{HU} / 1000)`.

    Args:
        hu (torch.Tensor | np.ndarray): The image in HU.
        proj_meta_or_water (CTGen3ProjMeta | float): The scan's projection metadata, whose ``water_attenuation`` (set
            by :func:`~pytomography.io.CT.read_dicom_ct_pd`) is used, or water's attenuation coefficient at the scan's
            effective energy, in 1/mm.

    Returns:
        torch.Tensor | np.ndarray: The image in 1/mm, of the type of ``hu``.

    Raises:
        ValueError: The projection metadata has no ``water_attenuation``, or water's attenuation is not positive.
    """
    water = _water(proj_meta_or_water)
    return water * (1 + hu / 1000)
