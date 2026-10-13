"""``attenuation_map``: the linear attenuation coefficients on a reconstruction grid, from a CT or from a mu map."""
from __future__ import annotations

import torch

from pytomography.metadata import ObjectMeta
from ._hu import check_water, hu_to_mu, mu_unit_factor
from ._image import Image


def attenuation_map(source, object_meta: ObjectMeta, energy_keV: float | None = None, *, technique="table",
                    layout: str | None = None, units: str | None = None) -> torch.Tensor:
    r"""The linear attenuation coefficients at a photon energy, in 1/mm, on the grid of ``object_meta``.

    * **A CT** (any image :func:`~pytomography.io.load_image` reads whose units are HU: a DICOM series, a NIfTI saved
      in HU...): its HU become mu at ``energy_keV`` on the CT's own grid, along a bilinear curve (one line from air to
      water, one from water to bone) whose coefficients ``technique`` chooses, and that is put on the grid by
      :meth:`~pytomography.io.Image.on`, which averages where the grid's voxels are larger than the CT's.
    * **A mu map** (a SIMIND ``.hct`` with ``layout=``, a GATE/STIR ``.hv``, a Siemens ``.v.hdr``, a NIfTI, a vendor's
      DICOM map...): its values are converted from their units (the file's, or ``units=``) to 1/mm and checked against
      NIST's water at ``energy_keV``: a warning says when the map's water is more than 5 % off, which happens for a map
      made for another photon energy or in other units.

    The map is placed by its own geometry; ``Image.on`` warns when the map and the grid are in different frames
    (``'patient'`` and ``'scanner'``). For an image in a GATE simulation's world, first place it with
    :func:`pytomography.io.PET.gate_image`.

    Args:
        source (str | os.PathLike | list | Image): Anything :func:`~pytomography.io.load_image` reads, or an
            :class:`~pytomography.io.Image`.
        object_meta (ObjectMeta): The reconstruction grid.
        energy_keV (float, optional): The photon energy, in keV. Defaults to None: 511 keV for a PET grid (modality
            ``'PT'``); it must be given for any other (SPECT: the photopeak's energy, e.g.
            ``proj_meta.energy_windows[0].photon_keV``).
        technique (str | Callable, optional): For a CT, where the curve's coefficients come from: ``'table'``
            (PyTomography's table of measured coefficients for the CT's model and kVp, at the tabulated photon energy
            nearest ``energy_keV``; when there is none within 5 %, ``'nist'`` with a warning), ``'cortical_bone_fit'``
            (the CT's effective energy from its cortical bone peak, 1200 to 1600 HU, and NIST's air, water and
            cortical bone), ``'nist'`` (NIST, with the CT's effective energy 75 % of its kVp), or a function that takes
            HU and returns mu in 1/mm. Defaults to ``'table'``.
        layout (str, optional): For a SIMIND density map, ``'mu'`` or ``'mu-castor'`` (see
            :func:`~pytomography.io.load_image`). Defaults to None.
        units (str, optional): The units of the source's values when the file does not give them, or to override
            them: ``'HU'``, ``'1/cm'``, ``'1/mm'``. Defaults to None (the source's).

    Returns:
        torch.Tensor: Mu in 1/mm, float32, of shape ``object_meta.shape``, on ``pytomography.device``.

    Raises:
        ValueError: If ``energy_keV`` is missing for a grid that is not PET, the source's units are unknown and not
            given, or ``layout`` is given with an :class:`~pytomography.io.Image`.
    """
    if energy_keV is None:
        if object_meta.modality != "PT":
            raise ValueError("give energy_keV, the photon energy the attenuation map is for (511 keV is the default for "
                             f"a PET grid only; this grid's modality is {object_meta.modality!r})")
        energy_keV = 511.0
    if isinstance(source, Image):
        image = source
        if layout is not None:
            raise ValueError("layout= is for reading a SIMIND density map; this source is an Image already")
    else:
        from ._load import load_image
        image = load_image(source, layout=layout)
    kind = units if units is not None else image.units
    name = image.meta.get("header") or image.meta.get("path") or ""
    if kind is None:
        if image.modality == "CT":
            raise ValueError("this CT does not say its values are HU (RescaleType): pass units='HU' if they are")
        raise ValueError("the source does not say the units of its values: pass units= (e.g. '1/cm' for a GATE or "
                         "STIR mu map)")
    if str(kind).strip().upper() == "HU":
        mu = hu_to_mu(image.data, energy_keV, technique=technique, kvp=image.meta.get("KVP"),
                      model=image.meta.get("ManufacturerModelName"))
    else:
        mu = (image.data * mu_unit_factor(kind)).astype("float32")
        check_water(mu, energy_keV, str(name))
    mu_image = Image(mu, image.affine, units="1/mm", modality=image.modality, meta=image.meta, frame=image.frame)
    return mu_image.on(object_meta, fill=0.0)
