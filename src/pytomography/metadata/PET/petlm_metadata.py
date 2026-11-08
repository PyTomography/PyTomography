"""Projection metadata of PET list mode data."""
from __future__ import annotations
import torch
from .pet_tof_metadata import PETTOFMeta

class PETLMProjMeta():
    r"""Projection space of PET list mode data: the events, the scanner, and the weights of the lines of response.

    List mode reconstruction uses two system matrices: :math:`H`, which projects onto the lines of response (LORs) of
    the detected events and is used for forward and back projection, and :math:`\tilde{H}`, which projects onto every
    valid LOR and gives the sensitivity image :math:`\tilde{H}^T w`. The readers
    (:func:`~pytomography.io.PET.read_gate`, :func:`~pytomography.io.PET.read_petsird`,
    :func:`~pytomography.io.PET.read_ge_hdf5`) make one.

    Args:
        detector_ids (torch.Tensor): [N, 2] (no TOF) or [N, 3] (TOF) integers: the two detector ids of each event, and
            its TOF bin. These are the LORs of :math:`H`. Kept on the CPU.
        info (dict, optional): The scanner's description (:func:`~pytomography.io.PET.scanner_info`). Needed when
            ``scanner_LUT`` is not given, for the sensitivity weights of individual events and for sorting the events
            in the projector. Defaults to None.
        scanner_LUT (torch.Tensor, optional): Position (x, y, z) of every crystal in mm, [N_crystals, 3]. Defaults to
            None: made from ``info`` with :func:`~pytomography.io.PET.scanner_lut`.
        tof_meta (PETTOFMeta | None, optional): Time of flight binning. Defaults to None (no time of flight).
        weights (torch.Tensor | None, optional): One weight per event, [N], that scales the forward projection and
            the back projection when the system matrix is made with ``scale_projection_by_sensitivity=True``
            (attenuation and normalization factors of each event's LOR). Without an additive term (scatter, randoms)
            they cancel in the list mode likelihood, and ``weights_sensitivity`` is enough. Defaults to None.
        detector_ids_sensitivity (torch.Tensor | None, optional): [M, 2] detector ids of the LORs of the sensitivity
            image (the LORs of :math:`\tilde{H}`). Defaults to None: every pair of crystals.
        weights_sensitivity (torch.Tensor | None, optional): The sensitivity weights :math:`w` (normalization, and
            attenuation when not given to the system matrix as a map): one per row of ``detector_ids_sensitivity``,
            or, without it, one per crystal pair in ``torch.combinations`` order (``N_crystals (N_crystals - 1) / 2``;
            see :func:`~pytomography.io.PET.crystal_pair_index`). Defaults to None (all 1).

    Attributes:
        shape (tuple): (N,), the number of events.
        info (dict | None): The scanner's description.
        detector_ids (torch.Tensor): The events' detector ids (and TOF bins).
        scanner_lut (torch.Tensor): Position of every crystal in mm, [N_crystals, 3].
        tof_meta (PETTOFMeta | None): Time of flight binning.
        weights (torch.Tensor | None): The events' weights.
        detector_ids_sensitivity (torch.Tensor | None): The LORs of the sensitivity image.
        weights_sensitivity (torch.Tensor | None): The sensitivity weights.
        duration (float | None): Seconds of acquisition the events were read from, where the reader knows it.
    """
    def __init__(
        self,
        detector_ids: torch.Tensor,
        info: dict | None = None,
        scanner_LUT: torch.Tensor | None = None,
        tof_meta: PETTOFMeta | None = None,
        weights: torch.tensor | None = None,
        detector_ids_sensitivity: torch.Tensor | None = None,
        weights_sensitivity: torch.tensor | None = None,
    ):
        self.shape = (detector_ids.shape[0],)
        self.info = info
        self.detector_ids = detector_ids.cpu()
        if detector_ids_sensitivity is not None:
            self.detector_ids_sensitivity = detector_ids_sensitivity.cpu()
        else:
            self.detector_ids_sensitivity = None
        if scanner_LUT is None:
            from pytomography.io.PET.scanner import scanner_lut
            self.scanner_lut = scanner_lut(info).cpu()
        else:
            self.scanner_lut = scanner_LUT
        self.tof_meta = tof_meta
        self.weights = weights
        self.weights_sensitivity = weights_sensitivity
        # seconds of acquisition the events were read from, where the reader knows it (None otherwise)
        self.duration = None