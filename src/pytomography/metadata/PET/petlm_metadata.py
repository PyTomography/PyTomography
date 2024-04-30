from __future__ import annotations
import torch
from .pet_tof_metadata import PETTOFMeta
from pytomography.io.PET import shared

class PETLMProjMeta():
    r"""Metadata required for PET listmode modeling. PET listmode projection actually requires two different projectors: the system matrix that projects to all detected crystal pair LORs (which is denoted as :math:`H`) and the system matrix that projects to all valid LORs (denoted as :math:`\tilde{H}`). The system matrix :math:`H` is used for forward/back projection in reconstruction algorithms, while :math:`\tilde{H}` is used for computing the normalization image :math:`\tilde{H}^T 1`. 
        Args:
            detector_ids (torch.Tensor): :math:`N \times 2` (non-TOF) or :math:`N \times 3` (TOF) tensor that provides detector ID pairs (and TOF bin) for coincidence events. This information is used to construct :math:`H`.
            info (dict, optional): Dictionary containing all relevant information about the scanner. If ``scanner_LUT`` is not provided, then info is used to create the ``scanner_LUT``. At least one of ``info`` or ``scanner_LUT`` should be provided as input arguments.
            scanner_LUT (torch.Tensor, optional): scanner lookup table that provides spatial coordinates for all detector ID pairs. If ``info`` is not provided, then ``scanner_LUT`` must be provided.
            tof_meta (PETTOFMeta | None, optional): PET time-of-flight metadata used to modify :math:`H` for time of flight projection. If None, then time of flight is not used. Defaults to None.
            weights (torch.tensor | None, optional): weights used to scale projections after forward projection and before back projection; these modify the system matrix :math:`H`. While such weights can be used to apply attenuation/normalization correction, they aren't required in the absence of randoms/scatter; these correction need only be performed using ``weights_sensitivity``. If provided, these weights must have the number of elements as the first dimension of ``detector_ids``. If none, then no scaling is done. Defaults to None.
            detector_ids_sensitivity (torch.tensor | None, optional): valid detector ids used to generate the sensitivity image :math:`\tilde{H}^T 1`. As such, these are used to construct :math:`\tilde{H}`. If None, then assumes all detector ids (specified by ``scanner_LUT``) are valid. Defaults to None.
            weights_sensitivity (torch.tensor | None, optional): weights used for scaling projections in the computation of the sensitivity image, if the weights are given as :math:`w` then the sensitivty image becomes :math:`\tilde{H}^T w`; these modify the system matrix :math:`\tilde{H}`. These weights are used for attenuation/normalization correction. If ``detector_ids_sensitivity`` is provided, then ``weights_sensitivity`` should have the same shape. If ``detector_ids_sensitivity`` is not provided, then ``weights_sensitivity`` should be the same length as all possible combinations of detectors in the ``scanner_LUT``. If None, then no scaling is performed. Defaults to None.
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
            self.scanner_lut = shared.get_scanner_LUT(info).cpu()
        else:
            self.scanner_lut = scanner_LUT
        self.tof_meta = tof_meta
        self.weights = weights
        self.weights_sensitivity = weights_sensitivity 