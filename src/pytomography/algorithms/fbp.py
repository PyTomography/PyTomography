"""This module contains filtered back projection, the analytic reconstruction algorithm.
"""
from __future__ import annotations
import torch
from pytomography.projectors import SystemMatrix
from pytomography.utils.fourier_filters import get_fbp_filter

class FilteredBackProjection:
    r"""Filtered back projection: each projection is ramp filtered (the ramp :math:`|f|` times a window) and back
    projected with the weights of the geometry. The geometry-specific work is done by the system matrix, through its
    private ``_fbp`` method, onto its object grid. System matrices without one raise ``NotImplementedError``; at present
    the CT system matrices support it: :class:`~pytomography.projectors.CT.CTGen3SystemMatrix` (helical and circular
    scans of third-generation scanners, with or without a flying focal spot, by WFBP) and
    :class:`~pytomography.projectors.CT.CTConeBeamFlatPanelSystemMatrix` (circular scans, by FDK).

    Example:
        >>> image = FilteredBackProjection(projections, system_matrix, filter='hann', slice_thickness=1.25)()

        Args:
            projections (torch.Tensor): Projections :math:`g` to reconstruct (line integrals for CT).
            system_matrix (SystemMatrix): System matrix of the scan; it defines the geometry and the object grid. Model
                no other effects in it.
            filter (optional): Window applied on top of the ramp: a name (``'ram-lak'``, ``'shepp-logan'``, ``'hann'``,
                ``'hamming'``, ``'cosine'``), an :class:`~pytomography.utils.FBPFilter` such as
                :class:`~pytomography.utils.TabulatedFilter`, or a function of the spatial frequency in cycles per mm.
                Defaults to ``'hann'``.
            **options: Passed to the system matrix's ``_fbp``. For :class:`~pytomography.projectors.CT.CTGen3SystemMatrix`:
                ``slice_thickness`` (mm), ``gpu_budget`` (bytes), ``Q`` (row weighting) and ``k_range``.
    """
    def __init__(
        self,
        projections: torch.Tensor,
        system_matrix: SystemMatrix,
        filter='hann',
        **options
        ) -> None:
        self.projections = projections
        self.system_matrix = system_matrix
        self.filter = get_fbp_filter(filter)
        self.options = options

    def __call__(self, projections: torch.Tensor | None = None) -> torch.Tensor:
        """Reconstructs.

        Args:
            projections (torch.Tensor, optional): Projections to reconstruct instead of those given at construction.

        Returns:
            torch.Tensor: The reconstructed object, on the object grid of the system matrix.

        Raises:
            NotImplementedError: The system matrix does not support filtered back projection.
        """
        projections = self.projections if projections is None else projections
        return self.system_matrix._fbp(projections, self.filter, **self.options)
