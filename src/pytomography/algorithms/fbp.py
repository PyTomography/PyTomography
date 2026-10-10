"""This module contains filtered back projection, the analytic reconstruction algorithm.
"""
from __future__ import annotations
import torch
from pytomography.projectors import SystemMatrix
from pytomography.utils.fourier_filters import get_fbp_filter
from pytomography.utils.memory import MemoryEstimate, memory_estimate

class FilteredBackProjection:
    r"""Filtered back projection: each projection is ramp filtered (the ramp :math:`|f|` times a window) and back
    projected with the weights of the geometry. The geometry-specific work is done by the system matrix, through its
    private ``_fbp`` method, onto its object grid. System matrices without one raise ``NotImplementedError``. Those
    that support it:

    * :class:`~pytomography.projectors.SPECT.SPECTSystemMatrix`: parallel-hole SPECT. Its transforms are not used, so
      the object is not corrected for attenuation, collimator blurring or scatter.
    * :class:`~pytomography.projectors.CT.CTGen3SystemMatrix`: helical and circular scans of third-generation CT
      scanners, with or without a flying focal spot, by WFBP.
    * :class:`~pytomography.projectors.CT.CTConeBeamFlatPanelSystemMatrix`: circular cone-beam scans, by FDK.

    Example:
        >>> image = FilteredBackProjection(projections, system_matrix, filter='hann')()

        Args:
            projections (torch.Tensor): Projections :math:`g` to reconstruct (line integrals for CT).
            system_matrix (SystemMatrix): System matrix of the scan; it defines the geometry and the object grid.
            filter (optional): Window applied on top of the ramp: a name (``'ram-lak'``, ``'shepp-logan'``, ``'hann'``,
                ``'hamming'``, ``'cosine'``), an :class:`~pytomography.utils.FBPFilter` such as
                :class:`~pytomography.utils.HannFilter` with a cut-off, or a function of the spatial frequency in cycles
                per mm (for a SPECT Butterworth window of cut-off 0.5 cycles per cm and order 5,
                ``lambda f: 1 / torch.sqrt(1 + (f / 0.05) ** 10)``). Defaults to ``'hann'``.
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

    def __call__(self) -> torch.Tensor:
        """Reconstructs.

        Returns:
            torch.Tensor: The reconstructed object, on the object grid of the system matrix.

        Raises:
            NotImplementedError: The system matrix does not support filtered back projection.
        """
        return self.system_matrix._fbp(self.projections, self.filter, **self.options)

    def estimate_memory(self) -> MemoryEstimate:
        """The peak memory (RAM and GPU) this reconstruction will take, without running it: what it holds (the
        projections, the image) and the most its steps hold at once, from the same sizes the system matrix uses to
        split the work. Print it to see the main arrays and what to change to use less, such as a lower
        :func:`pytomography.set_memory_budget` or ``gpu_budget``, which take longer and give the same image.

        Example:
            >>> fbp = FilteredBackProjection(projections, system_matrix, filter='hann')
            >>> print(fbp.estimate_memory())
            >>> image = fbp()

        Returns:
            MemoryEstimate: The estimate, with ``ram_gb``, ``gpu_gb`` and its parts.

        Raises:
            NotImplementedError: The system matrix does not support filtered back projection.
        """
        sm = self.system_matrix
        return memory_estimate(f'Filtered back projection ({type(sm).__name__})',
                               sm._fbp_memory_parts(self.projections, self.filter, **self.options),
                               sm._fbp_memory_alternatives(self.projections, self.filter, **self.options))
