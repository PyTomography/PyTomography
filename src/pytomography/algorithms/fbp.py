"""This module contains filtered back projection, the analytic reconstruction algorithm.
"""
from __future__ import annotations
from contextlib import nullcontext
import torch
from pytomography.projectors import SystemMatrix
from pytomography.utils.fourier_filters import get_fbp_filter
from pytomography.utils.memory import MemoryEstimate, MemoryPlan, memory_budget_set, memory_estimate, plan_settings

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

    def _estimate(self, settings: dict, alternatives: bool = True) -> MemoryEstimate:
        """The estimate with the settings of ``_fbp_memory_settings`` given (``settings``): ``memory_budget_gb`` is applied
        as the memory budget while the parts are computed, the others are passed to the system matrix's ``_fbp``."""
        sm = self.system_matrix
        options = dict(self.options)
        options.update({k: v for k, v in settings.items() if k != 'memory_budget_gb'})
        budget = memory_budget_set(settings['memory_budget_gb']) if 'memory_budget_gb' in settings else nullcontext()
        with budget:
            parts = sm._fbp_memory_parts(self.projections, self.filter, **options)
            others = sm._fbp_memory_alternatives(self.projections, self.filter, **options) if alternatives else []
        return memory_estimate(f'Filtered back projection ({type(sm).__name__})', parts, others, settings=settings)

    def estimate_memory(self) -> MemoryEstimate:
        """The peak memory (RAM and GPU) this reconstruction will take, without running it: what it holds (the
        projections, the image) and the most its steps hold at once, from the same sizes the system matrix uses to
        split the work. Print it to see the main arrays and what to change to use less, such as a lower
        :func:`pytomography.set_memory_budget` or ``gpu_budget``, which take longer and give the same image;
        ``to_dict()`` gives it as JSON. To choose those settings for the memory there is, see :meth:`plan_memory`.

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
        settings = {name: present for name, (present, _) in sm._fbp_memory_settings(self.projections, self.filter, **self.options).items()}
        return self._estimate(settings)

    def plan_memory(self, ram_gb: float | None = None, gpu_gb: float | None = None, **fixed) -> MemoryPlan:
        """The fastest settings of this reconstruction that fit in ``ram_gb`` of RAM and ``gpu_gb`` of GPU memory, as a
        :class:`~pytomography.utils.memory.MemoryPlan`. The settings are those the "To use less" line of
        :meth:`estimate_memory` lowers, none of which changes the image: for helical CT
        (:class:`~pytomography.projectors.CT.CTGen3SystemMatrix`), ``gpu_budget`` (bytes, an option of this class) and
        ``memory_budget_gb`` (:func:`pytomography.set_memory_budget`), largest first. Cone-beam and SPECT filtered back
        projection have nothing to choose: their plan says whether the reconstruction fits. Without budgets, the plan is
        for what is free on this machine now; on Windows GPU memory counts as RAM too. If nothing fits, the plan has
        ``fits`` False and the settings that come closest; it does not raise.

        Args:
            ram_gb (float | None, optional): RAM to fit in, in GB. Defaults to None: what is available now.
            gpu_gb (float | None, optional): GPU memory to fit in, in GB. Defaults to None: what is available now.
            **fixed: Settings to keep, by name (``gpu_budget``, ``memory_budget_gb``).

        Returns:
            MemoryPlan: The plan.

        Raises:
            NotImplementedError: The system matrix does not support filtered back projection.

        Example:
            >>> plan = FilteredBackProjection(projections, system_matrix).plan_memory(ram_gb=16)
            >>> pytomography.set_memory_budget(plan.settings['memory_budget_gb'])
            >>> image = FilteredBackProjection(projections, system_matrix, gpu_budget=plan.settings['gpu_budget'])()
        """
        sm = self.system_matrix
        settings = sm._fbp_memory_settings(self.projections, self.filter, **self.options)
        return plan_settings(f'Filtered back projection ({type(sm).__name__})', settings, self._estimate, fixed, ram_gb, gpu_gb)
