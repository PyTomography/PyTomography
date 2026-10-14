"""The system matrix of motion-corrected reconstruction from gated data (:class:`MotionSystemMatrix`)."""
from __future__ import annotations
from collections.abc import Sequence
from pytomography.projectors import SystemMatrix
from pytomography.transforms import Transform
from ..system_matrix import ExtendedSystemMatrix

class MotionSystemMatrix(ExtendedSystemMatrix):
    r"""System matrix of gated data with motion correction: projections of every gate, [N_gates, ...].

    The forward projection of gate :math:`n` is :math:`H_n M_n f`, and the back projection
    :math:`\sum_n M_n^T H_n^T g_n`, with :math:`H_n` the system matrix and :math:`M_n` the motion transform of gate
    :math:`n` (e.g. :class:`~pytomography.transforms.shared.DVFMotionTransform`). Motion transforms usually map a
    reference gate :math:`r` to gate :math:`n`: :math:`M_n = M_{r \to n}`, and :math:`M_n^T = M_{n \to r}`, so the
    object reconstructed is the reference gate's. It is an :class:`~pytomography.projectors.ExtendedSystemMatrix`
    with the motion transforms as its object-to-object transforms.

    Args:
        system_matrices (Sequence[SystemMatrix]): The system matrix of each gate :math:`n` (they may differ, e.g. a
            SPECT attenuation map per gate).
        motion_transforms (Sequence[Transform]): The motion transform :math:`M_n` of each gate.
    """
    def __init__(
        self,
        system_matrices: Sequence[SystemMatrix],
        motion_transforms: Sequence[Transform]
        ) -> None:
        super(MotionSystemMatrix, self).__init__(
            system_matrices=system_matrices,
            obj2obj_transforms = motion_transforms
        )
