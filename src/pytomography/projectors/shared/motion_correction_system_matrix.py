from __future__ import annotations
from collections.abc import Sequence
from pytomography.projectors import SystemMatrix
from pytomography.transforms import Transform
from ..system_matrix import ExtendedSystemMatrix

class MotionSystemMatrix(ExtendedSystemMatrix):
    def __init__(
        self,
        system_matrices: Sequence[SystemMatrix],
        motion_transforms: Sequence[Transform]
        ) -> None:
        r"""System matrix that supports motion correction. Maps to an extended image space :math:`\mathcal{V}^{*}` where projections have shape ``[N_gates,...]`` where ``...`` is the regular projeciton size. As such, this transform only supports objects with a batch size of 1. The forward transform is given by :math:`H_n M_n` and back projection is given by :math:`\sum_n M_n^{T} H_n^{T}` where :math:`n` corresponds to the nth motion transform, :math:`H_n` corresponds to the nth system matrix, and :math:`M_n` corresponds to the nth motion transform. Typical motion transforms map to a reference gate, e.g. :math:`M_n = M_{r \to n}` and :math:`M_n^{T} = M_{n \to r}` where :math:`r` is the reference gate. 

        Args:
            system_matrices (Sequence[SystemMatrix]): List of system matrices corresponding to each gate :math:`n`. Different system matrices may be required in SPECT imaging for example, if different attenuation maps are used for each phase. 
            motion_transforms (Sequence[Transform]): Motion transform corresponding to phase :math:`n`.
        """
        super(MotionSystemMatrix, self).__init__(
            system_matrices=system_matrices,
            obj2obj_transforms = motion_transforms
        )
