from __future__ import annotations
from collections.abc import Sequence
<<<<<<< HEAD
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
=======
import torch
import pytomography
from pytomography.projectors import SystemMatrix
from pytomography.transforms import Transform

class MotionSystemMatrix(SystemMatrix):
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
        self.object_meta = system_matrices[0].object_meta
        self.proj_meta = system_matrices[0].proj_meta
        self.system_matrices = system_matrices
        self.motion_transforms = motion_transforms
        for motion_transform, system_matrix in zip(motion_transforms, system_matrices):
            motion_transform.configure(system_matrix.object_meta, system_matrix.proj_meta)
        
    def forward(self, object, angle_subset=None):
        """Forward transform :math:`H_n M_n f`, This adds an additional dimension to the object, namely :math:`n`, corresponding to the :math:`n`th motion transform. The result of the forward projection is thus projections that contains all motion transforms in the batch dimension.

        Args:
            object (torch.Tensor[1,Lx,Ly,Lz]): Object to be forward projected. Must have a batch size of 1.
            angle_subset (Sequence[int], optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.

        Returns:
           torch.Tensor[N_gates,...]: Forward projection.
        """
        return torch.vstack([H.forward(m.forward(object), angle_subset) for m, H in zip(self.motion_transforms, self.system_matrices)])
    
    def backward(self, proj, angle_subset=None):
        """Back projection :math:`\sum_n M_n^{T} H_n^{T}`. This reduces the batch dimension :math:`n` obtained via forward projection to yield an object with a batch dimension of 1. As such, the batch projection of ``proj`` must be equal to the length of ``self.motion_transform``.

        Args:
            proj (torch.Tensor[N_gates,...]): Projection data to be back-projected.
            angle_subset (Sequence[int], optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.. Defaults to None.

        Returns:
            torch.Tensor[1,Lx,Ly,Lz]: Back projection.
        """
        objects = []
        for proj_i, system_matrix, motion_transform in zip(proj, self.system_matrices, self.motion_transforms):
            objects.append(motion_transform.backward(system_matrix.backward(proj_i.unsqueeze(0),angle_subset)))
        return torch.vstack(objects).mean(axis=0).unsqueeze(0)
    
    def get_subset_splits(
        self,
        n_subsets: int
    ) -> list:
        """Returns a list of subsets (where each subset contains indicies corresponding to different angles). For example, if the projections consisted of 6 total angles, then ``get_subsets_splits(2)`` would return ``[[0,2,4],[1,3,5]]``.
        
        Args:
            n_subsets (int): number of subsets used in OSEM 

        Returns:
            list: list of index arrays for each subset
        """
        return self.system_matrices[0].get_subset_splits(n_subsets)
    
    def compute_normalization_factor(self, angle_subset: list[int] = None):
        """Function called by reconstruction algorithms to get the normalization factor :math:`\sum_n M_n^{T} H_n^{T} 1`.

        Returns:
           torch.Tensor: Normalization factor :math:`\sum_n M_n^{T} H_n^{T} 1`.
        """
        norm_proj = torch.ones((len(self.motion_transforms), *self.proj_meta.shape)).to(pytomography.device)
        return self.backward(norm_proj, angle_subset)
>>>>>>> 0740d58afb33a2c3287f7f9af8c122017b7c6d18
