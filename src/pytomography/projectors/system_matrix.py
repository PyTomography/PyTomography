from __future__ import annotations
import abc
import torch
import pytomography
from pytomography.transforms import Transform
from pytomography.metadata import ObjectMeta, ProjMeta

class SystemMatrix():
    r"""Abstract class for a general system matrix :math:`H:\mathbb{U} \to \mathbb{V}` which takes in an object :math:`f \in \mathbb{U}` and maps it to corresponding projections :math:`g \in \mathbb{V}` that would be produced by the imaging system. A system matrix consists of sequences of object-to-object and proj-to-proj transforms that model various characteristics of the imaging system, such as attenuation and blurring. While the class implements the operator :math:`H:\mathbb{U} \to \mathbb{V}` through the ``forward`` method, it also implements :math:`H^T:\mathbb{V} \to \mathbb{U}` through the `backward` method, required during iterative reconstruction algorithms such as OSEM.
    
    Args:
            obj2obj_transforms (Sequence[Transform]): Sequence of object mappings that occur before forward projection.
            im2im_transforms (Sequence[Transform]): Sequence of proj mappings that occur after forward projection.
            object_meta (ObjectMeta): Object metadata.
            proj_meta (ProjMeta): Projection metadata.
    """
    def __init__(
        self,
        obj2obj_transforms: list[Transform],
        proj2proj_transforms: list[Transform],
        object_meta: ObjectMeta,
        proj_meta: ProjMeta,
    ) -> None:
        self.obj2obj_transforms = obj2obj_transforms
        self.proj2proj_transforms = proj2proj_transforms
        self.object_meta = object_meta
        self.proj_meta = proj_meta
        self.initialize_transforms()

    def initialize_transforms(self):
        """Initializes all transforms used to build the system matrix
        """
        for transform in self.obj2obj_transforms:
            transform.configure(self.object_meta, self.proj_meta)
        for transform in self.proj2proj_transforms:
            transform.configure(self.object_meta, self.proj_meta)
            
    @abc.abstractmethod
    def forward(self, object: torch.tensor, **kwargs):
        r"""Implements forward projection :math:`Hf` on an object :math:`f`.

        Args:
            object (torch.tensor[batch_size, Lx, Ly, Lz]): The object to be forward projected
            angle_subset (list, optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.

        Returns:
            torch.tensor[batch_size, Ltheta, Lx, Lz]: Forward projected proj where Ltheta is specified by `self.proj_meta` and `angle_subset`.
        """
        ...
    @abc.abstractmethod
    def backward(
        self,
        proj: torch.tensor,
        angle_subset: list | None = None,
        return_norm_constant: bool = False,
    ) -> torch.tensor:
        r"""Implements back projection :math:`H^T g` on a set of projections :math:`g`.

        Args:
            proj (torch.Tensor): proj which is to be back projected
            angle_subset (list, optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.
            return_norm_constant (bool): Whether or not to return :math:`1/\sum_j H_{ij}` along with back projection. Defaults to 'False'.

        Returns:
            torch.tensor[batch_size, Lr, Lr, Lz]: the object obtained from back projection.
        """
        ...
    @abc.abstractmethod
    def get_subset_splits(
        self,
        n_subsets: int
    ) -> list:
        """Returns a list of subsets corresponding to a partition of the projection data used in a reconstruction algorithm.
        
        Args:
            n_subsets (int): number of subsets used in OSEM 

        Returns:
            list: list of index arrays for each subset
        """
        ...

class ExtendedSystemMatrix(SystemMatrix):
    def __init__(
        self,
        system_matrices: Sequence[SystemMatrix],
        obj2obj_transforms: Sequence[Transform] = None,
        proj2proj_transforms: Sequence[Transform] = None,
        ) -> None:
        r"""System matrix that supports the extension of projection space. Maps to an extended image space :math:`\mathcal{V}^{*}` where projections have shape ``[N,...]`` where ``...`` is the regular projeciton size. As such, this projector only supports objects with a batch size of 1. The forward transform is given by :math:`H' = \sum_n v_n \otimes B_n H_n A_n` where :math:`\left\{A_n\right\}` are object-to-object space transforms, :math:`\left\{H_n\right\}` are a sequence of system matrices, :math:`\left\{B_n\right\}` are a sequence of projection-to-projection space transforms, :math:`v_n` is a basis vector of length :math:`N` with a value of 1 in component :math:`n` and :math:`\otimes` is a tensor product. 

        Args:
            system_matrices (Sequence[SystemMatrix]): List of system matrices corresponding to each dimension :math:`n`. 
            obj2obj_transforms (Sequence[Transform]): List of object to object transforms corresponding to each dimension :math:`n`. 
            proj2proj_transforms (Sequence[Transform]): List of projection to projection transforms corresponding to each dimension :math:`n`. 
        """
        self.object_meta = system_matrices[0].object_meta
        self.proj_meta = system_matrices[0].proj_meta
        self.system_matrices = system_matrices
        self.obj2obj_transforms = obj2obj_transforms
        self.proj2proj_transforms = proj2proj_transforms
        for i in range(len(self.system_matrices)):
            if self.obj2obj_transforms is not None:
                if self.obj2obj_transforms[i] is not None:
                    self.obj2obj_transforms[i].configure(self.system_matrices[i].object_meta, self.system_matrices[i].proj_meta)
            if self.proj2proj_transforms is not None:
                if self.proj2proj_transforms[i] is not None:
                    self.proj2proj_transforms[i].configure(self.system_matrices[i].object_meta, self.system_matrices[i].proj_meta)
        
    def forward(self, object, angle_subset=None):
        r"""Forward transform :math:`H' = \sum_n v_n \otimes B_n H_n A_n`, This adds an additional dimension to the projection space.

        Args:
            object (torch.Tensor[1,Lx,Ly,Lz]): Object to be forward projected. Must have a batch size of 1.
            angle_subset (Sequence[int], optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.

        Returns:
           torch.Tensor[N_gates,...]: Forward projection.
        """
        projs = []
        for i in range(len(self.system_matrices)):
            if self.obj2obj_transforms is not None:
                if self.obj2obj_transforms[i] is not None:
                    object_i = self.obj2obj_transforms[i].forward(object)
            proj_i = self.system_matrices[i].forward(object_i, angle_subset)
            if self.proj2proj_transforms is not None:
                if self.proj2proj_transforms[i] is not None:
                    proj_i = self.proj2proj_transforms[i].forward(proj_i)
            projs.append(proj_i.clone())
        return torch.vstack(projs)
    
    def backward(self, proj, angle_subset=None):
        r"""Back projection :math:`H' = \sum_n v_n^T \otimes A_n^T H_n^T B_n^T`. This maps an extended projection back to the original object space.

        Args:
            proj (torch.Tensor[N,...]): Projection data to be back-projected.
            angle_subset (Sequence[int], optional): Only uses a subset of angles (i.e. only certain values of :math:`j` in formula above) when back projecting. Useful for ordered-subset reconstructions. Defaults to None, which assumes all angles are used.. Defaults to None.

        Returns:
            torch.Tensor[1,Lx,Ly,Lz]: Back projection.
        """
        objects = []
        for i in range(len(self.system_matrices)):
            proj_i = proj[i].unsqueeze(0)
            if self.proj2proj_transforms is not None:
                if self.proj2proj_transforms[i] is not None:
                    proj_i = self.proj2proj_transforms[i].backward(proj_i)
            object_i = self.system_matrices[i].backward(proj_i, angle_subset)
            if self.obj2obj_transforms is not None:
                if self.obj2obj_transforms[i] is not None:
                    object_i = self.obj2obj_transforms[i].backward(object_i)
            objects.append(object_i.clone())
        return torch.vstack(objects).sum(axis=0).unsqueeze(0)
    
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
        r"""Function called by reconstruction algorithms to get the normalization factor :math:`H' = \sum_n v_n^T \otimes A_n^T H_n^T B_n^T` 1.

        Returns:
           torch.Tensor[1,Lx,Ly,Lz]: Normalization factor.
        """
        norm_proj = torch.ones((len(self.system_matrices), *self.proj_meta.shape)).to(pytomography.device)
        return self.backward(norm_proj, angle_subset)
