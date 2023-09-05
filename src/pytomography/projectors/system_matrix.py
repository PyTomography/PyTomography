from __future__ import annotations
import abc
import torch
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

