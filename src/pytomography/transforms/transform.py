"""The base class of the transforms that system matrices are built from."""
from __future__ import annotations
import abc
import torch
import torch.nn as nn
import pytomography
from pytomography.metadata import ObjectMeta, ProjMeta

class Transform(metaclass=abc.ABCMeta):
    """The base class of every transform: an operation within object space or within projection space.

    A system matrix applies its object-to-object transforms (``obj2obj_transforms``) to the object before projecting
    and its projection-to-projection transforms (``proj2proj_transforms``) to the projections after, and their
    :meth:`backward` (the transpose) in back projection. The system matrix calls :meth:`configure` with its metadata
    when it is made. Subclasses implement :meth:`forward` and :meth:`backward`; the SPECT system matrix passes the
    view's index as a second argument to object-to-object transforms.

    Attributes:
        device (torch.device | str): ``pytomography.device`` when the transform was made.
        object_meta (ObjectMeta): The system matrix's object grid, once configured.
        proj_meta (ProjMeta): The system matrix's projection space, once configured.
    """
    def __init__(self) -> None:
        self.device = pytomography.device

    def configure(self, object_meta: ObjectMeta, proj_meta: ProjMeta) -> None:
        """Gives the transform the metadata of the system matrix it belongs to; the system matrix calls it.

        Subclasses that need the metadata to set themselves up (kernels, padding) do so here.

        Args:
            object_meta (ObjectMeta): The object grid.
            proj_meta (ProjMeta): The projection space.
        """
        self.object_meta = object_meta
        self.proj_meta = proj_meta

    @abc.abstractmethod
    def forward(self, x: torch.tensor):
        """Applies the transform to an object or to projections; implemented by each subclass.

        Args:
            x (torch.Tensor): The object or projections.

        Returns:
            torch.Tensor: The transformed object or projections.
        """
        ...
    @abc.abstractmethod
    def backward(self, x: torch.tensor):
        """Applies the transpose of the transform, for back projection; implemented by each subclass.

        Args:
            x (torch.Tensor): The object or projections.

        Returns:
            torch.Tensor: The result of the transpose.
        """
        ...
