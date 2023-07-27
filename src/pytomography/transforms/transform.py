from __future__ import annotations
import abc
import torch
import torch.nn as nn
import pytomography
from pytomography.metadata import ObjectMeta, ImageMeta

class Transform(metaclass=abc.ABCMeta):
    """The parent class for all transforms used in reconstruction (obj2obj, im2im, obj2im). Subclasses must implement the ``__call__`` method.

    Args:
        device (str): Pytorch device used for computation
    """
    def __init__(self) -> None:
        """Used to initialize the correction network.
        """
        self.device = pytomography.device

    def configure(self, object_meta: ObjectMeta, image_meta: ImageMeta) -> None:
        """Configures the transform to the object/image metadata. This is done after creating the network so that it can be adjusted to the system matrix.

        Args:
            object_meta (ObjectMeta): Object metadata.
            image_meta (ImageMeta): Image metadata.
        """
        self.object_meta = object_meta
        self.image_meta = image_meta

    @abc.abstractmethod
    def forward(self, x: torch.tensor):
        """Abstract method; must be implemented in subclasses to apply a correction to an object/image and return it
        """
        ...
    @abc.abstractmethod
    def backward(self, x: torch.tensor):
        """Abstract method; must be implemented in subclasses to apply a correction to an object/image and return it
        """
        ...
