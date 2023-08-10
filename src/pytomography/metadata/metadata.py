from __future__ import annotations
import abc
from typing import Sequence
import pytomography
from pytomography.utils import compute_pad_size
import torch
import inspect

class ObjectMeta():
    """Abstract parent class for all different types of Object Space Metadata. In general, while this is fairly similar for all imaging modalities, required padding features/etc may be different for different modalities.
    """
    @abc.abstractmethod
    def __init__(self) -> None:
        """Abstract method for ``__init``; this will depend on imaging modality"""
        ...
    
    def __repr__(self):
        attributes = [f"{attr} = {getattr(self, attr)}\n" for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        return ''.join(attributes)


class ImageMeta():
    """Abstract parent class for all different types of Image Space Metadata. Implementation and required parameters will differ significantly between different imaging modalities.
    """
    @abc.abstractmethod
    def __init__(self) -> None:
        """Abstract method for ``__init``; this will depend on imaging modality"""
        ...
    
    def __repr__(self):
        attributes = [f"{attr} = {getattr(self, attr)}\n" for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        return ''.join(attributes)

        