from __future__ import annotations
import abc
import torch

class CallBack():
    """Abstract class used for callbacks. Subclasses must redefine the ``__init__`` and ``run`` methods. If a callback is used as an argument in an iterative reconstruction algorihtm, the ``__run__`` method is called after each subiteration.
    """
    @abc.abstractmethod
    def __init__(self):
        """Abstract method for ``__init__``.
        """
        ...
    @abc.abstractmethod
    def run(self, obj: torch.tensor, n_iter: int, n_subset: int):
        """Abstract method for ``run``.

        Args:
            obj (torch.tensor[batch_size, Lx, Ly, Lz]): An object which one can compute various statistics from.
            n_iter (int): The iteration number
            n_subset (int): The subiteration number
        """
        ...
        