from __future__ import annotations
import abc
import torch

class Callback():
    """Abstract class used for callbacks. Subclasses must redefine the ``__init__`` and ``run`` methods. If a callback is used as an argument in an iterative reconstruction algorihtm, the ``__run__`` method is called after each subiteration.
    """
    @abc.abstractmethod
    def __init__(self):
        """Abstract method for ``__init__``.
        """
        ...
    @abc.abstractmethod
    def run(self, object: torch.Tensor, n_iter: int):
        """Abstract method for ``run``.

        Args:
            object (torch.Tensor[Lx, Ly, Lz]): Object at current iteration/subset in the reconstruction algorithm
            n_iter (int): The iteration number
            
        Returns:
            torch.Tensor: Modified object from callback. This must be returned by all callbacks (if the callback doesn't change the object, then the passed object is returned)
        """
        return object
    def finalize(self, object: torch.Tensor):
        """Abstract method for ``run``.

        Args:
            object (torch.Tensor[Lx, Ly, Lz]): Reconstructed object (all iterations/subsets completed)
        """
        return None
        