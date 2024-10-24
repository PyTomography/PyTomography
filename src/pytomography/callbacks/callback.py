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
    
class MultiCallback(Callback):
    """Class for combining multiple callbacks into a single callback. This is useful for passing multiple callbacks to an iterative reconstruction algorithm.
    """
    def __init__(
        self,
        callbacks: list[Callback]
    ):
        self.callbacks = callbacks  
    def run(self, object: torch.Tensor, n_iter: int, n_subset: int) -> torch.Tensor:
        """Runs the callbacks sequentially

        Args:
            object (torch.Tensor): Object at current iteration/subset in the reconstruction algorithm
            n_iter (int): Iteration number
            n_subset (int): Subset number

        Returns:
            torch.Tensor: Modified object from callback. This must be returned by all callbacks (if the callback doesn't change the object, then the passed object is returned)
        """
        for callback in self.callbacks:
            object = callback.run(object, n_iter, n_subset)
        return object
    def finalize(self, object: torch.Tensor):
        """Finalizes the callback

        Args:
            object (torch.Tensor): Reconstructed object
        """
        for callback in self.callbacks:
            callback.finalize(object)
        