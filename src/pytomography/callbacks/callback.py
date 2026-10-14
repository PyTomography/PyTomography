"""The base class of the callbacks, and :class:`MultiCallback`, which runs several."""
from __future__ import annotations
import abc
import torch

class Callback():
    """The base class of the callbacks, which a reconstruction algorithm runs after each subiteration.

    Subclasses define ``__init__`` and :meth:`run` (and, if they need it, :meth:`finalize`). The algorithm calls
    ``run(object, n_iter, n_subset)`` after each subiteration and uses the object it returns, and ``finalize(object)``
    once at the end.
    """
    @abc.abstractmethod
    def __init__(self):
        ...
    @abc.abstractmethod
    def run(self, object: torch.Tensor, n_iter: int):
        """Runs after a subiteration; subclasses implement it.

        The algorithms call it as ``run(object, n_iter, n_subset)``, with the subset index as a third argument.

        Args:
            object (torch.Tensor): The object after the subiteration, [Lx, Ly, Lz].
            n_iter (int): The iteration number.

        Returns:
            torch.Tensor: The object the reconstruction continues from: the one given, unless the callback changes it.
        """
        return object
    def finalize(self, object: torch.Tensor):
        """Runs once after the last subiteration; this base class does nothing.

        Args:
            object (torch.Tensor): The reconstructed object, [Lx, Ly, Lz].
        """
        return None
    
class MultiCallback(Callback):
    """Several callbacks run as one, in order, to give a reconstruction algorithm more than one.

    Args:
        callbacks (list[Callback]): The callbacks, run one after another (each gets the object the previous one
            returned).
    """
    def __init__(
        self,
        callbacks: list[Callback]
    ):
        self.callbacks = callbacks  
    def run(self, object: torch.Tensor, n_iter: int, n_subset: int) -> torch.Tensor:
        """Runs every callback in order.

        Args:
            object (torch.Tensor): The object after the subiteration, [Lx, Ly, Lz].
            n_iter (int): The iteration number.
            n_subset (int): The subset index.

        Returns:
            torch.Tensor: The object the last callback returned.
        """
        for callback in self.callbacks:
            object = callback.run(object, n_iter, n_subset)
        return object
    def finalize(self, object: torch.Tensor):
        """Finalizes every callback in order.

        Args:
            object (torch.Tensor): The reconstructed object, [Lx, Ly, Lz].
        """
        for callback in self.callbacks:
            callback.finalize(object)
        