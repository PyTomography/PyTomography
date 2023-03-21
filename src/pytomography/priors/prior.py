import torch
import torch.nn as nn
import numpy as np
import abc
from pytomography.metadata import ObjectMeta, ImageMeta

class Prior(nn.Module):
    r"""Abstract class for implementation of prior :math:`V(f)` where :math:`V` is from the log-posterior probability :math:`\ln L(\tilde{f}, f) - \beta V(f)`. Any function inheriting from this class should implement a ``foward`` method that computes the tensor :math:`\frac{\partial V}{\partial f_r}` where :math:`f` is an object tensor.
    
    Args:
            beta (float): Used to scale the weight of the prior
            device (float): Pytorch device used for computation. Defaults to 'cpu'.
    """
    @abc.abstractmethod
    def __init__(self, beta: float, device : str ='cpu'):
        super(Prior, self).__init__()
        self.beta = beta
        self.beta_scale_factor = 1
        self.device = device

    def set_object_meta(self, object_meta: ObjectMeta) -> None:
        """Sets object metadata parameters.

        Args:
            object_meta (ObjectMeta): Object metadata describing the system.
        """
        self.object_meta = object_meta

    def set_beta_scale(self, factor: float) -> None:
        r"""Sets :math:`\beta` 

        Args:
            factor (float): Value of :math:`\beta` 
        """
        self.beta_scale_factor = factor

    def set_object(self, object: ObjectMeta) -> None:
        r"""Sets the object :math:`f_r` used to compute :math:`\frac{\partial V}{\partial f_r}` 

        Args:
            object (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] representing :math:`f_r`.
        """
        self.object = object

    def set_device(self, device: str = 'cpu') -> None:
        """Sets the pytorch computation device

        Args:
            device (str): sets device.
        """
        self.device=device

    @abc.abstractmethod
    def forward(self):
        """Abstract method to compute prior based on the ``self.object`` attribute.
        """
        ...