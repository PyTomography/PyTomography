from __future__ import annotations
import torch
import abc
import pytomography
from pytomography.metadata import ObjectMeta

class Prior():
    r"""Abstract class for implementation of prior :math:`V(f)` where :math:`V` is from the log-posterior probability :math:`\ln L(\tilde{f}, f) - \beta V(f)`. Any function inheriting from this class should implement a ``foward`` method that computes the tensor :math:`\frac{\partial V}{\partial f_r}` where :math:`f` is an object tensor.
    
    Args:
            beta (float): Used to scale the weight of the prior
            obj2obj_transforms (Sequence): Sequence of transforms applied after computation of prior or gradients.

    """
    @abc.abstractmethod
    def __init__(self, beta: float, obj2obj_transforms: list[Transform] = []):
        self.beta = beta
        self.obj2obj_transforms = []
        self.device = pytomography.device

    def set_object_meta(self, object_meta: ObjectMeta) -> None:
        """Sets object metadata parameters.

        Args:
            object_meta (ObjectMeta): Object metadata describing the system.
        """
        self.object_meta = object_meta

    def set_beta_scale(self, factor: float) -> None:
        r"""Sets a scale factor for :math:`\beta` required for OSEM when finite subsets are used per iteration.

        Args:
            factor (float): Value by which to scale :math:`\beta` 
        """
        self.beta_scale_factor = factor
        
    def set_FOV_scale(self, FOV_scale: torch.Tensor) -> None:
        r"""Sets a positionally dependent scaling factor within the FOV for the prior.

        Args:
            torch.Tensor (float): Scaling factor
        """
        self.FOV_scale = FOV_scale
                

    def set_object(self, object: torch.Tensor) -> None:
        r"""Sets the object :math:`f_r` used to compute :math:`\frac{\partial V}{\partial f_r}` 

        Args:
            object (torch.tensor): Tensor of size [batch_size, Lx, Ly, Lz] representing :math:`f_r`.
        """
        self.object = object.clone()

    @abc.abstractmethod
    def __call__(self):
        """Abstract method to compute the gradient of the prior based on the ``self.object`` attribute.
        """
        ...