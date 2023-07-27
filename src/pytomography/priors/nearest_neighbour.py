"""For all priors implemented here, the neighbouring voxels considered are those directly surrounding a given voxel, so :math:`\sum_s` is a sum over 26 points."""

from __future__ import annotations
import abc
import torch
import torch.nn as nn
import numpy as np
from .prior import Prior
from collections.abc import Callable
from pytomography.utils import get_object_nearest_neighbour
from pytomography.metadata import ObjectMeta

class NearestNeighbourPrior(Prior):
    r"""Implementation of priors where gradients depend on summation over nearest neighbours :math:`s` to voxel :math:`r` given by : :math:`\frac{\partial V}{\partial f_r}=\beta\sum_{r,s}w_{r,s}\phi(f_r, f_s)` where :math:`V` is from the log-posterior probability :math:`\ln L (\tilde{f}, f) - \beta V(f)`.
    
    Args:
            beta (float): Used to scale the weight of the prior
            phi (function): Function :math:`\phi` used in formula above. Input arguments should be :math:`f_r`, :math:`f_s`, and any `kwargs` passed to this initialization function.
            weight (NeighbourWeight, optional). Weighting scheme to use for nearest neighbours. If ``None``, then uses EuclideanNeighbourWeight. Defaults to None.
    """
    def __init__(
        self,
        beta: float,
        phi: Callable,
        weight: NeighbourWeight | None = None,
        **kwargs
    ) -> None:
        super(NearestNeighbourPrior, self).__init__(beta)
        if weight is None:
            self.weight = EuclideanNeighbourWeight()
        else:
            self.weight = weight
        self.phi = phi
        self.kwargs = kwargs
        
    def set_object_meta(self, object_meta: ObjectMeta) -> None:
        """Sets object metadata parameters.

        Args:
            object_meta (ObjectMeta): Object metadata describing the system.
        """
        self.weight.set_object_meta(object_meta)
        self.object_meta = object_meta
        

    @torch.no_grad()
    def compute_gradient(self) -> torch.tensor:
        r"""Computes the gradient of the prior on ``self.object``

        Returns:
            torch.tensor: Tensor of shape [batch_size, Lx, Ly, Lz] representing :math:`\frac{\partial V}{\partial f_r}`
        """
        dx, dy, dz = self.object_meta.dr
        object_return = torch.zeros(self.object.shape).to(self.device)
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                for k in [-1,0,1]:
                    if (i==0)*(j==0)*(k==0):
                        continue
                    neighbour = get_object_nearest_neighbour(self.object, (i,j,k))
                    object_return += self.phi(self.object, neighbour, **self.kwargs) * self.weight((i,j,k))
        return self.beta*self.beta_scale_factor * object_return
    

class QuadraticPrior(NearestNeighbourPrior):
    r"""Subclass of ``NearestNeighbourPrior`` where :math:`\phi(f_r, f_s)= (f_r-f_s)/\delta` corresponds to a quadratic prior :math:`V(f)=\frac{1}{4}\sum_{r,s} w_{r,s} \left(\frac{f_r-f_s}{\delta}\right)^2`
    
    Args:
            beta (float): Used to scale the weight of the prior
            delta (float, optional): Parameter :math:`\delta` in equation above. Defaults to 1.
            weight (NeighbourWeight, optional). Weighting scheme to use for nearest neighbours. If ``None``, then uses EuclideanNeighbourWeight. Defaults to None.
    """
    def __init__(
        self,
        beta: float,
        delta: float = 1,
        weight: NeighbourWeight | None = None,
    ) -> None:
        gradient = lambda object, nearest, delta: (object-nearest) / delta
        super(QuadraticPrior, self).__init__(beta, gradient, weight=weight, delta=delta)

class LogCoshPrior(NearestNeighbourPrior):
    r"""Subclass of ``NearestNeighbourPrior`` where :math:`\phi(f_r,f_s)=\tanh((f_r-f_s)/\delta)` corresponds to the logcosh prior :math:`V(f)=\sum_{r,s} w_{r,s} \log\cosh\left(\frac{f_r-f_s}{\delta}\right)`
    
    Args:
            beta (float): Used to scale the weight of the prior
            delta (float, optional): Parameter :math:`\delta` in equation above. Defaults to 1.
            weight (NeighbourWeight, optional). Weighting scheme to use for nearest neighbours. If ``None``, then uses EuclideanNeighbourWeight. Defaults to None.
    """
    def __init__(
        self,
        beta: float,
        delta: float = 1,
        weight: NeighbourWeight | None = None,
    ) -> None:
        gradient = lambda object, nearest, delta: torch.tanh((object-nearest) / delta)
        super(LogCoshPrior, self).__init__(beta, gradient, weight=weight, delta=delta)

class RelativeDifferencePrior(NearestNeighbourPrior):
    r"""Subclass of ``NearestNeighbourPrior`` where :math:`\phi(f_r,f_s)=\frac{2(f_r-f_s)(\gamma|f_r-f_s|+3f_s + f_r)}{(\gamma|f_r-f_s|+f_r+f_s)^2}` corresponds to the relative difference prior :math:`V(f)=\sum_{r,s} w_{r,s} \frac{(f_r-f_s)^2}{f_r+f_s+\gamma|f_r-f_s|}`
    
    Args:
            beta (float): Used to scale the weight of the prior
            gamma (float, optional): Parameter :math:`\gamma` in equation above. Defaults to 1.
            epsilon (float, optional): Prevent division by 0, Defaults to 1e-8.
            weight (NeighbourWeight, optional). Weighting scheme to use for nearest neighbours. If ``None``, then uses EuclideanNeighbourWeight. Defaults to None.
    """
    def __init__(
        self, 
        beta: float = 1, 
        gamma: float = 1, 
        epsilon: float = 1e-8,
        weight: NeighbourWeight | None = None,
    ) -> None:
        gradient = lambda object, nearest, gamma, epsilon: 2*(object-nearest)*(gamma*torch.abs(object-nearest)+3*nearest+object) / (object + nearest + gamma*torch.abs(object-nearest) + epsilon)**2
        super(RelativeDifferencePrior, self).__init__(beta, gradient, gamma=gamma, weight=weight, epsilon=epsilon)
        
class NeighbourWeight():
    r"""Abstract class for assigning weight :math:`w_{r,s}` in nearest neighbour priors. 
    """
    @abc.abstractmethod
    def __init__(self):
        return
    def set_object_meta(self, object_meta: ObjectMeta) -> None:
        """Sets object meta to get appropriate spacing information

        Args:
            object_meta (ObjectMeta): Object metadata.
        """ 
        self.object_meta = object_meta
    @abc.abstractmethod
    def __call__(self, coords):
        r"""Computes the weight :math:`w_{r,s}` given the relative position :math:`s` of the nearest neighbour

        Args:
            coords (Sequence[int,int,int]): Tuple of coordinates ``(i,j,k)`` that represent the shift of neighbour :math:`s` relative to :math:`r`.
        """
        return
    
class EuclideanNeighbourWeight(NeighbourWeight):
    """Implementation of ``NeighbourWeight`` where inverse Euclidean distance is the weighting between nearest neighbours.
    """
    def __init__(self):
        super(EuclideanNeighbourWeight, self).__init__()
    
    def __call__(self, coords):
        r"""Computes the weight :math:`w_{r,s}` using inverse Euclidean distance between :math:`r` and :math:`s`.

        Args:
            coords (Sequence[int,int,int]): Tuple of coordinates ``(i,j,k)`` that represent the shift of neighbour :math:`s` relative to :math:`r`.
        """
        i, j, k = coords
        return self.object_meta.dx/np.sqrt((self.object_meta.dx*i)**2 + (self.object_meta.dy*j)**2 + (self.object_meta.dz*k)**2)
    
class AnatomyNeighbourWeight(NeighbourWeight):
    r"""Implementation of ``NeighbourWeight`` where inverse Euclidean distance and anatomical similarity is used to compute neighbour weight.

    Args:
        anatomy_image (torch.Tensor[batch_size,Lx,Ly,Lz]): Object corresponding to an anatomical image (such as CT/MRI)
        similarity_function (Callable): User-defined function that computes the similarity between :math:`r` and :math:`s` in the anatomical image. The function should be bounded between 0 and 1 where 1 represets complete similarity and 0 represents complete dissimilarity.
    """
    def __init__(
        self,
        anatomy_image: torch.Tensor,
        similarity_function: Callable
    ):
        super(AnatomyNeighbourWeight, self).__init__()
        self.eucliden_neighbour_weight = EuclideanNeighbourWeight()
        self.anatomy_image = anatomy_image
        self.similarity_function = similarity_function
        
    def set_object_meta(self, object_meta):
        """Sets object meta to get appropriate spacing information

        Args:
            object_meta (ObjectMeta): Object metadata.
        """ 
        self.object_meta = object_meta
        self.eucliden_neighbour_weight.set_object_meta(object_meta)
    def __call__(self, coords):
        r"""Computes the weight :math:`w_{r,s}` using inverse Euclidean distance and anatomical similarity between :math:`r` and :math:`s`.

        Args:
            coords (Sequence[int,int,int]): Tuple of coordinates ``(i,j,k)`` that represent the shift of neighbour :math:`s` relative to :math:`r`.
        """
        # Get Euclidean weight
        weight = self.eucliden_neighbour_weight(coords)
        # Now get weight from anatomy image
        neighbour = get_object_nearest_neighbour(self.anatomy_image, coords)
        weight *= self.similarity_function(self.anatomy_image, neighbour)
        return weight