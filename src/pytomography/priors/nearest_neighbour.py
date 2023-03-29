"""For all priors implemented here, the neighbouring voxels considered are those directly surrounding a given voxel, so :math:`\sum_s` is a sum over 26 points."""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from .prior import Prior
from collections.abc import Callable
from pytomography.utils import get_object_nearest_neighbour

class NearestNeighbourPrior(Prior):
    r"""Implementation of priors where gradients depend on summation over nearest neighbours :math:`s` to voxel :math:`r` given by : :math:`\frac{\partial V}{\partial f_r}=\beta\sum_{r,s}w_{r,s}\phi(f_r, f_s)` where :math:`V` is from the log-posterior probability :math:`\ln L (\tilde{f}, f) - \beta V(f)`.
    
    Args:
            beta (float): Used to scale the weight of the prior
            phi (function): Function :math:`\phi` used in formula above. Input arguments should be :math:`f_r`, :math:`f_s`, and any `kwargs` passed to this initialization function.
            device (str, optional): Pytorch device used for computation. Defaults to 'cpu'.
    """
    def __init__(
        self,
        beta: float,
        phi: Callable, 
        **kwargs
    ) -> None:
        super(NearestNeighbourPrior, self).__init__(beta)
        self.phi = phi
        self.kwargs = kwargs

    @torch.no_grad()
    def __call__(self) -> torch.tensor:
        r"""Computes the prior on ``self.object``

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
                    weight = dx/np.sqrt((dx*i)**2 + (dy*j)**2 + (dz*k)**2)
                    object_return += self.phi(self.object, neighbour, **self.kwargs) * weight
        return self.beta*self.beta_scale_factor * object_return
    

class QuadraticPrior(NearestNeighbourPrior):
    r"""Subclass of ``NearestNeighbourPrior`` where :math:`\phi(f_r, f_s)= (f_r-f_s)/\delta` corresponds to a quadratic prior :math:`V(f)=\frac{1}{4}\sum_{r,s} w_{r,s} \left(\frac{f_r-f_s}{\delta}\right)^2`
    
    Args:
            beta (float): Used to scale the weight of the prior
            delta (float, optional): Parameter :math:`\delta` in equation above. Defaults to 1.
    """
    def __init__(
        self,
        beta: float,
        delta: float = 1,
    ) -> None:
        gradient = lambda object, nearest, delta: (object-nearest) / delta
        super(QuadraticPrior, self).__init__(beta, gradient, delta=delta)

class LogCoshPrior(NearestNeighbourPrior):
    r"""Subclass of ``NearestNeighbourPrior`` where :math:`\phi(f_r,f_s)=\tanh((f_r-f_s)/\delta)` corresponds to the logcosh prior :math:`V(f)=\sum_{r,s} w_{r,s} \log\cosh\left(\frac{f_r-f_s}{\delta}\right)`
    
    Args:
            beta (float): Used to scale the weight of the prior
            delta (float, optional): Parameter :math:`\delta` in equation above. Defaults to 1.
    """
    def __init__(
        self,
        beta: float,
        delta: float = 1,
    ) -> None:
        gradient = lambda object, nearest, delta: torch.tanh((object-nearest) / delta)
        super(LogCoshPrior, self).__init__(beta, gradient, delta=delta)

class RelativeDifferencePrior(NearestNeighbourPrior):
    r"""Subclass of ``NearestNeighbourPrior`` where :math:`\phi(f_r,f_s)=\frac{2(f_r-f_s)(\gamma|f_r-f_s|+3f_s + f_r)}{(\gamma|f_r-f_s|+f_r+f_s)^2}` corresponds to the relative difference prior :math:`V(f)=\sum_{r,s} w_{r,s} \frac{(f_r-f_s)^2}{f_r+f_s+\gamma|f_r-f_s|}`
    
    Args:
            beta (float): Used to scale the weight of the prior
            gamma (float, optional): Parameter :math:`\gamma` in equation above. Defaults to 1.
            epsilon (float, optional): Prevent division by 0, Defaults to 1e-8.
    """
    def __init__(
        self, 
        beta: float = 1, 
        gamma: float = 1, 
        epsilon: float = 1e-8
    ) -> None:
        gradient = lambda object, nearest, gamma, epsilon: 2*(object-nearest)*(gamma*torch.abs(object-nearest)+3*nearest+object) / (object + nearest + gamma*torch.abs(object-nearest) + epsilon)**2
        super(RelativeDifferencePrior, self).__init__(beta, gradient, gamma=gamma, epsilon=epsilon)