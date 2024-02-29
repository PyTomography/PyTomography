r"""The code here is implementation of priors that depend on summation over nearest neighbours :math:`s` to voxel :math:`r` given by :math:`V(f) = \beta \sum_{r,s}w_{r,s}\phi_0(f_r, f_s)`. These priors have first order gradients given by :math:`\nabla_r V(f) = \sum_s w_{r,s} \phi_1(f_r, f_s)` where :math:`\phi_1(f_r, f_s) = \nabla_r (\phi_0(f_r, f_s) + \phi_0(f_s, f_r))`. In addition, they have higher order gradients given by :math:`\nabla_{r'r} V(f) = \theta(r-r')\left(\sum_s w_{r,s} \phi_2^{(1)}(f_r, f_s)\right) + w_{r,r'}\phi_2^{(2)}(f_r, f_{r'})` where :math:`\phi_2^{(1)}(f_r, f_s) = \nabla_r \phi_1(f_r, f_s)` and :math:`\phi_2^{(2)}(f_r, f_s) = \nabla_s \phi_1(f_r, f_s)`. The particular :math:`\phi` functions must be implemented by subclasses depending on the functionality required. The second order derivative is only required to be implemented if one wishes to use the prior function in error estimation"""

from __future__ import annotations
import abc
import torch
import torch.nn as nn
import numpy as np
from .prior import Prior
from collections.abc import Callable
import pytomography
from pytomography.utils import get_object_nearest_neighbour
from pytomography.metadata import ObjectMeta

class NearestNeighbourPrior(Prior):
    r"""Generic class for the nearest neighbour prior.
    
    Args:
            beta (float): Used to scale the weight of the prior
            weight (NeighbourWeight, optional). Weighting scheme to use for nearest neighbours: this specifies :math:`w_{r,s}` above. If ``None``, then uses EuclideanNeighbourWeight, which weights neighbouring voxels based on their euclidean distance. Defaults to None.
    """
    def __init__(
        self,
        beta: float,
        weight: NeighbourWeight | None = None,
        **kwargs
    ) -> None:
        super(NearestNeighbourPrior, self).__init__(beta)
        if weight is None:
            self.weight = EuclideanNeighbourWeight()
        else:
            self.weight = weight
        self.__dict__.update(kwargs)
        
    def set_object_meta(self, object_meta: ObjectMeta) -> None:
        """Sets object metadata parameters.

        Args:
            object_meta (ObjectMeta): Object metadata describing the system.
        """
        self.weight.set_object_meta(object_meta)
        self.object_meta = object_meta
        
    @torch.no_grad()
    def _pair_contribution(self, phi: Callable, beta_scale=False, second_order_derivative_object: torch.Tensor | None = None, swap_object_and_neighbour: bool = False) -> torch.tensor:
        r"""Helper function used to compute prior and associated gradients

        Returns:
            torch.tensor: Tensor of shape [batch_size, Lx, Ly, Lz].
        """
        object_return = torch.zeros(self.object.shape).to(self.device)
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                for k in [-1,0,1]:
                    if (i==0)*(j==0)*(k==0):
                        continue
                    neighbour = get_object_nearest_neighbour(self.object, (i,j,k))
                    # Only done when computing higher order derivatives for error computation
                    if second_order_derivative_object is not None:
                        if swap_object_and_neighbour:
                            second_order_derivative_object_neighbour = get_object_nearest_neighbour(second_order_derivative_object, (i,j,k))
                            object_return += phi(neighbour, self.object) * second_order_derivative_object_neighbour * self.weight((i,j,k))
                        else:
                            object_return += phi(self.object, neighbour) * second_order_derivative_object * self.weight((i,j,k))
                    # Done for regular computation of priors
                    else:
                        object_return += phi(self.object, neighbour) * self.weight((i,j,k))
        for transform in self.obj2obj_transforms:
            object_return = transform.forward(object_return)
        if beta_scale:
            scale_factor = self.beta_scale_factor
        else:
            scale_factor = 1
        return self.beta * scale_factor * object_return
    
    def phi0(self, fr, fs):
        raise NotImplementedError(f"Prior evaluation not implemented")
    
    def phi1(self, fr, fs):
        raise NotImplementedError(f"Prior derivative of order 1 not implemented: must be implemented for incorporating priors in reconstruction")
    
    def phi2_1(self, fr, fs):
        raise NotImplementedError(f"Prior derivative of order 2 not fully implemented: must implement both phi2_1 and phi2_2 methods")
    
    def phi2_2(self, fr, fs):
        raise NotImplementedError(f"Prior derivative of order 2 not fully implemented: must implement both phi2_1 and phi2_2 methods")
    
    def __call__(self, derivative_order: int = 0) -> float | torch.Tensor | Callable:
        """Used to compute the prior with gradient of specified order. If order 0, then returns a float (the value of the prior). If order 1, then returns a torch.Tensor representative of the prior gradient at each voxel. If order 2, then returns a callable function (representative of a higher order tensor but without storing each component).

        Args:
            derivative_order (int, optional): The order of the derivative to compute. This will specify the ouput; only possible values are 0, 1, or 2. Defaults to 0.

        Raises:
            NotImplementedError: for cases where the derivative order is not between 0 and 2.

        Returns:
            float | torch.Tensor | Callable: The prior with derivative of specified order. 
        """
        if derivative_order==0:
            return self._pair_contribution(self.phi0).sum().item()
        elif derivative_order==1:
            return self._pair_contribution(self.phi1, beta_scale=True)
        elif derivative_order==2:
            diagonal_component = self._pair_contribution(self.phi2_1, second_order_derivative_object=torch.ones(self.object.shape).to(pytomography.device), beta_scale=True)
            blur_component = lambda input: self._pair_contribution(self.phi2_2, second_order_derivative_object = input, swap_object_and_neighbour=True, beta_scale=True)
            return lambda input: diagonal_component * input + blur_component(input)
        else:
            raise NotImplementedError(f"Prior not implemented for derivative order >2")

class QuadraticPrior(NearestNeighbourPrior):
    r"""Subclass of ``NearestNeighbourPrior`` corresponding to a quadratic prior: namely :math:`\phi_0(f_r, f_s) = 1/4 \left[(fr-fs)/\delta\right]^2` and where the gradient is determined by :math:`\phi_1(f_r, f_s) = (f_r-f_s)/\delta`
    
    Args:
            beta (float): Used to scale the weight of the prior
            weight (NeighbourWeight, optional). Weighting scheme to use for nearest neighbours. If ``None``, then uses EuclideanNeighbourWeight. Defaults to None.
            delta (float, optional): Parameter :math:`\delta` in equation above. Defaults to 1.
    """
    def __init__(
        self,
        beta: float,
        weight: NeighbourWeight | None = None,
        delta: float = 1,
    ) -> None:
        super(QuadraticPrior, self).__init__(beta, weight=weight, delta=delta)
        
    def phi0(self, fr, fs):
        return 1/4 * ((fr - fs)/self.delta)**2
    
    def phi1(self, fr, fs):
        return (fr - fs) / self.delta

class LogCoshPrior(NearestNeighbourPrior):
    r"""Subclass of ``NearestNeighbourPrior`` corresponding to a logcosh prior: namely :math:`\phi_0(f_r, f_s) = \tanh((f_r-f_s)/\delta)` and where the gradient is determined by :math:`\phi_1(f_r, f_s) = \log \cosh \left[(f_r-f_s)/\delta\right]`
    
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
        Vr = lambda object, nearest, delta: torch.log(torch.cosh((object-nearest) / delta))
        super(LogCoshPrior, self).__init__(beta, gradient, Vr=Vr, weight=weight, delta=delta)
        
    def phi0(self, fr, fs):
        return torch.tanh((fr - fs) / self.delta)
    
    def phi1(self, fr, fs):
        return torch.tanh((fr - fs) / self.delta)

class RelativeDifferencePrior(NearestNeighbourPrior):
    r"""Subclass of ``NearestNeighbourPrior`` corresponding to the relative difference prior: namely :math:`\phi_0(f_r, f_s) = \frac{(f_r-f_s)^2}{f_r+f_s+\gamma|f_r-f_s|}` and where the gradient is determined by :math:`\phi_1(f_r, f_s) = \frac{2(f_r-f_s)(\gamma|f_r-f_s|+3f_s + f_r)}{(\gamma|f_r-f_s|+f_r+f_s)^2}`
    
    Args:
            beta (float): Used to scale the weight of the prior
            gamma (float, optional): Parameter :math:`\gamma` in equation above. Defaults to 1.
            weight (NeighbourWeight, optional). Weighting scheme to use for nearest neighbours. If ``None``, then uses EuclideanNeighbourWeight. Defaults to None.
    """
    def __init__(
        self,
        beta: float,
        weight: NeighbourWeight | None = None,
        gamma: float = 1,
    ) -> None:
        super(RelativeDifferencePrior, self).__init__(beta, weight=weight, gamma = gamma)
        
    def phi0(self, fr, fs):
        return (fr-fs)**2 / (fr + fs + self.gamma*torch.abs(fr-fs) + pytomography.delta)
    
    def phi1(self, fr, fs):
        return (2*(fr-fs)*(self.gamma*torch.abs(fr-fs)+3*fs+fr) + pytomography.delta) / ((fr + fs + self.gamma*torch.abs(fr-fs))**2 + pytomography.delta)
    
    def phi2_1(self, fr, fs):
        return 16*fs**2 / ((self.gamma*torch.abs(fr-fs) + fr + fs)**3 + pytomography.delta)
    
    def phi2_2(self, fr, fs):
        return -16*fr*fs / ((self.gamma*torch.abs(fr-fs) + fr + fs)**3 + pytomography.delta)
        
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
    
class TopNAnatomyNeighbourWeight(NeighbourWeight):
    r"""Implementation of ``NeighbourWeight`` where inverse Euclidean distance and anatomical similarity is used. In this case, only the top N most similar neighbours are used as weight

    Args:
        anatomy_image (torch.Tensor[batch_size,Lx,Ly,Lz]): Object corresponding to an anatomical image (such as CT/MRI)
        N_neighbours (int): Number of most similar neighbours to use
    """
    def __init__(
        self,
        anatomy_image: torch.Tensor,
        N_neighbours: int,
    ):
        super(TopNAnatomyNeighbourWeight, self).__init__()
        self.eucliden_neighbour_weight = EuclideanNeighbourWeight()
        self.anatomy_image = anatomy_image
        self.N = N_neighbours
        self.compute_inclusion_tensor()
        
    def set_object_meta(self, object_meta):
        """Sets object meta to get appropriate spacing information

        Args:
            object_meta (ObjectMeta): Object metadata.
        """ 
        self.object_meta = object_meta
        self.eucliden_neighbour_weight.set_object_meta(object_meta)
        
    def compute_inclusion_tensor(self):
        shape = self.anatomy_image.shape[1:]
        self.inclusion_image = torch.zeros((3, 3, 3, *shape))
        anatomy_cpu = self.anatomy_image.cpu()
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                for k in [-1,0,1]:
                    if (i==0)*(j==0)*(k==0):
                        self.inclusion_image[i+1,j+1,k+1] = torch.inf
                        continue
                    self.inclusion_image[i+1,j+1,k+1] = torch.abs(anatomy_cpu - get_object_nearest_neighbour(anatomy_cpu, (i,j,k)))
        self.inclusion_image = self.inclusion_image.reshape((27,*shape))
        self.inclusion_image = (torch.argsort(torch.argsort(self.inclusion_image, dim=0), dim=0)<self.N)
        self.inclusion_image = self.inclusion_image.reshape((3,3,3,*shape))
    
    def __call__(self, coords):
        r"""Computes the weight :math:`w_{r,s}` using inverse Euclidean distance and anatomical similarity between :math:`r` and :math:`s`.

        Args:
            coords (Sequence[int,int,int]): Tuple of coordinates ``(i,j,k)`` that represent the shift of neighbour :math:`s` relative to :math:`r`.
        """
        # Get Euclidean weight
        weight = self.eucliden_neighbour_weight(coords)
        # Now get weight from anatomy image
        weight *= self.inclusion_image[coords[0]+1,coords[1]+1,coords[2]+1].to(pytomography.device).to(pytomography.dtype)
        return weight