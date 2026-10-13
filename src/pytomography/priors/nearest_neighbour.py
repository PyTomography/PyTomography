r"""Priors that sum over the 26 nearest neighbours of each voxel, and the weights of the neighbours.

The prior is :math:`V(f) = \beta \sum_{r,s}w_{r,s}\phi_0(f_r, f_s)`, over the voxels :math:`r` and their neighbours
:math:`s`. Its gradient is :math:`\nabla_r V(f) = \sum_s w_{r,s} \phi_1(f_r, f_s)`, where
:math:`\phi_1(f_r, f_s) = \nabla_r (\phi_0(f_r, f_s) + \phi_0(f_s, f_r))`, and its second derivative
:math:`\nabla_{r'r} V(f) = \theta(r-r')\left(\sum_s w_{r,s} \phi_2^{(1)}(f_r, f_s)\right) + w_{r,r'}\phi_2^{(2)}(f_r,
f_{r'})`, where :math:`\phi_2^{(1)}(f_r, f_s) = \nabla_r \phi_1(f_r, f_s)` and
:math:`\phi_2^{(2)}(f_r, f_s) = \nabla_s \phi_1(f_r, f_s)`. Subclasses implement the :math:`\phi` functions; the
second order ones are needed only for uncertainty estimates.
"""

from __future__ import annotations
import abc
import torch
import numpy as np
from .prior import Prior
from collections.abc import Callable
import pytomography
from pytomography.utils import get_object_nearest_neighbour
from pytomography.metadata import ObjectMeta

class NearestNeighbourPrior(Prior):
    r"""The base class of the nearest neighbour priors; subclasses define the functions :math:`\phi`.

    Args:
        beta (float): The weight :math:`\beta` of the prior.
        weight (NeighbourWeight, optional): The weights :math:`w_{r,s}` of the neighbours. Defaults to None:
            :class:`EuclideanNeighbourWeight` (inverse distance).
        **kwargs: The parameters of the subclass's :math:`\phi` (e.g. ``delta``, ``gamma``), kept as attributes.
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
        """Gives the prior and its weights the object grid, and resets the field of view scale to ones.

        Args:
            object_meta (ObjectMeta): The object grid.
        """
        self.weight.set_object_meta(object_meta)
        self.object_meta = object_meta
        # Set to ones, but possibly updated by recon algorithm in reconstruction
        self.FOV_scale = torch.ones(self.object_meta.shape).to(pytomography.device)
        
    @torch.no_grad()
    def _pair_contribution(self, phi: Callable, beta_scale=False, second_order_derivative_object: torch.Tensor | None = None) -> torch.tensor:
        r"""Helper function used to compute prior and associated gradients

        Returns:
            torch.tensor: Tensor of shape [batch_size, Lx, Ly, Lz].
        """
        object_return = torch.zeros(self.object.shape).to(self.device)
        valid_points = torch.ones(self.object.shape).to(pytomography.device)
        total_weight = torch.zeros(self.object.shape).to(pytomography.device)
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                for k in [-1,0,1]:
                    if (i==0)*(j==0)*(k==0):
                        continue
                    valid_points_ijk = get_object_nearest_neighbour(valid_points, (i,j,k))
                    total_weight += valid_points*valid_points_ijk * self.weight((i,j,k))
                    neighbour = get_object_nearest_neighbour(self.object, (i,j,k))
                    # Only done when computing higher order derivatives for error computation
                    if second_order_derivative_object is not None:
                        second_order_derivative_object_neighbour = get_object_nearest_neighbour(second_order_derivative_object, (i,j,k))
                        object_return += phi(self.object, neighbour) * second_order_derivative_object_neighbour * self.weight((i,j,k)) * valid_points_ijk * valid_points
                    # Done for regular computation of priors
                    else:
                        object_return += phi(self.object, neighbour) * self.weight((i,j,k)) * valid_points_ijk * valid_points
        for transform in self.obj2obj_transforms:
            object_return = transform.forward(object_return)
        if beta_scale:
            scale_factor = self.beta_scale_factor
        else:
            scale_factor = 1
        return self.beta * scale_factor * object_return * self.FOV_scale
    
    def phi0(self, fr, fs):
        r"""The function :math:`\phi_0(f_r, f_s)` of the prior's value; subclasses implement it.

        Args:
            fr (torch.Tensor): The object :math:`f_r`.
            fs (torch.Tensor): The object shifted to a neighbour, :math:`f_s`.

        Raises:
            NotImplementedError: In this base class.
        """
        raise NotImplementedError(f"Prior evaluation not implemented")

    def phi1(self, fr, fs):
        r"""The function :math:`\phi_1(f_r, f_s)` of the gradient; subclasses implement it.

        Args:
            fr (torch.Tensor): The object :math:`f_r`.
            fs (torch.Tensor): The object shifted to a neighbour, :math:`f_s`.

        Raises:
            NotImplementedError: In this base class.
        """
        raise NotImplementedError(f"Prior derivative of order 1 not implemented: must be implemented for incorporating priors in reconstruction")

    def phi2_1(self, fr, fs):
        r"""The function :math:`\phi_2^{(1)}(f_r, f_s)` of the second derivative; subclasses implement it.

        Args:
            fr (torch.Tensor): The object :math:`f_r`.
            fs (torch.Tensor): The object shifted to a neighbour, :math:`f_s`.

        Raises:
            NotImplementedError: In this base class.
        """
        raise NotImplementedError(f"Prior derivative of order 2 not fully implemented: must implement both phi2_1 and phi2_2 methods")

    def phi2_2(self, fr, fs):
        r"""The function :math:`\phi_2^{(2)}(f_r, f_s)` of the second derivative; subclasses implement it.

        Args:
            fr (torch.Tensor): The object :math:`f_r`.
            fs (torch.Tensor): The object shifted to a neighbour, :math:`f_s`.

        Raises:
            NotImplementedError: In this base class.
        """
        raise NotImplementedError(f"Prior derivative of order 2 not fully implemented: must implement both phi2_1 and phi2_2 methods")

    def __call__(self, derivative_order: int = 0) -> float | torch.Tensor | Callable:
        """The prior, its gradient or its second derivative at the object set with :meth:`~Prior.set_object`.

        The derivatives are scaled by the factor set with :meth:`~Prior.set_beta_scale` (the subset's share of the
        data); every order is multiplied by :math:`\\beta` and the field of view scale.

        Args:
            derivative_order (int, optional): 0 for the prior's value, 1 for its gradient, 2 for its second derivative.
                Defaults to 0.

        Returns:
            float | torch.Tensor | Callable: The value (a number); the gradient, [Lx, Ly, Lz]; or the second
            derivative, as a function that applies it to an object.

        Raises:
            NotImplementedError: If ``derivative_order`` is more than 2.
        """
        if derivative_order==0:
            return self._pair_contribution(self.phi0).sum().item()
        elif derivative_order==1:
            return self._pair_contribution(self.phi1, beta_scale=True)
        elif derivative_order==2:
            diagonal_component = self._pair_contribution(self.phi2_1, beta_scale=True)
            blur_component = lambda input: self._pair_contribution(self.phi2_2, second_order_derivative_object = input, beta_scale=True)
            return lambda input: diagonal_component * input + blur_component(input)
        else:
            raise NotImplementedError(f"Prior not implemented for derivative order >2")

class QuadraticPrior(NearestNeighbourPrior):
    r"""The quadratic prior: :math:`\phi_0(f_r, f_s) = \frac{1}{4} \left[(f_r-f_s)/\delta\right]^2`.

    Its gradient uses :math:`\phi_1(f_r, f_s) = (f_r-f_s)/\delta`.

    Args:
        beta (float): The weight :math:`\beta` of the prior.
        weight (NeighbourWeight, optional): The weights of the neighbours. Defaults to None:
            :class:`EuclideanNeighbourWeight`.
        delta (float, optional): The scale :math:`\delta`, in the units of the image. Defaults to 1.
    """
    def __init__(
        self,
        beta: float,
        weight: NeighbourWeight | None = None,
        delta: float = 1,
    ) -> None:
        super(QuadraticPrior, self).__init__(beta, weight=weight, delta=delta)
        
    def phi0(self, fr, fs):
        r"""The quadratic prior's :math:`\phi_0(f_r, f_s) = \frac{1}{4} \left[(f_r-f_s)/\delta\right]^2`."""
        return 1/4 * ((fr - fs)/self.delta)**2

    def phi1(self, fr, fs):
        r"""The quadratic prior's :math:`\phi_1(f_r, f_s) = (f_r-f_s)/\delta`."""
        return (fr - fs) / self.delta

class LogCoshPrior(NearestNeighbourPrior):
    r"""The log-cosh prior: :math:`\phi_0(f_r, f_s) = \log \cosh \left[(f_r-f_s)/\delta\right]`.

    Its gradient uses :math:`\phi_1(f_r, f_s) = \tanh\left[(f_r-f_s)/\delta\right]`: quadratic for small differences,
    linear for large ones, so edges are penalised less than by :class:`QuadraticPrior`.

    Args:
        beta (float): The weight :math:`\beta` of the prior.
        delta (float, optional): The scale :math:`\delta`, in the units of the image. Defaults to 1.
        weight (NeighbourWeight, optional): The weights of the neighbours. Defaults to None:
            :class:`EuclideanNeighbourWeight`.
    """
    def __init__(
        self,
        beta: float,
        delta: float = 1,
        weight: NeighbourWeight | None = None,
    ) -> None:
        super(LogCoshPrior, self).__init__(beta, weight=weight, delta=delta)
        
    def phi0(self, fr, fs):
        r"""The log-cosh prior's :math:`\phi_0(f_r, f_s) = \log \cosh \left[(f_r-f_s)/\delta\right]`."""
        return torch.log(torch.cosh((fr-fs)/self.delta))

    def phi1(self, fr, fs):
        r"""The log-cosh prior's :math:`\phi_1(f_r, f_s) = \tanh\left[(f_r-f_s)/\delta\right]`."""
        return torch.tanh((fr - fs) / self.delta)

class RelativeDifferencePrior(NearestNeighbourPrior):
    r"""The relative difference prior: :math:`\phi_0(f_r, f_s) = \frac{(f_r-f_s)^2}{f_r+f_s+\gamma|f_r-f_s|+\delta}`.

    Its gradient uses :math:`\phi_1(f_r, f_s) = \frac{(f_r-f_s)(\gamma|f_r-f_s|+3f_s + f_r + 2\delta)}{(\gamma|f_r-f_s|
    +f_r+f_s+\delta)^2}`; the second order functions are implemented too, for uncertainty estimates.

    Args:
        beta (float): The weight :math:`\beta` of the prior.
        weight (NeighbourWeight, optional): The weights of the neighbours. Defaults to None:
            :class:`EuclideanNeighbourWeight`.
        gamma (float, optional): The edge-preservation parameter :math:`\gamma`. Defaults to 1.
        delta (float, optional): A small number :math:`\delta` that keeps the denominator from vanishing. Defaults to
            ``pytomography.delta``.
    """
    def __init__(
        self,
        beta: float,
        weight: NeighbourWeight | None = None,
        gamma: float = 1,
        delta = pytomography.delta
    ) -> None:
        super(RelativeDifferencePrior, self).__init__(beta, weight=weight, gamma = gamma, delta=delta)
        
    def phi0(self, fr, fs):
        r"""The relative difference prior's :math:`\phi_0(f_r, f_s)` (see the class)."""
        return (fr-fs)**2 / (fr + fs + self.gamma*torch.abs(fr-fs) + self.delta)

    def phi1(self, fr, fs):
        r"""The relative difference prior's :math:`\phi_1(f_r, f_s)` (see the class)."""
        return ((fr-fs)*(self.gamma*torch.abs(fr-fs)+3*fs+fr + 2*self.delta)) / ((fr + fs + self.gamma*torch.abs(fr-fs)+ self.delta)**2)

    def phi2_1(self, fr, fs):
        r"""The relative difference prior's :math:`\phi_2^{(1)} = 2(2f_s+\delta)^2 / (\gamma|f_r-f_s|+f_r+f_s+\delta)^3`."""
        return (2*(2*fs+self.delta)**2) / ((self.gamma*torch.abs(fr-fs) + fr + fs + self.delta)**3)

    def phi2_2(self, fr, fs):
        r"""The relative difference prior's :math:`\phi_2^{(2)} = -2(2f_r+\delta)(2f_s+\delta) / (\gamma|f_r-f_s|+f_r+f_s+\delta)^3`."""
        return -(2*(2*fr+self.delta)*(2*fs+self.delta)) / ((self.gamma*torch.abs(fr-fs) + fr + fs + self.delta)**3)
        
class NeighbourWeight():
    r"""The base class of the weights :math:`w_{r,s}` of the neighbours in nearest neighbour priors.

    Calling a weight with a neighbour's shift ``(i, j, k)`` gives the weight of that neighbour for every voxel (a
    number, or a tensor [Lx, Ly, Lz]).
    """
    @abc.abstractmethod
    def __init__(self):
        return
    def set_object_meta(self, object_meta: ObjectMeta) -> None:
        """Gives the weight the object grid (its voxel sizes set the distances); the prior calls it.

        Args:
            object_meta (ObjectMeta): The object grid.
        """
        self.object_meta = object_meta
    @abc.abstractmethod
    def __call__(self, coords):
        r"""The weight :math:`w_{r,s}` of the neighbour :math:`s` at a shift from :math:`r`; subclasses implement it.

        Args:
            coords (Sequence[int]): The shift ``(i, j, k)`` of the neighbour, each -1, 0 or 1.
        """
        return
    
class EuclideanNeighbourWeight(NeighbourWeight):
    """Neighbour weights by inverse distance: :math:`w_{r,s} = dx / |r - s|`, 1 for the neighbours along x."""
    def __init__(self):
        super(EuclideanNeighbourWeight, self).__init__()

    def __call__(self, coords):
        r"""The weight :math:`dx / |r - s|` of the neighbour at a shift, with the distance in mm.

        Args:
            coords (Sequence[int]): The shift ``(i, j, k)`` of the neighbour, each -1, 0 or 1.

        Returns:
            float: The weight.
        """
        i, j, k = coords
        return self.object_meta.dx/np.sqrt((self.object_meta.dx*i)**2 + (self.object_meta.dy*j)**2 + (self.object_meta.dz*k)**2)
    
class AnatomyNeighbourWeight(NeighbourWeight):
    r"""Neighbour weights by inverse distance times the similarity of the two voxels in an anatomical image.

    Args:
        anatomy_image (torch.Tensor): The anatomical image (e.g. CT or MRI) on the object grid, [Lx, Ly, Lz].
        similarity_function (Callable): A function of the anatomical image and the image shifted to the neighbour
            that gives their similarity voxel by voxel, between 0 (completely different) and 1 (the same).
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
        """Gives the weight the object grid (its voxel sizes set the distances); the prior calls it.

        Args:
            object_meta (ObjectMeta): The object grid.
        """
        self.object_meta = object_meta
        self.eucliden_neighbour_weight.set_object_meta(object_meta)
    def __call__(self, coords):
        r"""The weight of the neighbour at a shift: inverse distance times the anatomical similarity.

        Args:
            coords (Sequence[int]): The shift ``(i, j, k)`` of the neighbour, each -1, 0 or 1.

        Returns:
            torch.Tensor: The weight of every voxel's neighbour, [Lx, Ly, Lz].
        """
        # Get Euclidean weight
        weight = self.eucliden_neighbour_weight(coords)
        # Now get weight from anatomy image
        neighbour = get_object_nearest_neighbour(self.anatomy_image, coords)
        weight *= self.similarity_function(self.anatomy_image, neighbour)
        return weight
    
class TopNAnatomyNeighbourWeight(NeighbourWeight):
    r"""Neighbour weights by inverse distance, for only the N neighbours most alike in an anatomical image.

    For each voxel, the ``N_neighbours`` of its 26 neighbours whose anatomical values are closest to its own get their
    inverse distance weight; the others get 0.

    Args:
        anatomy_image (torch.Tensor): The anatomical image (e.g. CT or MRI) on the object grid, [Lx, Ly, Lz].
        N_neighbours (int): Number of neighbours kept per voxel.
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
        """Gives the weight the object grid (its voxel sizes set the distances); the prior calls it.

        Args:
            object_meta (ObjectMeta): The object grid.
        """
        self.object_meta = object_meta
        self.eucliden_neighbour_weight.set_object_meta(object_meta)

    def compute_inclusion_tensor(self):
        """Finds the ``N_neighbours`` most alike neighbours of every voxel and keeps them in ``inclusion_image``.

        ``inclusion_image`` is a boolean tensor [3, 3, 3, Lx, Ly, Lz], True where a neighbour is kept; it is computed
        on the CPU when the weight is made.
        """
        shape = self.anatomy_image.shape
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
        r"""The weight of the neighbour at a shift: inverse distance where it is among the most alike, 0 elsewhere.

        Args:
            coords (Sequence[int]): The shift ``(i, j, k)`` of the neighbour, each -1, 0 or 1.

        Returns:
            torch.Tensor: The weight of every voxel's neighbour, [Lx, Ly, Lz].
        """
        # Get Euclidean weight
        weight = self.eucliden_neighbour_weight(coords)
        # Now get weight from anatomy image
        weight *= self.inclusion_image[coords[0]+1,coords[1]+1,coords[2]+1].to(pytomography.device).to(pytomography.dtype)
        return weight