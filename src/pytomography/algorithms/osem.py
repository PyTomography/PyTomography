"""This module contains classes that implement ordered-subset maximum liklihood iterative reconstruction algorithms. Such algorithms compute :math:`f_i^{n,m+1}` from :math:`f_i^{n,m}` where :math:`n` is the index for an iteration, and :math:`m` is the index for a subiteration (i.e. for a given subset). The notation is defined such that given :math:`M` total subsets of equal size, :math:`f_i^{n+1,0} \equiv f_i^{n,M}` (i.e. after completing a subiteration for each subset, we start the next iteration). Any class that inherits from this class must implement the ``forward`` method. ``__init__`` initializes the reconstruction algorithm with the image data :math:`g_j`, the forward and back projections used (i.e. networks to compute :math:`\sum_i c_{ij} a_i` and :math:`\sum_j c_{ij} b_j`), the initial object guess :math:`f_i^{0,0}`, the estimated scatter contribution :math:`s_j`, and the Bayesian Prior function :math:`V(f)`. Once the class is initialized, the number of iterations and subsets are specified at recon time when the ``forward`` method is called.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from pytomography.projections import ForwardProjectionNet, BackProjectionNet
import abc
from pytomography.priors import Prior
from pytomography.callbacks import CallBack
from collections.abc import Callable


class OSML(nn.Module):
    r"""Abstract class for different algorithms. The difference between subclasses of this class is the method by which they include prior information. If no prior function is used, they are all equivalent.

        Args:
            image (torch.tensor[batch_size, Lr, Ltheta, Lz]): image data :math:`g_j` to be reconstructed
            object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): represents the initial object guess :math:`f_i^{0,0}` for the algorithm in object space
            forward_projection_net (ForwardProjectionNet): the forward projection network used to compute :math:`\sum_{i} c_{ij} a_i` where :math:`a_i` is the object being forward projected.
            back_projection_net (BackProjectionNet): the back projection network used to compute :math:`\sum_{j} c_{ij} b_j` where :math:`b_j` is the image being back projected.
            scatter (torch.tensor[batch_size, Lr, Ltheta, Lz]): estimate of scatter contribution :math:`s_j`.
            prior (Prior, optional): the Bayesian prior; computes :math:`\beta \frac{\partial V}{\partial f_r}`. If ``None``, then this term is 0. Defaults to None.
    """

    def __init__(
        self,
        image: torch.tensor,
        forward_projection_net: ForwardProjectionNet,
        back_projection_net: BackProjectionNet,
        object_initial: torch.tensor | None = None,
        scatter: torch.tensor | float = 0,
        prior: Prior = None,
    ) -> None:
        super(OSML, self).__init__()
        self.forward_projection_net = forward_projection_net
        self.back_projection_net = back_projection_net
        if forward_projection_net.device!=back_projection_net.device:
            Exception('Forward projection net and back projection net should be on same device')
        self.device = forward_projection_net.device
        if object_initial is None:
            self.object_prediction = torch.ones((image.shape[0], *self.forward_projection_net.object_meta.shape)).to(self.device)
        else:
            self.object_prediction = object_initial.to(self.device)
        self.prior = prior
        self.image = image.to(self.device)
        if type(scatter) is torch.Tensor:
            self.scatter = scatter.to(self.device)
        else:
            self.scatter = scatter
        if self.prior is not None:
            self.prior.set_kernel(self.forward_projection_net.object_meta)

    def get_subset_splits(
        self,
        n_subsets: int,
        n_angles: int,
    ) -> list:
        """Returns a list of arrays; each array contains indices, corresponding to projection numbers, that are used in ordered-subsets. For example, ``get_subsets_splits(2, 6)`` would return ``[[0,2,4],[1,3,5]]``.
        Args:
            n_subsets (int): number of subsets used in OSEM 
            n_angles (int): total number of projections

        Returns:
            list: list of index arrays for each subset
        """
        
        indices = np.arange(n_angles).astype(int)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        return subset_indices_array

    @abc.abstractmethod
    def forward(self,
        n_iters: int,
        n_subsets: int,
        callbacks: CallBack | None = None
    ) -> None:
        """Abstract method for performing reconstruction: must be implemented by subclasses.

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
            callbacks (CallBack, optional): CallBacks to be evaluated after each subiteration. Defaults to None.
        """
        ...
    

class OSEMOSL(OSML):
    r"""Implements the ordered subset expectation algorithm using the one-step-late method to include prior information: :math:`f_i^{n,m+1} = \frac{f_i^{n,m}}{\sum_j c_{ij} + \beta \frac{\partial V}{\partial f_r}|_{f_i=f_i^{n,m}}} \sum_j c_{ij}\frac{g_j}{\sum_i c_{ij}f_i^{n,m}+s_j}`.

    Args:
        object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): represents the initial object guess :math:`f_i^{0,0}` for the algorithm in object space
        forward_projection_net (ForwardProjectionNet): the forward projection network used to compute :math:`\sum_{i} c_{ij} a_i` where :math:`a_i` is the object being forward projected.
        back_projection_net (BackProjectionNet): the back projection network used to compute :math:`\sum_{j} c_{ij} b_j` where :math:`b_j` is the image being back projected.
        prior (Prior, optional): the Bayesian prior; computes :math:`\beta \frac{\partial V}{\partial f_r}`. If ``None``, then this term is 0. Defaults to None.
    """
    def forward(
        self,
        n_iters: int,
        n_subsets: int,
        callback: CallBack | None = None,
        delta: float = 1e-11
    ) -> torch.tensor:
        """Performs the reconstruction using `n_iters` iterations and `n_subsets` subsets.

        Args:
            n_iters (int): _description_
            n_subsets (int): _description_
            callback (CallBack, optional): Callback function to be evaluated after each subiteration. Defaults to None.
            delta (float, optional): Used to prevent division by zero when calculating ratio, defaults to 1e-11.

        Returns:
            torch.tensor[batch_size, Lx, Ly, Lz]: reconstructed object
        """
        subset_indices_array = self.get_subset_splits(n_subsets, self.image.shape[1])
        # Scale beta by number of subsets
        if self.prior is not None:
            self.prior.set_beta_scale(1/n_subsets)
        for j in range(n_iters):
            for k, subset_indices in enumerate(subset_indices_array):
                # Set OSL Prior to have object from previous prediction
                if self.prior:
                    self.prior.set_object(torch.clone(self.object_prediction))
                ratio = (self.image+delta) / (self.forward_projection_net(self.object_prediction, angle_subset=subset_indices) + self.scatter + delta)
                self.object_prediction = self.object_prediction * self.back_projection_net(ratio, angle_subset=subset_indices, normalize=True, prior=self.prior)
                if callback is not None:
                    callback.run(self.object_prediction)
        return self.object_prediction
    

class OSEMBSR(OSML):
    r"""Implements the ordered subset expectation algorithm using the block-sequential-regularized (BSREM) method to include prior information. In particular, each iteration consists of two steps: :math:`\tilde{f}_i^{n,m+1} = \frac{f_i^{n,m}}{\sum_j c_{ij}} \sum_j c_{ij}\frac{g_j^m}{\sum_i c_{ij}f_i^{n,m}+s_j}` followed by :math:`f_i^{n,m+1} = \tilde{f}_i^{n,m+1} \left(1-\beta\frac{\alpha_n}{\sum_j c_{ij}}\frac{\partial V}{\partial \tilde{f}_i^{n,m+1}} \right)`.

    Args:
        object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): represents the initial object guess :math:`f_i^{0,0}` for the algorithm in object space
        forward_projection_net (ForwardProjectionNet): the forward projection network used to compute :math:`\sum_{i} c_{ij} a_i` where :math:`a_i` is the object being forward projected.
        back_projection_net (BackProjectionNet): the back projection network used to compute :math:`\sum_{j} c_{ij} b_j` where :math:`b_j` is the image being back projected.
        prior (Prior, optional): the Bayesian prior; computes :math:`\beta \frac{\partial V}{\partial f_r}`. If ``None``, then this term is 0. Defaults to None.

    """
    
    def forward(
        self,
        n_iters: int,
        n_subsets: int,
        relaxation_function: Callable =lambda x: 1,
        callback: CallBack|None =None,
        delta: float = 1e-11
    ) -> torch.tensor:
        r"""Performs the reconstruction using `n_iters` iterations and `n_subsets` subsets.

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
            relaxation_function (function): Specifies relaxation sequence :math:`\alpha_n` where :math:`n` is the iteration number. Defaults to :math:`\alpha_n=1` for all :math:`n`.
            callback (CallBack, optional): Callback function to be called after each subiteration. Defaults to None.
            delta (_type_, optional): Used to prevent division by zero when calculating ratio, defaults to 1e-11.

        Returns:
            torch.tensor[batch_size, Lx, Ly, Lz]: reconstructed object
        """
        subset_indices_array = self.get_subset_splits(n_subsets, self.image.shape[1])
        # Scale beta by number of subsets
        if self.prior is not None:
            self.prior.set_beta_scale(1/n_subsets)
        for j in range(n_iters):
            for k, subset_indices in enumerate(subset_indices_array):
                ratio = (self.image+delta) / (self.forward_projection_net(self.object_prediction, angle_subset=subset_indices) + self.scatter + delta)
                bp, norm_factor = self.back_projection_net(ratio, angle_subset=subset_indices, normalize=True, return_norm_constant=True)
                self.object_prediction = self.object_prediction * bp
                # Apply BSREM after all subsets in this iteration has been ran
                if self.prior:
                    self.prior.set_object(torch.clone(self.object_prediction))
                    self.object_prediction = self.object_prediction * (1 - relaxation_function(j)*self.prior() / norm_factor)
                    self.object_prediction[self.object_prediction<=0] = 0
                # Run any callbacks
                if callback:
                    callback.run(self.object_prediction)
        return self.object_prediction