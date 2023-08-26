"""This module contains classes that implement ordered-subset maximum liklihood iterative reconstruction algorithms. Note that Bayesian algorithm are equivalent to OSEM when no prior is used.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from pytomography.projections import SystemMatrix
import abc
import pytomography
from pytomography.priors import Prior
from pytomography.callbacks import CallBack
from collections.abc import Callable

class OSML():
    r"""Parent class for all variants of ordered subset maximum liklihood algorithms. All child classes must implement the ``__call__`` method to perform reconstruction.

        Args:
            image (torch.Tensor): image data :math:`g` to be reconstructed
            system_matrix (SystemMatrix): system matrix that models the imaging system. In particular, corresponds to :math:`H` in :math:`g=Hf`.
            object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): the initial object guess :math:`f^{0,0}`. If None, then initial guess consists of all 1s. Defaults to None.
            scatter (torch.Tensor): estimate of scatter contribution :math:`s`. Defaults to 0.
            prior (Prior, optional): the Bayesian prior; used to compute :math:`\beta \frac{\partial V}{\partial f}`. If ``None``, then this term is 0. Defaults to None.
    """

    def __init__(
        self,
        image: torch.tensor,
        system_matrix: SystemMatrix,
        object_initial: torch.tensor | None = None,
        scatter: torch.tensor | float = 0,
        prior: Prior = None,
    ) -> None:
        self.system_matrix = system_matrix
        if object_initial is None:
            self.object_prediction = torch.ones((image.shape[0], *self.system_matrix.object_meta.shape)).to(pytomography.device).to(pytomography.dtype)
        else:
            self.object_prediction = object_initial.to(pytomography.device).to(pytomography.dtype)
        self.prior = prior
        self.image = image.to(pytomography.device).to(pytomography.dtype)
        if type(scatter) is torch.Tensor:
            self.scatter = scatter.to(pytomography.device).to(pytomography.dtype)
        else:
            self.scatter = scatter
        if self.prior is not None:
            self.prior.set_object_meta(self.system_matrix.object_meta)
        # Unique string used to identify the type of reconstruction performed
        self.recon_method_string = ''

    def get_subset_splits(
        self,
        n_subsets: int
    ) -> list:
        """Returns a list of subsets (where each subset contains indicies corresponding to projections). For example, if the image consisted of 6 total projections, then ``get_subsets_splits(2)`` would return ``[[0,2,4],[1,3,5]]``.
        
        Args:
            n_subsets (int): number of subsets used in OSEM 

        Returns:
            list: list of index arrays for each subset
        """
        
        indices = torch.arange(self.image.shape[1]).to(torch.long).to(pytomography.device)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        return subset_indices_array

    @abc.abstractmethod
    def __call__(self,
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

class OSEMOSL(OSML):
    r"""Implementation of the ordered subset expectation algorithm using the one-step-late method to include prior information: :math:`\hat{f}^{n,m+1} = \left[\frac{1}{H_m^T 1  + \beta \frac{\partial V}{\partial \hat{f}}|_{\hat{f}=\hat{f}^{n,m}}} H_m^T \left(\frac{g_m}{H_m\hat{f}^{n,m}+s}\right)\right] \hat{f}^{n,m}`.

    Args:
        image (torch.Tensor): image data :math:`g` to be reconstructed
        system_matrix (SystemMatrix): System matrix :math:`H` used in :math:`g=Hf`.
        object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): represents the initial object guess :math:`f^{0,0}` for the algorithm in object space
        scatter (torch.Tensor): estimate of scatter contribution :math:`s`.
        prior (Prior, optional): the Bayesian prior; computes :math:`\beta \frac{\partial V}{\partial f}`. If ``None``, then this term is 0. Defaults to None.
    """
    def _set_recon_name(self, n_iters, n_subsets):
        """Set the unique identifier for the type of reconstruction performed. Useful when saving reconstructions to DICOM files

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
        """
        if self.prior is None:
            self.recon_name = f'OSEM_{n_iters}it{n_subsets}ss'
        else:
            self.recon_name = f'OSEMOSL_{n_iters}it{n_subsets}ss'
    
    def __call__(
        self,
        n_iters: int,
        n_subsets: int,
        callback: CallBack | None = None,
    ) -> torch.tensor:
        """Performs the reconstruction using ``n_iters`` iterations and ``n_subsets`` subsets.

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
            callback (CallBack, optional): Callback function to be evaluated after each subiteration. Defaults to None.
        Returns:
            torch.tensor[batch_size, Lx, Ly, Lz]: reconstructed object
        """
        subset_indices_array = self.get_subset_splits(n_subsets)
        # Scale beta by number of subsets
        if self.prior is not None:
            self.prior.set_beta_scale(1/n_subsets)
        for j in range(n_iters):
            for k, subset_indices in enumerate(subset_indices_array):
                # Set OSL Prior to have object from previous prediction
                if self.prior:
                    self.prior.set_object(torch.clone(self.object_prediction))
                ratio = (self.image+pytomography.delta) / (self.system_matrix.forward(self.object_prediction, angle_subset=subset_indices) + self.scatter + pytomography.delta)
                ratio_BP, norm_BP = self.system_matrix.backward(ratio, angle_subset=subset_indices, return_norm_constant=True)
                if self.prior:
                    norm_BP += self.prior.compute_gradient()
                self.object_prediction = self.object_prediction * ratio_BP /norm_BP
            if callback is not None:
                callback.run(self.object_prediction, n_iter=j)
        # Set unique string for identifying the type of reconstruction
        self._set_recon_name(n_iters, n_subsets)
        return self.object_prediction
    

class OSEMBSR(OSML):
    r"""Implementation of the ordered subset expectation algorithm using the block-sequential-regularized (BSREM) method to include prior information. In particular, each iteration consists of two steps: :math:`\tilde{\hat{f}}^{n,m+1} = \left[\frac{1}{H_m^T 1} H_m^T \left(\frac{g_m}{H_m\hat{f}^{n,m}+s}\right)\right] \hat{f}^{n,m}` followed by :math:`\hat{f}^{n,m+1} = \tilde{\hat{f}}^{n,m+1} \left(1-\beta\frac{\alpha_n}{H_m^T 1}\frac{\partial V}{\partial \tilde{\hat{f}}^{n,m+1}} \right)`.

    Args:
        image (torch.Tensor): image data :math:`g` to be reconstructed
        object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): represents the initial object guess :math:`f^{0,0}` for the algorithm in object space
        system_matrix (SystemMatrix): System matrix :math:`H` used in :math:`g=Hf`.
        scatter (torch.Tensor): estimate of scatter contribution :math:`s`.
        prior (Prior, optional): the Bayesian prior; computes :math:`\beta \frac{\partial V}{\partial f}`. If ``None``, then this term is 0. Defaults to None.

    """
    
    def _set_recon_name(self, n_iters, n_subsets):
        """Set the unique identifier for the type of reconstruction performed. Useful for saving to DICOM files

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
        """
        if self.prior is None:
            self.recon_name = f'OSEM_{n_iters}it{n_subsets}ss'
        else:
            self.recon_name = f'BSREM_{n_iters}it{n_subsets}ss'
    
    def __call__(
        self,
        n_iters: int,
        n_subsets: int,
        relaxation_function: Callable =lambda x: 1,
        callback: CallBack|None =None,
    ) -> torch.tensor:
        r"""Performs the reconstruction using ``n_iters`` iterations and ``n_subsets`` subsets.

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
            relaxation_function (function): Specifies relaxation sequence :math:`\alpha_n` where :math:`n` is the iteration number. Defaults to :math:`\alpha_n=1` for all :math:`n`.
            callback (CallBack, optional): Callback function to be called after each subiteration. Defaults to None.

        Returns:
            torch.tensor[batch_size, Lx, Ly, Lz]: reconstructed object
        """
        subset_indices_array = self.get_subset_splits(n_subsets)
        # Scale beta by number of subsets
        if self.prior is not None:
            self.prior.set_beta_scale(1/n_subsets)
        for j in range(n_iters):
            for k, subset_indices in enumerate(subset_indices_array):
                ratio = (self.image+pytomography.delta) / (self.system_matrix.forward(self.object_prediction, angle_subset=subset_indices) + self.scatter + pytomography.delta)
                ratio_BP, norm_BP = self.system_matrix.backward(ratio, angle_subset=subset_indices, return_norm_constant=True)
                self.object_prediction = self.object_prediction * ratio_BP / norm_BP
                # Apply BSREM after all subsets in this iteration has been ran
                if self.prior:
                    self.prior.set_object(torch.clone(self.object_prediction))
                    gradient = self.prior.compute_gradient()
                    self.object_prediction = self.object_prediction * (1 - (relaxation_function(j)*(gradient +torch.sign(gradient)*pytomography.delta)) / (norm_BP+torch.sign(norm_BP)*pytomography.delta))
                    self.object_prediction[self.object_prediction<=0] = 0
                # Run any callbacks
            if callback:
                callback.run(self.object_prediction, n_iter=j)
        # Set unique string for identifying the type of reconstruction
        self._set_recon_name(n_iters, n_subsets)
        return self.object_prediction

class OSEM(OSEMOSL):
    r"""Implementation of the ordered subset expectation maximum algorithm :math:`\hat{f}^{n,m+1} = \left[\frac{1}{H_m^T 1} H_m^T \left(\frac{g_m}{H_m\hat{f}^{n,m}+s}\right)\right] \hat{f}^{n,m}`.

    Args:
        image (torch.Tensor): image data :math:`g` to be reconstructed
        object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): represents the initial object guess :math:`f^{0,0}` for the algorithm in object space
        system_matrix (SystemMatrix): System matrix :math:`H` used in :math:`g=Hf`.
        scatter (torch.Tensor): estimate of scatter contribution :math:`s`.
    """
    def __init__(
        self,
        image: torch.tensor,
        system_matrix: SystemMatrix,
        object_initial: torch.tensor | None = None,
        scatter: torch.tensor | float = 0,
    ) -> None:
        super(OSEM, self).__init__(image, system_matrix, object_initial, scatter)