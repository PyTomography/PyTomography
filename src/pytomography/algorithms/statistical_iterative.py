"""This module contains classes that implement statistical iterative reconstruction algorithms.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from pytomography.projectors import SystemMatrix
import abc
import pytomography
from pytomography.priors import Prior
from pytomography.callbacks import Callback
from pytomography.transforms import KEMTransform
from pytomography.projectors import KEMSystemMatrix
from collections.abc import Callable

class StatisticalIterative():
    r"""Parent class for all statistical iterative algorithms. All child classes must implement the ``__call__`` method to perform reconstruction.

        Args:
            projections (torch.Tensor): photopeak window projection data :math:`g` to be reconstructed
            system_matrix (SystemMatrix): system matrix that models the imaging system. In particular, corresponds to :math:`H` in :math:`g=Hf`.
            object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): the initial object guess :math:`f^{0,0}`. If None, then initial guess consists of all 1s. Defaults to None.
            scatter (torch.Tensor): estimate of scatter contribution :math:`s`. Defaults to 0.
            prior (Prior, optional): the Bayesian prior; used to compute :math:`\beta \frac{\partial V}{\partial f}`. If ``None``, then this term is 0. Defaults to None.
    """

    def __init__(
        self,
        projections: torch.tensor,
        system_matrix: SystemMatrix,
        object_initial: torch.tensor | None = None,
        scatter: torch.tensor | float = 0,
        prior: Prior = None,
    ) -> None:
        self.system_matrix = system_matrix
        if object_initial is None:
            self.object_prediction = torch.ones((projections.shape[0], *self.system_matrix.object_meta.shape)).to(pytomography.device).to(pytomography.dtype)
        else:
            self.object_prediction = object_initial.to(pytomography.device).to(pytomography.dtype)
        self.prior = prior
        self.proj = projections.to(pytomography.device).to(pytomography.dtype)
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
        """Returns a list of subsets (where each subset contains indicies corresponding to different angles). For example, if the projections consisted of 6 total angles, then ``get_subsets_splits(2)`` would return ``[[0,2,4],[1,3,5]]``.
        
        Args:
            n_subsets (int): number of subsets used in OSEM 

        Returns:
            list: list of index arrays for each subset
        """
        
        indices = torch.arange(self.proj.shape[1]).to(torch.long).to(pytomography.device)
        subset_indices_array = []
        for i in range(n_subsets):
            subset_indices_array.append(indices[i::n_subsets])
        return subset_indices_array

    @abc.abstractmethod
    def __call__(self,
        n_iters: int,
        n_subsets: int,
        callbacks: Callback | None = None
    ) -> None:
        """Abstract method for performing reconstruction: must be implemented by subclasses.

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
            callbacks (Callback, optional): Callbacks to be evaluated after each subiteration. Defaults to None.
        """

class OSEMOSL(StatisticalIterative):
    r"""Implementation of the ordered subset expectation algorithm using the one-step-late method to include prior information: :math:`\hat{f}^{n,m+1} = \left[\frac{1}{H_m^T 1  + \beta \frac{\partial V}{\partial \hat{f}}|_{\hat{f}=\hat{f}^{n,m}}} H_m^T \left(\frac{g_m}{H_m\hat{f}^{n,m}+s}\right)\right] \hat{f}^{n,m}`.

    Args:
        proj (torch.Tensor): projection data :math:`g` to be reconstructed
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
        callback: Callback | None = None,
    ) -> torch.tensor:
        """Performs the reconstruction using ``n_iters`` iterations and ``n_subsets`` subsets.

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
            callback (Callback, optional): Callback function to be evaluated after each subiteration. Defaults to None.
        Returns:
            torch.tensor[batch_size, Lx, Ly, Lz]: reconstructed object
        """
        subset_indices_array = self.get_subset_splits(n_subsets)
        for j in range(n_iters):
            for k, subset_indices in enumerate(subset_indices_array):
                # Set OSL Prior to have object from previous prediction
                if self.prior:
                    self.prior.set_object(torch.clone(self.object_prediction))
                ratio = (self.proj+pytomography.delta) / (self.system_matrix.forward(self.object_prediction, angle_subset=subset_indices) + self.scatter + pytomography.delta)
                ratio_BP, norm_BP = self.system_matrix.backward(ratio, angle_subset=subset_indices, return_norm_constant=True)
                if self.prior:
                    self.prior.set_beta_scale(len(subset_indices) / self.proj.shape[1])
                    norm_BP += self.prior.compute_gradient()
                self.object_prediction = self.object_prediction * ratio_BP / (norm_BP + pytomography.delta)
            if callback is not None:
                callback.run(self.object_prediction, n_iter=j)
        # Set unique string for identifying the type of reconstruction
        self._set_recon_name(n_iters, n_subsets)
        return self.object_prediction
    

class BSREM(StatisticalIterative):
    r"""Implementation of the block-sequential-regularized (BSREM) reconstruction algorithm: :math:`\hat{f}^{n,m+1} = \hat{f}^{n,m} + \alpha_n D \left[H_m^T \left(\frac{g_m}{H_m \hat{f}^{n,m} + s} -1 \right) - \beta \nabla_{f^{n,m}} V \right]`. The implementation of this algorithm corresponds to Modified BSREM-II with :math:`U=\infty`, :math:`t=0`, and :math:`\epsilon=0` (see https://ieeexplore.ieee.org/document/1207396). There is one difference in this implementation: rather than using FBP to get an initial estimate (as is done in the paper), a single iteration of OSEM is used; this initialization is required here due to the requirement for global scaling (see discussion on page 620 of paper).

    Args:
        proj (torch.Tensor): projection data :math:`g` to be reconstructed
        object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): represents the initial object guess :math:`f^{0,0}` for the algorithm in object space
        system_matrix (SystemMatrix): System matrix :math:`H` used in :math:`g=Hf`.
        scatter (torch.Tensor): estimate of scatter contribution :math:`s`.
        prior (Prior, optional): the Bayesian prior; computes :math:`\beta \frac{\partial V}{\partial f}`. If ``None``, then this term is 0. Defaults to None.
        relaxation_function (Callable, optional): Sequence :math:`\alpha_n` used for relaxation. Defaults to :math:`\alpha_n=1/(n+1)`.
        scaling_matrix_type (str, optional): The form of the scaling matrix :math:`D` used. If ``subind_norm`` (sub-iteration independent + normalized), then :math:`D=\left(S_m/M \cdot H^T 1 \right)^{-1}`. If ``subdep_norm`` (sub-iteration dependent + normalized) then :math:`D = \left(H_m^T 1\right)^{-1}`. See section III.D in the paper above for a discussion on this.

    """
    
    def __init__(
        self,
        projections: torch.tensor,
        system_matrix: SystemMatrix,
        object_initial: torch.tensor | None = None,
        scatter: torch.tensor | float = 0,
        prior: Prior = None,
        relaxation_function: Callable = lambda x: 1,
        scaling_matrix_type: str = 'subind_norm',
    ) -> None:
        # Initial estimate given by OSEM
        if object_initial is None:
            object_initial = OSEM(projections, system_matrix, object_initial, scatter)(1,1)
        super(BSREM, self).__init__(projections, system_matrix, object_initial, scatter, prior)
        self.relaxation_function = relaxation_function
        self.scaling_matrix_type = scaling_matrix_type
    
    def _set_recon_name(self, n_iters, n_subsets):
        """Set the unique identifier for the type of reconstruction performed. Useful for saving to DICOM files

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
        """
        self.recon_name = f'BSREM_{n_iters}it{n_subsets}ss'
    
    def __call__(
        self,
        n_iters: int,
        n_subsets: int,
        callback: Callback|None =None,
    ) -> torch.tensor:
        r"""Performs the reconstruction using ``n_iters`` iterations and ``n_subsets`` subsets.

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
            callback (Callback, optional): Callback function to be called after each subiteration. Defaults to None.

        Returns:
            torch.tensor[batch_size, Lx, Ly, Lz]: reconstructed object
        """
        subset_indices_array = self.get_subset_splits(n_subsets)
        if self.scaling_matrix_type=='subind_norm':
            # Normalization factor does not depend on subset index
            _, norm_BP_allsubsets = self.system_matrix.backward(self.proj, return_norm_constant=True)
        for j in range(n_iters):
            for k, subset_indices in enumerate(subset_indices_array):
                ratio = (self.proj+pytomography.delta) / (self.system_matrix.forward(self.object_prediction, angle_subset=subset_indices) + self.scatter + pytomography.delta)
                # Obtain the scaling matrix D and the ratio to be back projected
                if self.scaling_matrix_type=='subdep_norm':
                    quantity_BP, norm_BP = self.system_matrix.backward(ratio-1, angle_subset=subset_indices, return_norm_constant=True)
                    scaling_matrix = 1 / (norm_BP+pytomography.delta)
                elif self.scaling_matrix_type=='subind_norm':
                    quantity_BP = self.system_matrix.backward(ratio-1, angle_subset=subset_indices)
                    norm_BP = norm_BP_allsubsets * len(subset_indices) / self.proj.shape[1]
                    scaling_matrix = 1 / (norm_BP+pytomography.delta)
                if self.prior:
                    self.prior.set_beta_scale(len(subset_indices) / self.proj.shape[1])
                    self.prior.set_object(torch.clone(self.object_prediction))
                    gradient = self.prior.compute_gradient()
                else:
                    gradient = 0
                self.object_prediction = self.object_prediction * (1 + scaling_matrix * self.relaxation_function(j) * (quantity_BP - gradient))
                # Get rid of small negative values
                self.object_prediction[self.object_prediction<0] = 0
            if callback:
                callback.run(self.object_prediction, n_iter=j)
        # Set unique string for identifying the type of reconstruction
        self._set_recon_name(n_iters, n_subsets)
        return self.object_prediction

class OSEM(OSEMOSL):
    r"""Implementation of the ordered subset expectation maximum algorithm :math:`\hat{f}^{n,m+1} = \left[\frac{1}{H_m^T 1} H_m^T \left(\frac{g_m}{H_m\hat{f}^{n,m}+s}\right)\right] \hat{f}^{n,m}`.

    Args:
        proj (torch.Tensor): projection data :math:`g` to be reconstructed
        object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): represents the initial object guess :math:`f^{0,0}` for the algorithm in object space
        system_matrix (SystemMatrix): System matrix :math:`H` used in :math:`g=Hf`.
        scatter (torch.Tensor): estimate of scatter contribution :math:`s`.
    """
    def __init__(
        self,
        projections: torch.tensor,
        system_matrix: SystemMatrix,
        object_initial: torch.tensor | None = None,
        scatter: torch.tensor | float = 0,
    ) -> None:
        super(OSEM, self).__init__(projections, system_matrix, object_initial, scatter)
        
class KEM(OSEM):
    r"""Implementation of the KEM reconstruction algorithm given by :math:`\hat{\alpha}^{n,m+1} = \left[\frac{1}{K^T H_m^T 1} K^T H_m^T \left(\frac{g_m}{H_m K \hat{\alpha}^{n,m}+s}\right)\right] \hat{\alpha}^{n,m}` and where the final predicted object is :math:`\hat{f}^{n,m} = K \hat{\alpha}^{n,m}`.

    Args:
        proj (torch.Tensor): projection data :math:`g` to be reconstructed
        system_matrix (SystemMatrix): System matrix :math:`H` used in :math:`g=Hf`.
        kem_transform (KEMTransform): The transform corresponding to the matrix :math:`K`.
        object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): represents the initial object guess :math:`f^{0,0}` for the algorithm in object space
        scatter (torch.Tensor): estimate of scatter contribution :math:`s`.
    """
    def __init__(
        self,
        projections: torch.tensor,
        system_matrix: SystemMatrix,
        kem_transform: KEMTransform,
        object_initial: torch.tensor | None = None,
        scatter: torch.tensor | float = 0,
    ) -> None:
        kem_transform.configure(system_matrix.object_meta, system_matrix.proj_meta)
        self.kem_transform = kem_transform
        system_matrix_kem = KEMSystemMatrix(system_matrix, kem_transform)
        super(KEM, self).__init__(projections, system_matrix_kem, object_initial, scatter)
        
    def __call__(
        self,
        n_iters: int,
        n_subsets: int,
        callback: Callback | None = None,
    ) -> torch.tensor:
        r"""Performs the reconstruction using ``n_iters`` iterations and ``n_subsets`` subsets.

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
            callback (Callback, optional): Callback function to be called after each subiteration. Defaults to None.

        Returns:
            torch.tensor[batch_size, Lx, Ly, Lz]: reconstructed object
        """
        object_prediction = super(KEM, self).__call__(n_iters, n_subsets, callback)
        return self.kem_transform.forward(object_prediction)