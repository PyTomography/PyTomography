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
from pytomography.transforms import KEMTransform, CutOffTransform
from pytomography.projectors import KEMSystemMatrix
from collections.abc import Callable

def get_projection_subset(projections, subset_indices, device):
    if (len(projections.shape)>1)*(subset_indices is not None):
        proj_subset = projections[:,subset_indices.to(device)]
    else:
        proj_subset = projections
    return proj_subset

class StatisticalIterative():
    r"""Parent class for all statistical iterative algorithms. All child classes must implement the ``__call__`` method to perform reconstruction.

        Args:
            projections (torch.Tensor): photopeak window projection data :math:`g` to be reconstructed
            system_matrix (SystemMatrix): system matrix that models the imaging system. In particular, corresponds to :math:`H` in :math:`g=Hf`.
            object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): the initial object guess :math:`f^{0,0}`. If None, then initial guess consists of all 1s. Defaults to None.
            scatter (torch.Tensor): estimate of scatter contribution :math:`s`. Defaults to 0.
            prior (Prior, optional): the Bayesian prior; used to compute :math:`\beta \frac{\partial V}{\partial f}`. If ``None``, then this term is 0. Defaults to None.
            precompute_normalization_factors (bool). Whether or not to precompute the normalization factors :math:`H_m^T 1` for each subset :math:`m` before reconstruction. This saves computational time during each iteration, but requires more GPU memory. Defaults to True.
            device (str): The device correpsonding to the tensors output by the system matrix. In some cases, although the system matrix implementation uses ``pytomography.device`` in its internal computation, it will output tensors on the CPU due to their size (such as in listmode PET). Defaults to ``pytomography.device``. 
    """

    def __init__(
        self,
        projections: torch.tensor,
        system_matrix: SystemMatrix,
        object_initial: torch.tensor | None = None,
        scatter: torch.tensor | None = None,
        prior: Prior = None,
        precompute_normalization_factors: bool = True,
        device: str = pytomography.device
    ) -> None:
        self.device = device
        self.system_matrix = system_matrix
        if object_initial is None:
            self.object_prediction = torch.ones((projections.shape[0], *self.system_matrix.object_meta.shape)).to(self.device).to(pytomography.dtype)
        else:
            self.object_prediction = object_initial.to(self.device).to(pytomography.dtype)
        self.prior = prior
        self.proj = projections.to(self.device).to(pytomography.dtype)
        if type(scatter) is torch.Tensor:
            self.scatter = scatter.to(self.device).to(pytomography.dtype)
        else:
            self.scatter = torch.zeros(projections.shape).to(self.device).to(pytomography.dtype)
        if self.prior is not None:
            self.prior.set_object_meta(self.system_matrix.object_meta)
        # Unique string used to identify the type of reconstruction performed
        self.recon_method_string = ''
        self.precompute_normalization_factors = precompute_normalization_factors
        # Set n_subsets_previous, which is used for determining whether normalization factors need to be recomputed during successive calls to __call__
        self.n_subsets_previous = -1 

    @abc.abstractmethod
    def __call__(self,
        n_iters: int,
        n_subsets: int,
        callback: Callback | None = None
    ) -> None:
        """Abstract method for performing reconstruction: must be implemented by subclasses.

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
            callbacks (Callback, optional): Callbacks to be evaluated after each subiteration. Defaults to None.
        """
    def _compute_callback(self, n_iter: int):
        self.callback.run(self.object_prediction, n_iter)

class OSEMOSL(StatisticalIterative):
    r"""Implementation of the ordered subset expectation algorithm using the one-step-late method to include prior information: :math:`\hat{f}^{n,m+1} = \left[\frac{1}{H_m^T 1  + \beta \frac{\partial V}{\partial \hat{f}}|_{\hat{f}=\hat{f}^{n,m}}} H_m^T \left(\frac{g_m}{H_m\hat{f}^{n,m}+s}\right)\right] \hat{f}^{n,m}`.

    Args:
        projections (torch.Tensor): photopeak window projection data :math:`g` to be reconstructed
        system_matrix (SystemMatrix): system matrix that models the imaging system. In particular, corresponds to :math:`H` in :math:`g=Hf`.
        object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): the initial object guess :math:`f^{0,0}`. If None, then initial guess consists of all 1s. Defaults to None.
        scatter (torch.Tensor): estimate of scatter contribution :math:`s`. Defaults to 0.
        prior (Prior, optional): the Bayesian prior; used to compute :math:`\beta \frac{\partial V}{\partial f}`. If ``None``, then this term is 0. Defaults to None.
        precompute_normalization_factors (bool). Whether or not to precompute the normalization factors :math:`H_m^T 1` for each subset :math:`m` before reconstruction. This saves computational time during each iteration, but requires more GPU memory. Defaults to True.
        device (str): The device correpsonding to the tensors output by the system matrix. In some cases, although the system matrix implementation uses ``pytomography.device`` in its internal computation, it will output tensors on the CPU due to their size (such as in listmode PET). Defaults to ``pytomography.device``. 
    """
    def _set_recon_name(self, n_iters: int, n_subsets: int):
        """Set the unique identifier for the type of reconstruction performed. Useful when saving reconstructions to DICOM files

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
        """
        if self.prior is None:
            self.recon_name = f'OSEM_{n_iters}it{n_subsets}ss'
        else:
            self.recon_name = f'OSEMOSL_{n_iters}it{n_subsets}ss'
            
    def _compute_normalization_factors(self):
        """Computes normalization factors :math:`H_m^T 1` for all subsets :math:`m`.
        """
        # Looks for change in n_subsets during successive calls to __call__. First call this is always true, since n_subsets_previous is initially set to -1
        if self.n_subsets_previous!=self.n_subsets:
            self.norm_BPs = []
            for k in range(self.n_subsets):
                self.norm_BPs.append(self.system_matrix.compute_normalization_factor(k))
        
    def __call__(
        self,
        n_iters: int,
        n_subsets: int,
        n_subset_specific: int | None = None,
        callback: Callback | None = None,
    ) -> torch.tensor:
        """Performs the reconstruction using ``n_iters`` iterations and ``n_subsets`` subsets.

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
            n_subset_specific (int): Iterate only over the subset specified. Defaults to None
            callback (Callback, optional): Callback function to be evaluated after each subiteration. Defaults to None.
        Returns:
            torch.tensor[batch_size, Lx, Ly, Lz]: reconstructed object
        """
        self.n_subsets = n_subsets
        self.callback = callback
        self.system_matrix.set_n_subsets(n_subsets)
        self._compute_normalization_factors()
        for j in range(n_iters):
            for k in range(n_subsets):
                if n_subset_specific is not None:
                    # For considering only a specific subset
                    if n_subset_specific!=k:
                        continue
                # Set OSL Prior to have object from previous prediction
                if self.prior:
                    self.prior.set_object(torch.clone(self.object_prediction).to(pytomography.device))
                # Get subsets
                proj_subset = self.system_matrix.get_projection_subset(self.proj, k)
                scatter_subset = self.system_matrix.get_projection_subset(self.scatter, k)
                # Compute ratio
                ratio = (proj_subset+pytomography.delta) / (self.system_matrix.forward(self.object_prediction, subset_idx=k) + scatter_subset + pytomography.delta)
                # Back project ratio
                ratio_BP = self.system_matrix.backward(ratio, subset_idx=k)
                norm_BP = self.norm_BPs[k].to(self.device)
                if self.prior:
                    self.prior.set_beta_scale(self.system_matrix.get_weighting_subset(k))
                    prior = self.prior.compute_gradient().to(self.device)
                    prior[-prior>=norm_BP] = 0 # prevents negative updates
                else:
                    prior = 0
                self.object_prediction = self.object_prediction * ratio_BP / (norm_BP + prior + pytomography.delta)
            if self.callback is not None:
                self._compute_callback(n_iter=j)
        # Set unique string for identifying the type of reconstruction
        self._set_recon_name(n_iters, n_subsets)
        # Set previous subsets used, for if normalization factors need to be recomputed for different subset config in future __call__
        self.n_subsets_previous = n_subsets
        return self.object_prediction
    

class BSREM(StatisticalIterative):
    r"""Implementation of the block-sequential-regularized (BSREM) reconstruction algorithm: :math:`\hat{f}^{n,m+1} = \hat{f}^{n,m} + \alpha_n D \left[H_m^T \left(\frac{g_m}{H_m \hat{f}^{n,m} + s} -1 \right) - \beta \nabla_{f^{n,m}} V \right]`. The implementation of this algorithm corresponds to Modified BSREM-II with :math:`U=\infty`, :math:`t=0`, and :math:`\epsilon=0` (see https://ieeexplore.ieee.org/document/1207396). There is one difference in this implementation: rather than using FBP to get an initial estimate (as is done in the paper), a single iteration of OSEM is used; this initialization is required here due to the requirement for global scaling (see discussion on page 620 of paper).

    Args:
        projections (torch.Tensor): projection data :math:`g` to be reconstructed.
        system_matrix (SystemMatrix): System matrix :math:`H` used in :math:`g=Hf`.
        object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): represents the initial object guess :math:`f^{0,0}` for the algorithm in object space
        scatter (torch.Tensor): estimate of scatter contribution :math:`s`.
        prior (Prior, optional): the Bayesian prior; computes :math:`\beta \frac{\partial V}{\partial f}`. If ``None``, then this term is 0. Defaults to None.
        relaxation_function (Callable, optional): Sequence :math:`\alpha_n` used for relaxation. Defaults to :math:`\alpha_n=1/(n+1)`.
        scaling_matrix_type (str, optional): The form of the scaling matrix :math:`D` used. If ``subind_norm`` (sub-iteration independent + normalized), then :math:`D=\left(S_m/M \cdot H^T 1 \right)^{-1}`. If ``subdep_norm`` (sub-iteration dependent + normalized) then :math:`D = \left(H_m^T 1\right)^{-1}`. See section III.D in the paper above for a discussion on this.
        device (str): The device correpsonding to the tensors output by the system matrix. In some cases, although the system matrix implementation uses ``pytomography.device`` in its internal computation, it will output tensors on the CPU due to their size (such as in listmode PET). Defaults to ``pytomography.device``. 
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
        precompute_normalization_factors: bool = True,
        device: str = pytomography.device
    ) -> None:
        self.device = device
        # Initial estimate given by OSEM
        if object_initial is None:
            object_initial = OSEM(projections, system_matrix, object_initial, scatter, precompute_normalization_factors, device)(1,1)
        super(BSREM, self).__init__(projections, system_matrix, object_initial, scatter, prior, precompute_normalization_factors, device)
        self.relaxation_function = relaxation_function
        self.scaling_matrix_type = scaling_matrix_type
        
    def _compute_normalization_factors(self):
        """Computes normalization factors :math:`H_m^T 1` for all subsets :math:`m`.
        """
        # Looks for change in n_subsets during successive calls to __call__. First call this is always true, since n_subsets_previous is initially set to -1
        if self.n_subsets_previous!=self.n_subsets:
            self.norm_BPs = []
            for subset_indices in self.subset_indices_array:
                self.norm_BPs.append(self.system_matrix.compute_normalization_factor(subset_indices))
    
    def _set_recon_name(self, n_iters: int, n_subsets: int):
        """Set the unique identifier for the type of reconstruction performed. Useful for saving to DICOM files

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
        """
        self.recon_name = f'BSREM_{n_iters}it{n_subsets}ss'
        
    def _scale_prior_gradient(self, gradient: torch.tensor):
        """Used to scale gradient to avoid divisional errors in null regions when using CutOffTransform

        Args:
            gradient (torch.tensor): Gradient returned by prior function

        Returns:
            torch.tensor: New gradient tensor where values are set to 0 outside the cutoff region.
        """
        proj2proj_types = [type(x) for x in self.system_matrix.proj2proj_transforms]
        if CutOffTransform in proj2proj_types:
            idx = proj2proj_types.index(CutOffTransform)
            # Note: this only works because CutoffTransform can be used on objects or projections even though its a projection space transform
            gradient = self.system_matrix.proj2proj_transforms[idx].forward(gradient)
        return gradient
    
    def __call__(
        self,
        n_iters: int,
        n_subsets: int,
        n_subset_specific: None | int = None,
        callback: Callback | None = None,
    ) -> torch.tensor:
        r"""Performs the reconstruction using ``n_iters`` iterations and ``n_subsets`` subsets.

        Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
            n_subset_specific (int): Iterate only over the subset specified. Defaults to None
            callback (Callback, optional): Callback function to be called after each subiteration. Defaults to None.

        Returns:
            torch.tensor[batch_size, Lx, Ly, Lz]: reconstructed object
        """
        self.callback = callback
        self.system_matrix.set_n_subsets(n_subsets)
        # Set normalization factor H^T 1 if it hasnt already been set in previous call to __call__
        if self.scaling_matrix_type=='subdep_norm':
            self._compute_normalization_factors()
        elif (self.scaling_matrix_type=='subind_norm') * (not(hasattr(self, 'norm_BP_allsubsets'))):
            self.norm_BP_allsubsets = self.system_matrix.compute_normalization_factor()
        for j in range(n_iters):
            for k in range(n_subsets):
                if n_subset_specific is not None:
                    # For considering only a specific subset
                    if n_subset_specific!=k:
                        continue
                # Compute subsets
                proj_subset = self.system_matrix.get_projection_subset(self.proj, k)
                scatter_subset = self.system_matrix.get_projection_subset(self.scatter, k)
                # Compute ratio
                ratio = (proj_subset + pytomography.delta) / (self.system_matrix.forward(self.object_prediction, subset_idx=k) + scatter_subset + pytomography.delta)
                # Obtain the scaling matrix D and the ratio to be back projected
                if self.scaling_matrix_type=='subdep_norm':
                    ratio_BP = self.system_matrix.backward(ratio, subset_idx=k)
                    norm_BP = self.norm_BPs[k]
                    quantity_BP = ratio_BP - norm_BP
                    scaling_matrix = 1 / (norm_BP+pytomography.delta)
                elif self.scaling_matrix_type=='subind_norm':
                    ratio_BP = self.system_matrix.backward(ratio, subset_idx=k)
                    norm_BP = self.norm_BP_allsubsets * self.system_matrix.get_weighting_subset(k)
                    quantity_BP = ratio_BP - norm_BP
                    scaling_matrix = 1 / (norm_BP+pytomography.delta)
                if self.prior:
                    self.prior.set_beta_scale(self.system_matrix.get_weighting_subset(k))
                    self.prior.set_object(torch.clone(self.object_prediction).to(pytomography.device))
                    gradient = self.prior.compute_gradient().to(self.device)
                    # Gradient not applied to null regions
                    self._scale_prior_gradient(gradient)
                else:
                    gradient = 0
                self.object_prediction = self.object_prediction * (1 + scaling_matrix * self.relaxation_function(j) * (quantity_BP - gradient))
                # Get rid of small negative values
                self.object_prediction[self.object_prediction<0] = 0
            if self.callback is not None:
                self._compute_callback(n_iter = j)
        # Set unique string for identifying the type of reconstruction
        self._set_recon_name(n_iters, n_subsets)
        return self.object_prediction
    
class OSEM(OSEMOSL):
    r"""Implementation of the ordered subset expectation maximum algorithm :math:`\hat{f}^{n,m+1} = \left[\frac{1}{H_m^T 1} H_m^T \left(\frac{g_m}{H_m\hat{f}^{n,m}+s}\right)\right] \hat{f}^{n,m}`.

    Args:
        projections (torch.Tensor): photopeak window projection data :math:`g` to be reconstructed
        system_matrix (SystemMatrix): system matrix that models the imaging system. In particular, corresponds to :math:`H` in :math:`g=Hf`.
        object_initial (torch.tensor[batch_size, Lx, Ly, Lz]): the initial object guess :math:`f^{0,0}`. If None, then initial guess consists of all 1s. Defaults to None.
        scatter (torch.Tensor): estimate of scatter contribution :math:`s`. Defaults to 0.
        precompute_normalization_factors (bool). Whether or not to precompute the normalization factors :math:`H_m^T 1` for each subset :math:`m` before reconstruction. This saves computational time during each iteration, but requires more GPU memory. Defaults to True.
    """
    def __init__(
        self,
        projections: torch.tensor,
        system_matrix: SystemMatrix,
        object_initial: torch.tensor | None = None,
        scatter: torch.tensor | float = 0,
        precompute_normalization_factors: bool = True,
        device: str = pytomography.device,
    ) -> None:
        super(OSEM, self).__init__(projections, system_matrix, object_initial, scatter, precompute_normalization_factors=precompute_normalization_factors, device=device)
        
class KEM(OSEM):
    r"""Implementation of the KEM reconstruction algorithm given by :math:`\hat{\alpha}^{n,m+1} = \left[\frac{1}{K^T H_m^T 1} K^T H_m^T \left(\frac{g_m}{H_m K \hat{\alpha}^{n,m}+s}\right)\right] \hat{\alpha}^{n,m}` and where the final predicted object is :math:`\hat{f}^{n,m} = K \hat{\alpha}^{n,m}`.

    Args:
        projections (torch.Tensor): projection data :math:`g` to be reconstructed
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
        
    def _compute_callback(self, n_iter: int):
        r"""Computes callback for KEM transform; this is reimplemented here because the `self.object_prediction` corresponds to the :math:`\alpha` value and not :math:`f`. As such, the `KEMTransform` needs to be applied before the object is input to the callback.

        Args:
            n_iter (int): _description_
        """
        self.callback.run(self.kem_transform.forward(self.object_prediction), n_iter)
        
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
    
    
class DIPRecon(StatisticalIterative):
    r"""Implementation of the Deep Image Prior reconstruction technique (see https://ieeexplore.ieee.org/document/8581448). This reconstruction technique requires an instance of a user-defined ``prior_network`` that implements two functions: (i) a ``fit`` method that takes in an ``object`` (:math:`x`) which the network ``f(z;\theta)`` is subsequently fit to, and (ii) a ``predict`` function that returns the current network prediction :math:`f(z;\theta)`. For more details, see the Deep Image Prior tutorial.

        Args:
            projections (torch.tensor): projection data :math:`g` to be reconstructed
            system_matrix (SystemMatrix): System matrix :math:`H` used in :math:`g=Hf`.
            prior_network (nn.Module): User defined prior network that implements the neural network ``f(z;\theta)``
            rho (float, optional): Value of :math:`\rho` used in the optimization procedure. Defaults to 1.
            scatter (torch.tensor | float, optional): Projection space scatter estimate. Defaults to 0.
            precompute_normalization_factors (bool, optional): Whether to precompute :math:`H_m^T 1` and store on GPU in the OSEM network before reconstruction. Defaults to True.
        """
    def __init__(
        self,
        projections: torch.tensor,
        system_matrix: SystemMatrix,
        prior_network: nn.Module,
        rho: float = 3e-3,
        scatter: torch.tensor | float = 0,
        precompute_normalization_factors: bool = True,
        
    ) -> None:
        
        super(DIPRecon, self).__init__(
            projections,
            system_matrix,
            scatter=scatter)
        self.osem_network = OSEM(
            projections,
            system_matrix,
            scatter=scatter,
            precompute_normalization_factors=precompute_normalization_factors)
        self.prior_network = prior_network
        self.rho = rho
        
    def __call__(
        self,
        n_iters,
        subit1,
        n_subsets_osem=1,
        callback=None,
    ):  
        r"""Implementation of Algorithm 1 in https://ieeexplore.ieee.org/document/8581448. This implementation gives the additional option to use ordered subsets. The quantity SubIt2 specified in the paper is controlled by the user-defined ``prior_network`` class.

        Args:
            n_iters (int): Number of iterations (MaxIt in paper)
            subit1 (int): Number of OSEM iterations before retraining neural network (SubIt1 in paper)
            n_subsets_osem (int, optional): Number of subsets to use in OSEM reconstruction. Defaults to 1.

        Returns:
            torch.Tensor: Reconstructed image
        """
        self.callback = callback
        # Initialize quantities
        mu = 0 
        norm_BP = self.system_matrix.compute_normalization_factor()
        x = self.prior_network.predict()
        x_network = x.clone()
        for _ in range(n_iters):
            for j in range(subit1):
                for k in range(n_subsets_osem):
                    self.osem_network.object_prediction = nn.ReLU()(x.clone())
                    x_EM = self.osem_network(1,n_subsets_osem, k)
                    x = 0.5 * (x_network - mu - norm_BP / self.rho) + 0.5 * torch.sqrt((x_network - mu - norm_BP / self.rho)**2 + 4 * x_EM * norm_BP / self.rho)
            self.prior_network.fit(x + mu)
            x_network = self.prior_network.predict()
            mu += x - x_network
            self.object_prediction = nn.ReLU()(x_network)
            # evaluate callback
            if self.callback is not None:
                self._compute_callback(n_iter = _)
        return self.object_prediction