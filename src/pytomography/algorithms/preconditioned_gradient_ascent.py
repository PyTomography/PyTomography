r"""This module consists of preconditioned gradient ascent (PGA) algorithms: these algorithms are both statistical (since they depend on a likelihood function dependent on the imaging system) and iterative. Common clinical reconstruction algorithms, such as OSEM, correspond to a subclass of PGA algorithms. PGA algorithms are characterized by the update rule :math:`f^{n+1} = f^{n} + C^{n}(f^{n}) \left[\nabla_{f} L(g^n|f^{n}) - \beta \nabla_{f} V(f^{n}) \right]` where :math:`L(g^n|f^{n})` is the likelihood function, :math:`V(f^{n})` is the prior function, :math:`C^{n}(f^{n})` is the preconditioner, and :math:`\beta` is a scalar used to scale the prior function."""

from __future__ import annotations
from collections.abc import Callable, Sequence
import pytomography
import torch
from pytomography.callbacks import Callback, DataStorageCallback
from pytomography.likelihoods import Likelihood
from pytomography.priors import Prior

class PreconditionedGradientAscentAlgorithm:
    r"""Generic class for preconditioned gradient ascent algorithms: i.e. those that have the form :math:`f^{n+1} = f^{n} + C^{n}(f^{n}) \left[\nabla_{f} L(g^n|f^{n}) - \beta \nabla_{f} V(f^{n}) \right]`. 

    Args:
        likelihood (Likelihood): Likelihood class that facilitates computation of :math:`L(g^n|f^{n})` and its associated derivatives.
        prior (Prior, optional): Prior class that faciliates the computation of function :math:`V(f)` and its associated derivatives. If None, then no prior is used Defaults to None.
        object_initial (torch.Tensor | None, optional): Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
        norm_BP_subset_method (str, optional): Specifies how :math:`H^T 1` is calculated when subsets are used. If 'subset_specific', then uses :math:`H_n^T 1`. If `average_of_subsets`, then uses the average of all :math:`H_n^T 1`s for any given subset (scaled to the relative size of the subset if subsets are not equal size). Defaults to 'subset_specific'.
    """
    def __init__(
        self,
        likelihood: Likelihood,
        prior: Prior = None,
        object_initial: torch.Tensor | None = None,
        norm_BP_subset_method: str = 'subset_specific',
        **kwargs,
    ) -> None:
        self.likelihood = likelihood
        if object_initial is None:
            self.object_prediction = self.likelihood.system_matrix._get_object_initial()
        else:
            self.object_prediction = object_initial.to(pytomography.device).to(pytomography.dtype)
        self.prior = prior
        if self.prior is not None:
            self.prior.set_object_meta(self.likelihood.system_matrix.object_meta)
        self.norm_BP_subset_method = norm_BP_subset_method
        # These are if objects / FPS are stored during reconstruction for uncertainty analysis afterwards
        self.objects_stored = []
        self.projections_predicted_stored = []
                
    def _set_n_subsets(self, n_subsets: int):
        """Sets the number of subsets used in the reconstruction algorithm.

        Args:
            n_subsets (int): Number of subsets
        """
        self.n_subsets = n_subsets
        self.likelihood._set_n_subsets(n_subsets)
    
    def _compute_preconditioner(
        self,
        object: torch.Tensor,
        n_iter: int,
        n_subset: int
        ) -> None:
        r"""Computes the preconditioner factor :math:`C^{n}(f^{n})`. Must be implemented by any reconstruction algorithm that inherits from this generic class.

        Args:
            object (torch.Tensor): Object :math:`f^n`
            n_iter (int): Iteration number
            n_subset (int): Subset number

        Raises:
            NotImplementedError: .
        """
        raise NotImplementedError("_compute_preconditioner not implemented for this reconstruction algorithm; this must be implemented by any subclass of PreconditionedGradientAscentAlgorithm")
        
    def _compute_callback(self, n_iter: int, n_subset: int):
        """Method for computing callbacks after each reconstruction iteration

        Args:
            n_iter (int): Number of iterations
            n_subset (int): Number of subsets
        """
        self.callback.run(self.object_prediction, n_iter, n_subset)
    
    def __call__(
        self,
        n_iters: int,
        n_subsets: int,
        n_subset_specific: int | None = None,
        callback: Callback | None = None,
        ):
        """_summary_

        Args:
            Args:
            n_iters (int): Number of iterations
            n_subsets (int): Number of subsets
            n_subset_specific (int): Ignore all updates except for this subset.
            callback (Callback, optional): Callback function to be called after each subiteration. Defaults to None.

        Returns:
            torch.Tensor: Reconstructed object.
        """
        self.callback = callback
        self.n_iters = n_iters
        self._set_n_subsets(n_subsets)
        # Perform reconstruction loop
        for j in range(n_iters):
            for k in range(n_subsets):
                if n_subset_specific is not None:
                    if n_subset_specific!=k:
                        continue
                if bool(self.prior):
                    self.prior.set_object(torch.clone(self.object_prediction).to(pytomography.device))
                    self.prior.set_beta_scale(self.likelihood.system_matrix.get_weighting_subset(k))
                    self.prior_gradient = self.prior(derivative_order=1)
                else:
                    self.prior_gradient = 0
                likelihood_gradient = self.likelihood.compute_gradient(self.object_prediction, k, self.norm_BP_subset_method)
                preconditioner = self._compute_preconditioner(self.object_prediction, j, k)
                self.object_prediction += preconditioner * (likelihood_gradient - self.prior_gradient)
                # Get rid of small negative values
                self.object_prediction[self.object_prediction<0] = 0
                if self.callback is not None:
                    self._compute_callback(n_iter=j, n_subset=k)
        if self.callback is not None:
            self.callback.finalize(self.object_prediction)
        return self.object_prediction 
                
class LinearPreconditionedGradientAscentAlgorithm(PreconditionedGradientAscentAlgorithm):
    r"""Implementation of a special case of ``PreconditionedGradientAscentAlgorithm`` whereby :math:`C^{n}(f^n) = D^{n} f^{n}`

    Args:
        likelihood (Likelihood): Likelihood class that facilitates computation of :math:`L(g^n|f^{n})` and its associated derivatives.
        prior (Prior, optional): Prior class that faciliates the computation of function :math:`V(f)` and its associated derivatives. If None, then no prior is used Defaults to None.
        object_initial (torch.Tensor | None, optional): Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
        norm_BP_subset_method (str, optional): Specifies how :math:`H^T 1` is calculated when subsets are used. If 'subset_specific', then uses :math:`H_n^T 1`. If `average_of_subsets`, then uses the average of all :math:`H_n^T 1`s for any given subset (scaled to the relative size of the subset if subsets are not equal size). Defaults to 'subset_specific'.
    """
    def _linear_preconditioner_factor(self, n_iter: int, n_subset: int):
        r"""Implementation of object independent scaling factor :math:`D^{n}` in :math:`C^{n}(f^{n}) = D^{n} f^{n}`

        Args:
            n_iter (int): iteration number
            n_subset (int): subset number

        Raises:
            NotImplementedError: .
        """
        raise NotImplementedError("_linear_preconditioner_factor not implemented for this reconstruction algorithm; this must be implemented by any subclass of LinearPreconditionedGradientAscentAlgorithm")
        
    def _compute_preconditioner(
        self,
        object: torch.Tensor,
        n_iter: int,
        n_subset: int
        ) -> torch.Tensor:
        r"""Computes the preconditioner :math:`C^{n}(f^n) = D^{n} \text{diag}\left(f^{n}\right)` using the associated `_linear_preconditioner_factor` method.

        Args:
            object (torch.Tensor): Object :math:`f^{n}`
            n_iter (int): Iteration :math:`n`
            n_subset (int): Subset :math:`m`

        Returns:
            torch.Tensor: Preconditioner factor
        """
        return object * self._linear_preconditioner_factor(n_iter, n_subset)
    
    def compute_uncertainty(
        self,
        mask: torch.Tensor,
        data_storage_callback: DataStorageCallback,
        subiteration_number : int | None = None,
        return_pct: bool = False
        ) -> float | Sequence[float]:
        """Estimates the uncertainty of the sum of voxels in a reconstructed image. Calling this method requires a masked region `mask` as well as an instance of `DataStorageCallback` that has been used in a reconstruction algorithm: this data storage contains the estimated object and associated forward projection at each subiteration number.

        Args:
            mask (torch.Tensor): Masked region of the reconstructed object: a boolean Tensor.
            data_storage_callback (Callback): Callback that has been used in a reconstruction algorithm.
            subiteration_number (int | None, optional): Subiteration number to compute the uncertainty for. If None, then computes the uncertainty for the last iteration. Defaults to None.
            return_pct (bool, optional): If true, then additionally returns the percent uncertainty for the sum of counts. Defaults to False.

        Returns:
            float | Sequence[float]: Absolute uncertainty in the sum of counts in the masked region (if `return_pct` is False) OR absolute uncertainty and relative uncertainty in percent (if `return_pct` is True)
        """
        if subiteration_number is None:
            subiteration_number = len(data_storage_callback.objects) - 1
        V = self._compute_uncertainty_matrix(mask, data_storage_callback, subiteration_number)
        uncertainty_abs = torch.sqrt(torch.sum(V * self.likelihood.projections * V)).item()
        if not(return_pct):
            return uncertainty_abs
        else:
            uncertainty_rel = uncertainty_abs / data_storage_callback.objects[subiteration_number].to(pytomography.device)[mask].sum().item() * 100
            return uncertainty_abs, uncertainty_rel
    
    def _compute_uncertainty_matrix(
        self,
        mask: torch.Tensor,
        data_storage_callback: DataStorageCallback,
        n: int
        ) -> torch.Tensor:
        r"""Computes the quantity :math:`V^{n+1}\chi = V^{n} Q^{n} \chi + B^{n}\chi` where :math:`Q^{n} = \left[\nabla_{ff} L(g^n|f^n) -  \nabla_{ff} U(f^n)\right] D^{n} \text{diag}\left(f^{n}\right) + \text{diag}\left(f^{n+1}/f^n\right)` and :math:`B^{n}=\nabla_{gf} L(g^n|f^n) D^n \text{diag}\left(f^{n}\right)` and :math:`V^{0} = 0 . This function is meant to be called recursively.

        Args:
            mask (torch.Tensor): Masked region :math:`\chi`.
            data_storage_callback (DataStorageCallback): Callback that has been used in a reconstruction algorithm.
            n (int): Subiteration number.

        Returns:
            torch.Tensor: the quantity :math:`V^{n+1}\chi`
        """
        if n==0:
            return torch.zeros((1, *self.likelihood.system_matrix.proj_meta.shape)).to(pytomography.device)
        else:
            subset_idx = (n-1)%self.n_subsets
            object_current_update = data_storage_callback.objects[n].to(pytomography.device)
            object_previous_update = data_storage_callback.objects[n-1].to(pytomography.device)
            FP_previous_update = data_storage_callback.projections_predicted[n-1].to(pytomography.device)
            likelihood_grad_ff = self.likelihood.compute_gradient_ff(object_previous_update, FP_previous_update, subset_idx)
            # TODO Fix None argument later (required for relaxation sequence)
            term1 = mask * object_previous_update * self._linear_preconditioner_factor(None, subset_idx)
            if self.prior is not None:
                self.prior.set_beta_scale(self.likelihood.system_matrix.get_weighting_subset(subset_idx))
                self.prior.set_object(object_previous_update)
                term1 = likelihood_grad_ff(term1) - self.prior(derivative_order=2)(term1)
            else:
                term1 = likelihood_grad_ff(term1)
            term1 = - term1 + (1 - object_current_update/(object_previous_update+pytomography.delta)) * mask
            # Save memory before doing recursion
            del(object_current_update)
            del(object_previous_update)
            del(FP_previous_update)
            del(likelihood_grad_ff)
            torch.cuda.empty_cache()
            term1 = self._compute_uncertainty_matrix(1*mask-term1, data_storage_callback, n-1)
            # Additive step after recursion
            object_previous_update = data_storage_callback.objects[n-1].to(pytomography.device)
            FP_previous_update = data_storage_callback.projections_predicted[n-1].to(pytomography.device)
            likelihood_grad_gf = self.likelihood.compute_gradient_gf(object_previous_update, FP_previous_update, subset_idx)
            term2 = mask * object_previous_update * self._linear_preconditioner_factor(None, subset_idx)
            term2 = likelihood_grad_gf(term2)
            # Add to term
            subset_indices_array = self.likelihood.system_matrix.subset_indices_array[subset_idx]
            term_return = torch.zeros((1, *self.likelihood.system_matrix.proj_meta.shape)).to(pytomography.device)
            term_return += term1
            term_return[:,subset_indices_array] += term2
            # Delete to save memory
            del(term1)
            del(term2)
            del(object_previous_update)
            del(FP_previous_update)
            del(likelihood_grad_gf)
            torch.cuda.empty_cache()
            return term_return     
        
class OSEM(LinearPreconditionedGradientAscentAlgorithm):
    r"""Implementation of the ordered subset expectation maximum algorithm :math:`f^{n+1} = f^{n} + \frac{f^n}{H_n^T} \nabla_{f} L(g^n|f^{n})`.

        Args:
            likelihood (Likelihood): Likelihood function :math:`L`.
            object_initial (torch.Tensor | None, optional): Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
        """
    def __init__(
        self,
        likelihood: Likelihood,
        object_initial: torch.tensor | None = None,
        ):

        super(OSEM, self).__init__(
            likelihood = likelihood,
            object_initial = object_initial,
            )
        
    def _linear_preconditioner_factor(self, n_iter: int, n_subset: int) -> torch.Tensor:
        """Computes the linear preconditioner factor :math:`D^n = 1/H_n^T 1`

        Args:
            n_iter (int): iteration number
            n_subset (int): subset number

        Returns:
           torch.Tensor: linear preconditioner factor
        """
        return 1/(self.likelihood.norm_BPs[n_subset] + pytomography.delta)
    
class OSMAPOSL(PreconditionedGradientAscentAlgorithm):
    r"""Implementation of the ordered subset maximum a posteriori one step late algorithm :math:`f^{n+1} = f^{n} + \frac{f^n}{H_n^T+\nabla_f V(f^n)} \left[ \nabla_{f} L(g^n|f^{n}) - \nabla_f V(f^n) \right]`

        Args:
            likelihood (Likelihood): Likelihood function :math:`L`.
            object_initial (torch.Tensor | None, optional): Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
            prior (Prior, optional): Prior class that faciliates the computation of function :math:`V(f)` and its associated derivatives. If None, then no prior is used. Defaults to None.
        """
    def __init__(
        self,
        likelihood: Likelihood,
        object_initial: torch.tensor | None = None,
        prior: Prior | None = None,
    ):
        super(OSMAPOSL, self).__init__(
            likelihood = likelihood,
            object_initial = object_initial,
            prior = prior
            )
        
    def _compute_preconditioner(self, object: torch.Tensor, n_iter: int, n_subset: int) -> torch.Tensor:
        r"""Computes the preconditioner factor :math:`C^n(f^n) = \frac{f^n}{H_n^T+\nabla_f V(f^n)}`

        Args:
            object (torch.Tensor): Object estimate :math:`f^n`
            n_iter (int): iteration number
            n_subset (int): subset number

        Returns:
            torch.Tensor: preconditioner factor.
        """
        return object/(self.likelihood.norm_BPs[n_subset] + self.prior_gradient + pytomography.delta)
    
class BSREM(LinearPreconditionedGradientAscentAlgorithm):
    r"""Implementation of the block sequential regularized expectation maximum algorithm :math:`f^{n+1} = f^{n} + \frac{\alpha(n)}{\omega_n H^T 1} \left[\nabla_{f} L(g^n|f^{n}) - \nabla_f V(f^n) \right]`

        Args:
            likelihood (Likelihood): likelihood function :math:`L`
            object_initial (torch.Tensor | None, optional): Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
            prior (Prior, optional): Prior class that faciliates the computation of function :math:`V(f)` and its associated derivatives. If None, then no prior is used. Defaults to None.
            relaxation_sequence (Callable, optional): Relxation sequence :math:`\alpha(n)` used to scale future updates. Defaults to 1 for all :math:`n`. Note that when this function is provided, it takes the iteration number (not the subiteration) so that e.g. if 4 iterations and 8 subsets are used, it would call :math:`\alpha(4)` for all 8 subiterations of the final iteration.
        """
    def __init__(
        self,
        likelihood: Likelihood,
        object_initial: torch.tensor | None = None,
        prior: Prior | None = None,
        relaxation_sequence: Callable = lambda _: 1,
    ):
        self.relaxation_sequence = relaxation_sequence
        super(BSREM, self).__init__(
            likelihood = likelihood,
            object_initial = object_initial,
            prior = prior,
            norm_BP_subset_method = 'average_of_subsets',
            )
    def _linear_preconditioner_factor(self, n_iter: int, n_subset: int):
        """Computes the linear preconditioner factor :math:`D^n = 1/(\omega_n H^T 1)` where :math:`\omega_n` corresponds to the fraction of subsets at subiteration :math:`n`. 

        Args:
            n_iter (int): iteration number
            n_subset (int): subset number

        Returns:
           torch.Tensor: linear preconditioner factor
        """
        relaxation_factor = self.relaxation_sequence(n_iter)
        norm_BP = torch.stack(self.likelihood.norm_BPs).sum(axis=0)
        norm_BP_weight = self.likelihood.system_matrix.get_weighting_subset(n_subset)
        return relaxation_factor/(norm_BP_weight * norm_BP + pytomography.delta)
    
class KEM(OSEM):
    r"""Implementation of the ordered subset expectation maximum algorithm :math:`\alpha^{n+1} = \alpha^{n} + \frac{\alpha^n}{\tilde{H}_n^T} \nabla_{f} L(g^n|\alpha^{n})` and where the final predicted object is :math:`f^n = K \hat{\alpha}^{n}`. The system matrix :math:`\tilde{H}` includes the kernel transform :math:`K`.

        Args:
            likelihood (Likelihood): Likelihood function :math:`L`.
            object_initial (torch.Tensor | None, optional): Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
        """
    def _compute_callback(self, n_iter: int, n_subset: int):
        r"""Method for computing callbacks after each reconstruction iteration. This is reimplemented for KEM because the callback needs to be called on :math:`f^n = K \hat{\alpha}^{n}` as opposed to :math:`\hat{\alpha}^{n}`
        
        Args:
            n_iter (int): Number of iterations
            n_subset (int): Number of subsets
        """
        self.callback.run(self.likelihood.system_matrix.kem_transform.forward(self.object_prediction), n_iter, n_subset)
    def __call__(
        self, *args, **kwargs
    ):
        r"""Reimplementation of the call method such that :math:`f^n = K \hat{\alpha}^{n}` is returned as opposed to :math:`\hat{\alpha}^{n}`

        Returns:
            torch.Tensor: reconstructed object
        """
        object_prediction = super(KEM, self).__call__(*args, **kwargs)
        return self.likelihood.system_matrix.kem_transform.forward(object_prediction)