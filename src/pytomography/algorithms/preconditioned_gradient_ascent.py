r"""This module consists of preconditioned gradient ascent (PGA) algorithms: these algorithms are both statistical (since they depend on a likelihood function dependent on the imaging system) and iterative. Common clinical reconstruction algorithms, such as OSEM, correspond to a subclass of PGA algorithms. PGA algorithms are characterized by the update rule :math:`f^{n+1} = f^{n} + C^{n}(f^{n}) \left[\nabla_{f} L(g^n|f^{n}) - \beta \nabla_{f} V(f^{n}) \right]` where :math:`L(g^n|f^{n})` is the likelihood function, :math:`V(f^{n})` is the prior function, :math:`C^{n}(f^{n})` is the preconditioner, and :math:`\beta` is a scalar used to scale the prior function."""

from __future__ import annotations
from collections.abc import Callable, Sequence
import pytomography
import torch
from pytomography.callbacks import Callback, DataStorageCallback
from pytomography.likelihoods import Likelihood, SARTWeightedNegativeMSELikelihood
from pytomography.priors import Prior
from pytomography.io.SPECT import dicom
from pytomography.projectors import SystemMatrix

class PreconditionedGradientAscentAlgorithm:
    r"""Generic class for preconditioned gradient ascent algorithms: i.e. those that have the form :math:`f^{n+1} = f^{n} + C^{n}(f^{n}) \left[\nabla_{f} L(g^n|f^{n}) - \beta \nabla_{f} V(f^{n}) \right]`. 

    Args:
        likelihood (Likelihood): Likelihood class that facilitates computation of :math:`L(g^n|f^{n})` and its associated derivatives.
        prior (Prior, optional): Prior class that faciliates the computation of function :math:`V(f)` and its associated derivatives. If None, then no prior is used Defaults to None.
        object_initial (torch.Tensor | None, optional): Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
        addition_after_iteration (float, optional): Value to add to the object after each iteration. This prevents image voxels getting "locked" at values of 0 for certain algorithms. Defaults to 0.
    """
    def __init__(
        self,
        likelihood: Likelihood,
        prior: Prior = None,
        object_initial: torch.Tensor | None = None,
        addition_after_iteration: float = 0,
        **kwargs,
    ) -> None:
        self.likelihood = likelihood
        if object_initial is None:
            self.object_prediction = self.likelihood.system_matrix._get_object_initial(pytomography.device)
        else:
            self.object_prediction = object_initial.to(pytomography.device).to(pytomography.dtype)
        self.prior = prior
        if self.prior is not None:
            self.prior.set_object_meta(self.likelihood.system_matrix.object_meta)
            self.prior.set_FOV_scale(self.likelihood.system_matrix._get_prior_FOV_scale())
        # These are if objects / FPS are stored during reconstruction for uncertainty analysis afterwards
        self.objects_stored = []
        self.projections_predicted_stored = []
        self.addition_after_iteration = addition_after_iteration
                
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
        self.object_prediction = self.callback.run(self.object_prediction, n_iter, n_subset)
    
    def __call__(
        self,
        n_iters: int,
        n_subsets: int = 1,
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
                if n_subsets==1:
                    subset_idx = None
                else:
                    subset_idx = k
                # Adjust object before iteration: note, because of uncertainty analysis this must be at beginning
                if bool(self.prior):
                    self.prior.set_object(torch.clone(self.object_prediction).to(pytomography.device))
                    self.prior.set_beta_scale(self.likelihood.system_matrix.get_weighting_subset(subset_idx))
                    self.prior_gradient = self.prior(derivative_order=1)
                else:
                    self.prior_gradient = 0
                likelihood_gradient = self.likelihood.compute_gradient(self.object_prediction, subset_idx)
                preconditioner = self._compute_preconditioner(self.object_prediction, j, subset_idx)
                self.object_prediction += preconditioner * (likelihood_gradient - self.prior_gradient)
                self.object_prediction[self.object_prediction<=self.addition_after_iteration] = self.addition_after_iteration
                if self.callback is not None:
                    self._compute_callback(n_iter=j, n_subset=k)
        # Remove the addition after the last iteration
        #self.object_prediction -= self.addition_after_iteration
        if self.callback is not None:
            self.callback.finalize(self.object_prediction)
        return self.object_prediction 
                
class LinearPreconditionedGradientAscentAlgorithm(PreconditionedGradientAscentAlgorithm):
    r"""Implementation of a special case of ``PreconditionedGradientAscentAlgorithm`` whereby :math:`C^{n}(f^n) = D^{n} f^{n}`

    Args:
        likelihood (Likelihood): Likelihood class that facilitates computation of :math:`L(g^n|f^{n})` and its associated derivatives.
        prior (Prior, optional): Prior class that faciliates the computation of function :math:`V(f)` and its associated derivatives. If None, then no prior is used Defaults to None.
        object_initial (torch.Tensor | None, optional): Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
        addition_after_iteration (float, optional): Value to add to the object after each iteration. This prevents image voxels getting "locked" at values of 0 for certain algorithms. Defaults to 0.
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
        return_pct: bool = False,
        include_additive_term: bool = False,
        post_recon_filter: Transform | None = None
        ) -> float | Sequence[float]:
        """Estimates the uncertainty of the sum of voxels in a reconstructed image. Calling this method requires a masked region `mask` as well as an instance of `DataStorageCallback` that has been used in a reconstruction algorithm: this data storage contains the estimated object and associated forward projection at each subiteration number.

        Args:
            mask (torch.Tensor): Masked region of the reconstructed object: a boolean Tensor.
            data_storage_callback (Callback): Callback that has been used in a reconstruction algorithm.
            subiteration_number (int | None, optional): Subiteration number to compute the uncertainty for. If None, then computes the uncertainty for the last iteration. Defaults to None.
            return_pct (bool, optional): If true, then additionally returns the percent uncertainty for the sum of counts. Defaults to False.
            include_additive_term (bool): Whether or not to include uncertainty contribution from the additive term. This requires the ``additive_term_variance_estimate`` as an argument to the initialized likelihood. Defaults to False.

        Returns:
            float | Sequence[float]: Absolute uncertainty in the sum of counts in the masked region (if `return_pct` is False) OR absolute uncertainty and relative uncertainty in percent (if `return_pct` is True)
        """
        if subiteration_number is None:
            subiteration_number = len(data_storage_callback.objects) - 1
        # Get final reconstruciton
        final_recon = data_storage_callback.objects[subiteration_number].to(pytomography.device)
        # Apply filter if provided
        if post_recon_filter is not None:
            Q_sequence_current = post_recon_filter(mask)
            final_recon = post_recon_filter(final_recon)
        else:
            Q_sequence_current = mask.clone()
        V = 0
        for n in range(subiteration_number, 0, -1):
            V += self._compute_B(Q_sequence_current, data_storage_callback, n-1, include_additive_term=include_additive_term)
            if n>1:
                Q_sequence_current = self._compute_Q(Q_sequence_current, data_storage_callback, n-1)
        uncertainty_abs2 = torch.sum(V[0] * self.likelihood.projections * V[0])
        # If uncertainty estimated in the additive term
        if include_additive_term:
            uncertainty_abs2 += torch.sum(self.likelihood.additive_term_variance_estimate(V[1]))
        uncertainty_abs = torch.sqrt(uncertainty_abs2).item()
        if not(return_pct):
            return uncertainty_abs
        else:
            uncertainty_rel = uncertainty_abs / (final_recon*mask).sum().item() * 100
            return uncertainty_abs, uncertainty_rel
        
    def _compute_Q(
        self,
        input: torch.Tensor,
        data_storage_callback: Callback,
        n: int,
    ) -> torch.Tensor:
        """Computes the operation of :math:`Q` on an input object; this is a helper function for ``compute_uncertainty``. For more details, see the uncertainty paper.

        Args:
            input (torch.Tensor): Object on which Q operates
            data_storage_callback (Callback): Data storage callback containing all objects and forward projections at each subiteration
            n (int): Subiteration number

        Returns:
            torch.Tensor: Resulting output object from the operation of :math:`Q` on the input object
        """ 
        if self.n_subsets==1:
            subset_idx = None
        else:
            subset_idx = n%self.n_subsets
        object_current_update = data_storage_callback.objects[n].to(pytomography.device)
        object_future_update = data_storage_callback.objects[n+1].to(pytomography.device)
        FP_current_update = data_storage_callback.projections_predicted[n].to(pytomography.device)
        likelihood_grad_ff = self.likelihood.compute_gradient_ff(object_current_update, FP_current_update, subset_idx)
        # TODO Fix None argument later (required for relaxation sequence)
        output = input * object_current_update * self._linear_preconditioner_factor(None, subset_idx)
        if self.prior is not None:
            self.prior.set_beta_scale(self.likelihood.system_matrix.get_weighting_subset(subset_idx))
            self.prior.set_object(object_current_update)
            output = likelihood_grad_ff(output) - self.prior(derivative_order=2)(output)
        else:
            output = likelihood_grad_ff(output)
        output += (object_future_update / (object_current_update+pytomography.delta)) * input
        return output
    
    def _compute_B(
        self,
        input: torch.Tensor,
        data_storage_callback: Callback,
        n: int,
        include_additive_term: bool = False
    ) -> torch.Tensor:
        """Computes the operation of :math:`B` on an input object; this is a helper function for ``compute_uncertainty``. For more details, see the uncertainty paper.

        Args:
            input (torch.Tensor): Object on which B operates
            data_storage_callback (Callback): Data storage callback containing all objects and forward projections at each subiteration
            n (int): Subiteration number
            include_additive_term (bool, optional): Whether or not to include uncertainty estimation for the additive term. Defaults to False.

        Returns:
            torch.Tensor: Resulting output projections from the operation of :math:`B` on the input object
        """
        if self.n_subsets==1:
            subset_idx = None
            subset_indices_array = torch.arange(self.likelihood.system_matrix.proj_meta.shape[0]).to(torch.long).to(pytomography.device)
        else:
            subset_idx = n%self.n_subsets
            subset_indices_array = self.likelihood.system_matrix.subset_indices_array[subset_idx]
        object_current_update = data_storage_callback.objects[n].to(pytomography.device)
        FP_current_update = data_storage_callback.projections_predicted[n].to(pytomography.device)
        output = input * object_current_update * self._linear_preconditioner_factor(None, subset_idx)
        output_primary = self.likelihood.compute_gradient_gf(object_current_update, FP_current_update, subset_idx)(output)
        
        if include_additive_term:
            output_additive = self.likelihood.compute_gradient_sf(object_current_update, FP_current_update, subset_idx)(output)
            # This is weird because sometimes dual peak reconstruction is used and this dimenion needs to be obtained from the current FP update if it exists
            output_total = torch.zeros((
                2,
                *FP_current_update.shape[:-3],
                *self.likelihood.system_matrix.proj_meta.shape[-3:]
            )).to(pytomography.device)
            output_total[0,...,subset_indices_array,:,:] = output_primary
            output_total[1,...,subset_indices_array,:,:] = output_additive
            return output_total
        else:
            output_total = torch.zeros((
                1,
                *FP_current_update.shape[:-3],
                *self.likelihood.system_matrix.proj_meta.shape[-3:]
            )).to(pytomography.device)
            output_total[0,...,subset_indices_array,:,:] = output_primary
        return output_total
        
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
        return 1/(self.likelihood._get_normBP(n_subset) + pytomography.delta)
    
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
        return object/(self.likelihood._get_normBP(n_subset) + self.prior_gradient + pytomography.delta)
    
class RBIEM(LinearPreconditionedGradientAscentAlgorithm):
    r"""Implementation of the rescaled block iterative expectation maximum algorithm

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
        super(RBIEM, self).__init__(
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
        norm_BP = self.likelihood._get_normBP(n_subset)
        norm_BP_allsubsets = self.likelihood._get_normBP(n_subset, return_sum=True)
        rm = torch.max(norm_BP / (norm_BP_allsubsets + pytomography.delta))
        return object/(norm_BP_allsubsets*rm + pytomography.delta)
    
class RBIMAP(PreconditionedGradientAscentAlgorithm):
    r"""Implementation of the rescaled block iterative maximum a posteriori algorithm

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
        super(RBIMAP, self).__init__(
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
        norm_BP = self.likelihood._get_normBP(n_subset)
        norm_BP_allsubsets = self.likelihood._get_normBP(n_subset, return_sum=True)
        rm = torch.max((norm_BP + self.prior_gradient) / (norm_BP_allsubsets + self.prior_gradient + pytomography.delta))
        return object/(norm_BP_allsubsets*rm + self.prior_gradient + pytomography.delta)
    
class BSREM(LinearPreconditionedGradientAscentAlgorithm):
    r"""Implementation of the block sequential regularized expectation maximum algorithm :math:`f^{n+1} = f^{n} + \frac{\alpha(n)}{\omega_n H^T 1} \left[\nabla_{f} L(g^n|f^{n}) - \nabla_f V(f^n) \right]`

        Args:
            likelihood (Likelihood): likelihood function :math:`L`
            object_initial (torch.Tensor | None, optional): Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
            prior (Prior, optional): Prior class that faciliates the computation of function :math:`V(f)` and its associated derivatives. If None, then no prior is used. Defaults to None.
            relaxation_sequence (Callable, optional): Relxation sequence :math:`\alpha(n)` used to scale future updates. Defaults to 1 for all :math:`n`. Note that when this function is provided, it takes the iteration number (not the subiteration) so that e.g. if 4 iterations and 8 subsets are used, it would call :math:`\alpha(4)` for all 8 subiterations of the final iteration.
            addition_after_iteration (float, optional): Value to add to the object after each iteration. This prevents image voxels getting "locked" at values of 0. Defaults to 1e-4.
        """
    def __init__(
        self,
        likelihood: Likelihood,
        object_initial: torch.tensor | None = None,
        prior: Prior | None = None,
        relaxation_sequence: Callable = lambda _: 1,
        addition_after_iteration = 1e-4, # good for typical counts in Lu177 SPECT
    ):
        self.relaxation_sequence = relaxation_sequence
        super(BSREM, self).__init__(
            likelihood = likelihood,
            object_initial = object_initial,
            prior = prior,
            addition_after_iteration = addition_after_iteration
            )
    def _linear_preconditioner_factor(self, n_iter: int, n_subset: int):
        r"""Computes the linear preconditioner factor :math:`D^n = 1/(\omega_n H^T 1)` where :math:`\omega_n` corresponds to the fraction of subsets at subiteration :math:`n`. 

        Args:
            n_iter (int): iteration number
            n_subset (int): subset number

        Returns:
           torch.Tensor: linear preconditioner factor
        """
        relaxation_factor = self.relaxation_sequence(n_iter)
        norm_BP = self.likelihood._get_normBP(n_subset, return_sum=True)
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
    
class MLEM(OSEM):
    r"""Implementation of the maximum likelihood expectation maximum algorithm :math:`f^{n+1} = f^{n} + \frac{f^n}{H^T} \nabla_{f} L(g|f^{n})`.

        Args:
            likelihood (Likelihood): Likelihood function :math:`L`.
            object_initial (torch.Tensor | None, optional): Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
        """
    def __call__(self, n_iters, callback=None):
        return super(MLEM, self).__call__(n_iters, n_subsets=1, callback=callback)

class SART(PreconditionedGradientAscentAlgorithm):
    r"""Implementation of the SART algorithm. This algorithm takes as input the system matrix and projections (as opposed to a likelihood). This is an implementation of equation 3 of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8506772/

        Args:
            system_matrix (SystemMatrix): System matrix for the imaging system.
            projections (torch.Tensor): Projections for the imaging system.
            additive_term (torch.Tensor | None, optional): Additive term for the imaging system. If None, then no additive term is used. Defaults to None.
            object_initial (torch.Tensor | None, optional): Initial object for reconstruction algorithm. If None, then an object with 1 in every voxel is used. Defaults to None.
        """
    def __init__(
        self,
        system_matrix: SystemMatrix,
        projections: torch.Tensor,
        additive_term: torch.Tensor | None = None,
        object_initial: torch.tensor | None = None,
        relaxation_sequence: Callable = lambda _: 1,
        ):
        likelihood = SARTWeightedNegativeMSELikelihood(system_matrix, projections, additive_term=additive_term)
        super().__init__(
            likelihood = likelihood,
            object_initial = object_initial,
            )
        self.relaxation_sequence = relaxation_sequence
        
    def _compute_preconditioner(self, object: torch.Tensor, n_iter: int, n_subset: int) -> torch.Tensor:
        r"""Computes the preconditioner factor :math:`C^n(f^n) = \frac{1}{H_n^T+\nabla_f V(f^n)}`

        Args:
            object (torch.Tensor): Object estimate :math:`f^n`
            n_iter (int): iteration number
            n_subset (int): subset number

        Returns:
            torch.Tensor: preconditioner factor.
        """
        relaxation_factor = self.relaxation_sequence(n_iter)
        return relaxation_factor/(self.likelihood._get_normBP(n_subset) + pytomography.delta)
    
class PGAAMultiBedSPECT(PreconditionedGradientAscentAlgorithm):
    """Assistant class for performing reconstruction on multi-bed SPECT data. This class is a wrapper around a reconstruction algorithm that is called for each bed and then the results are stitched together.

        Args:
            files_NM (Sequence[str]): Sequence of SPECT raw data paths corresponding to each likelihood
            reconstruction_algorithm (Algorithm): Reconstruction algorithm used for reconstruction of each bed position
        """
    def __init__(
        self,
        files_NM: Sequence[str],
        reconstruction_algorithms: Sequence[object],
        ) -> None:
        self.files_NM = files_NM
        self.reconstruction_algorithms = reconstruction_algorithms
        
    def __call__(
        self,
        n_iters: int,
        n_subsets: int,
        callback: Callback | Sequence[Callback] | None = None,
        ) -> torch.Tensor:
        """Perform reconstruction of each bed position for specified iteraitons and subsets, and return the stitched image

        Args:
            n_iters (int): Number of iterations to perform reconstruction for.
            n_subsets (int): Number of subsets to perform reconstruction for.
            callback (Callback | Sequence[Callback] | None, optional): Callback function. If a single Callback is given, then the callback is computed for the stitched image. If a sequence of callbacks is given, then it must be the same length as the number of bed positions; each callback is called on the reconstruction for each bed position. If None, no Callback is used. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        self.callback = callback
        for i in range(n_iters):
            for j in range(n_subsets):
                self.recons = []
                for recon_algo in self.reconstruction_algorithms:
                    self.recons.append(recon_algo(1, n_subsets, n_subset_specific=j))
                self.object_prediction = dicom.stitch_multibed(
                    recons=torch.stack(self.recons),
                    files_NM = self.files_NM
                )
                self._compute_callback(i,j)
        self._finalize_callback()
        return self.object_prediction
    
    def _compute_callback(self, n_iter: int, n_subset: int):
        """Computes the callback at iteration ``n_iter`` and subset ``n_subset``.

        Args:
            n_iter (int): Iteration number
            n_subset (int): Subset index
        """
        if self.callback is not None:
            if type(self.callback) is list:
                for recon_algo_k, callback_k in zip(self.reconstruction_algorithms, self.callback):
                    recon_algo_k.callback = callback_k
                    recon_algo_k._compute_callback(n_iter=n_iter, n_subset=n_subset)
            else:
                self.object_prediction = dicom.stitch_multibed(
                    recons=torch.stack(self.recons),
                    files_NM = self.files_NM)  
                super()._compute_callback(n_iter=n_iter, n_subset=n_subset)
    
    def _finalize_callback(self):
        """Finalizes callbacks after reconstruction. This method is called after the reconstruction algorithm has finished.
        """
        if self.callback is not None:
            if type(self.callback) is list:
                for recon_algo_k, callback_k in zip(self.reconstruction_algorithms, self.callback):
                    recon_algo_k.callback = callback_k
                    recon_algo_k.callback.finalize(recon_algo_k.object_prediction)
            else:
                self.callback.finalize(self.object_prediction)
    
    def compute_uncertainty(
        self,
        mask: torch.Tensor,
        data_storage_callbacks: Sequence[Callback],
        subiteration_number: int | None = None,
        return_pct: bool = False,
        include_additive_term: bool = False
    ):
        """Estimates the uncertainty in a mask (should be same shape as the stitched image). Calling this method requires a sequence of ``DataStorageCallback`` instances that have been used in a reconstruction algorithm: these data storage contain required information for each bed position.

        Args:
            mask (torch.Tensor): Masked region of the reconstructed object: a boolean Tensor. This mask should be the same shape as the stitched object.
            data_storage_callbacks (Sequence[Callback]): Sequence of data storage callbacks used in reconstruction corresponding to each bed position.
            subiteration_number (int | None, optional): Subiteration number to compute the uncertainty for. If None, then computes the uncertainty for the last iteration. Defaults to None. 
            return_pct (bool, optional): If true, then additionally returns the percent uncertainty for the sum of counts. Defaults to False.
            include_additive_term (bool): Whether or not to include uncertainty contribution from the additive term. This requires the ``additive_term_variance_estimate`` as an argument to the initialized likelihood. Defaults to False.

        Returns:
            float | Sequence[float]: Absolute uncertainty in the sum of counts in the masked region (if `return_pct` is False) OR absolute uncertainty and relative uncertainty in percent (if `return_pct` is True)
        """
        if subiteration_number is None:
            subiteration_number = len(data_storage_callbacks[0].objects) - 1
        # Crop mask to FOV region
        recons = [data_storage_callback.objects[subiteration_number].to(pytomography.device) for data_storage_callback in data_storage_callbacks]
        stitching_weights, zs = dicom.stitch_multibed(torch.stack(recons), self.files_NM, return_stitching_weights=True)
        uncertainty_abs = []
        total_counts = 0
        for k in range(len(recons)):
            mask_k = mask[:,:,zs[k]:zs[k]+recons[0].shape[-1]] * stitching_weights[k]
            if mask_k.sum()==0:
                continue
            uncertainty_abs_k = self.reconstruction_algorithms[k].compute_uncertainty(mask_k, data_storage_callbacks[k], subiteration_number, return_pct=False, include_additive_term=include_additive_term)
            total_counts += (recons[k]*mask_k).sum().item()
            uncertainty_abs.append(uncertainty_abs_k)
        uncertainty_abs_total = torch.sqrt(torch.sum(torch.tensor(uncertainty_abs)**2)).item()
        if return_pct:
            uncertainty_pct = uncertainty_abs_total / total_counts * 100
            return uncertainty_abs_total, uncertainty_pct
        else:
            return uncertainty_abs_total