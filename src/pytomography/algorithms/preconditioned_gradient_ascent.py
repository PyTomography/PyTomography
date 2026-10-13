r"""Preconditioned gradient ascent (PGA) algorithms: OSEM, MLEM, OSMAPOSL, BSREM, RBI-EM, RBI-MAP, KEM and SART.

These algorithms are statistical (they maximise a likelihood of the data) and iterative. Each subiteration updates the
object with the gradient of one subset :math:`m` of the data,

.. math::

    f^{n+1} = f^{n} + C^{n}(f^{n}) \left[\nabla_{f} L(g_m|f^{n}) - \beta \nabla_{f} V(f^{n}) \right],

where :math:`L` is the likelihood, :math:`V` the prior, :math:`\beta` its weight and :math:`C^{n}` the preconditioner,
which is what distinguishes the algorithms. Call an algorithm with the number of iterations and subsets to
reconstruct; :meth:`~PreconditionedGradientAscentAlgorithm.estimate_memory` and
:meth:`~PreconditionedGradientAscentAlgorithm.plan_memory` say beforehand what it will need.
"""

from __future__ import annotations
from collections.abc import Callable, Sequence
import math
import pytomography
import torch
from pytomography.callbacks import Callback, DataStorageCallback
from pytomography.likelihoods import Likelihood, SARTWeightedNegativeMSELikelihood
from pytomography.priors import Prior
from pytomography.io.SPECT.dicom import stitch_beds
from pytomography.projectors import SystemMatrix
from pytomography.projectors.system_matrix import held_parts, on_device
from pytomography.utils.memory import MemoryEstimate, MemoryPart, MemoryPlan, memory_estimate, plan_settings, tensor_bytes

class PreconditionedGradientAscentAlgorithm:
    r"""The base class of the preconditioned gradient ascent algorithms, whose subclasses define the preconditioner.

    Each subiteration does :math:`f^{n+1} = f^{n} + C^{n}(f^{n}) \left[\nabla_{f} L(g_m|f^{n}) - \beta \nabla_{f}
    V(f^{n}) \right]` for one subset :math:`m`. Calling the algorithm reconstructs. Before running one,
    :meth:`estimate_memory` predicts its peak RAM and GPU memory for given settings (e.g. the number of subsets), and
    :meth:`plan_memory` chooses the fastest settings that fit the memory there is.

    Args:
        likelihood (Likelihood): The likelihood :math:`L`, with the data and the system matrix.
        prior (Prior, optional): The prior :math:`V` (with its weight :math:`\beta`). Defaults to None (no prior).
        object_initial (torch.Tensor | None, optional): The initial object, [Lx, Ly, Lz]. Defaults to None: the
            system matrix's initial object (ones; for SPECT, zero outside the camera's path when the system matrix asks
            for it).
        addition_after_iteration (float, optional): A floor for the object: after every update, voxels at or below
            this value are set to it, so that voxels cannot get "locked" at 0. Defaults to 0.
        **kwargs: Accepted and not used.
    """
    #: Whether the algorithm adds up the :math:`H_m^T 1` of every subset at each update (BSREM, RBI-EM, RBI-MAP), which
    #: stacks them: for memory estimates.
    _sums_normalisation = False

    #: Settings the algorithm always uses, whatever its call is given (MLEM: one subset): for memory estimates and plans.
    _memory_fixed: dict = {}
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

    # ------------------------------------------------------------------ memory
    def _memory_held(self) -> dict:
        """The arrays this reconstruction holds for its whole run that its system matrix's estimate leaves to the caller:
        the likelihood's projections (the data) and additive term (tensors or lazy sinograms)."""
        held = {}
        projections = getattr(self.likelihood, 'projections', None)
        if projections is not None:
            held['projections (the data)'] = projections
        additive_term = getattr(self.likelihood, 'additive_term', None)
        if additive_term is not None:
            held['additive term'] = additive_term
        return held

    def _memory_extra(self, n_subsets: int) -> list[MemoryPart]:
        r"""The algorithm's own arrays, for ``n_subsets`` subsets: the prior's (its copy of the image, its field of view
        scale and its gradient for the whole run; the tensors of its neighbour weight, such as an anatomical image; and
        six images of temporaries while its gradient is computed, rounded up from the sums over neighbours) and, for an
        algorithm that adds up the :math:`H_m^T 1` of every subset at each update, those images stacked and their sum."""
        image = 4 * math.prod(self.likelihood.system_matrix.object_meta.shape)
        device = pytomography.device
        parts = []
        if self.prior is not None:
            parts.append(MemoryPart('prior: its copy of the image, field of view scale and gradient', **on_device(device, 3 * image)))
            ram, gpu = tensor_bytes(getattr(self.prior, 'weight', None))
            if ram or gpu:
                parts.append(MemoryPart('prior: its neighbour weights', ram, gpu))
            parts.append(MemoryPart('prior: temporaries of its gradient', **on_device(device, 6 * image), scope='subset'))
        if self._sums_normalisation and n_subsets > 1:
            parts.append(MemoryPart(f'H_m^T 1 of the {n_subsets} subsets stacked, and their sum', **on_device(device, (n_subsets + 1) * image), scope='subset'))
        return parts

    def _memory_title(self) -> str:
        return f"{type(self).__name__} ({type(self.likelihood.system_matrix).__name__})"

    def estimate_memory(self, n_subsets: int | None = None, N_splits: int | None = None) -> MemoryEstimate:
        """The predicted peak RAM and GPU memory of this reconstruction, without running it.

        The estimate is for ``n_subsets`` subsets (the number its call will be given) and ``N_splits`` projector
        splits: the estimate of its system matrix (:meth:`~pytomography.projectors.SystemMatrix.estimate_memory`) with
        the arrays the algorithm holds, which are the likelihood's projections and additive term, the prior's arrays,
        and the stacked :math:`H_m^T 1` of BSREM and RBI. Callbacks are not counted (a ``DataStorageCallback`` keeps an
        image and the expected projections of every subiteration). ``print`` it, or ``to_dict()`` for JSON. For the
        opposite question, see :meth:`plan_memory`.

        Args:
            n_subsets (int | None, optional): Number of subsets. Defaults to None: those of the last call, or 1.
            N_splits (int | None, optional): Projector splits, for system matrices that take them. Defaults to None:
                the system matrix's own.

        Returns:
            MemoryEstimate: The estimate.

        Example:
            >>> print(OSEM(likelihood).estimate_memory(n_subsets=8))
        """
        system_matrix = self.likelihood.system_matrix
        settings = {name: present for name, (present, _) in system_matrix._memory_settings().items()}
        settings['n_subsets'] = self._memory_fixed.get('n_subsets', n_subsets if n_subsets is not None else getattr(self, 'n_subsets', 1))
        if N_splits is not None and 'N_splits' in settings:
            settings['N_splits'] = N_splits
        return system_matrix._memory_estimate(settings, held_parts(self._memory_held()), extra=self._memory_extra,
                                              title=self._memory_title(), title_settings={'N_splits': N_splits})

    def plan_memory(self, ram_gb: float | None = None, gpu_gb: float | None = None, **fixed) -> MemoryPlan:
        """The fastest settings of this reconstruction that fit in the given RAM and GPU memory.

        The result is a :class:`~pytomography.utils.memory.MemoryPlan`: as
        :meth:`~pytomography.projectors.SystemMatrix.plan_memory` of its system matrix (the fewest projector splits,
        then the fewest subsets that fit), with the arrays the algorithm holds counted as in :meth:`estimate_memory`.
        Settings passed by name are kept, e.g. ``n_subsets=8`` (the number its call will be given). Without budgets,
        the plan is for what is free on this machine now. If nothing fits, the plan has ``fits`` False and the
        settings that come closest; it does not raise.

        Args:
            ram_gb (float | None, optional): RAM to fit in, in GB (on Windows it covers GPU memory too). Defaults to
                None: what is available now.
            gpu_gb (float | None, optional): GPU memory to fit in, in GB. Defaults to None: what is available now.
            **fixed: Settings to keep, by name: ``n_subsets``, and ``N_splits`` for system matrices that take it.

        Returns:
            MemoryPlan: The plan.

        Example:
            >>> algorithm = OSEM(likelihood)
            >>> plan = algorithm.plan_memory(ram_gb=25)
            >>> recon = algorithm(n_iters=4, n_subsets=plan.settings['n_subsets'])
        """
        system_matrix = self.likelihood.system_matrix
        held = held_parts(self._memory_held())
        title = self._memory_title()
        return plan_settings(title, system_matrix._memory_settings(),
                             lambda s, alternatives: system_matrix._memory_estimate(s, held, alternatives, extra=self._memory_extra, title=title),
                             {**fixed, **self._memory_fixed}, ram_gb, gpu_gb)

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
        """Reconstructs: ``n_iters`` iterations of ``n_subsets`` subiterations each, starting from the current object.

        Calling the algorithm again continues from where the last call stopped.

        Args:
            n_iters (int): Number of iterations.
            n_subsets (int, optional): Number of subsets. Defaults to 1.
            n_subset_specific (int | None, optional): Do only the subiterations of this subset, skipping the others
                (used by algorithms that drive this one, such as :class:`~pytomography.algorithms.DIPRecon`). Defaults to
                None: every subset.
            callback (Callback, optional): Run after each subiteration, e.g. a
                :class:`~pytomography.callbacks.DataStorageCallback` for uncertainty estimates. Defaults to None.

        Returns:
            torch.Tensor: The reconstructed object, [Lx, Ly, Lz], on ``pytomography.device``.
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
        # Release the GPU memory PyTorch keeps cached from the reconstruction: on Windows it also counts as the
        # process's host memory (10 GB after a one-subset list mode OSEM of the GATE mMR's 50.8M events)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.object_prediction

class LinearPreconditionedGradientAscentAlgorithm(PreconditionedGradientAscentAlgorithm):
    r"""The algorithms whose preconditioner is linear in the object, :math:`C^{n}(f^n) = D^{n} f^{n}`.

    :math:`D^n` does not depend on the object (OSEM, BSREM, RBI-EM). For them, :meth:`compute_uncertainty` estimates
    the uncertainty of the counts in a region from a reconstruction run with a
    :class:`~pytomography.callbacks.DataStorageCallback`. The arguments are those of
    :class:`PreconditionedGradientAscentAlgorithm`.
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
        """Estimates the uncertainty (standard deviation) of the sum of the reconstructed image over a region.

        The reconstruction must have been run with a :class:`~pytomography.callbacks.DataStorageCallback`, which keeps
        the object and its expected projections at every subiteration: the uncertainty is propagated from the Poisson
        noise of the data through every update.

        Args:
            mask (torch.Tensor): The region, a boolean tensor [Lx, Ly, Lz] on the object grid.
            data_storage_callback (DataStorageCallback): The callback the reconstruction was run with.
            subiteration_number (int | None, optional): The subiteration whose image is used. Defaults to None: the
                last.
            return_pct (bool, optional): Also return the uncertainty in percent of the sum. Defaults to False.
            include_additive_term (bool, optional): Include the uncertainty of the additive term; this needs the
                likelihood's ``additive_term_variance_estimate``. Defaults to False.
            post_recon_filter (Transform | None, optional): A filter applied to the reconstruction (and the region)
                after it, e.g. a :class:`~pytomography.transforms.shared.GaussianFilter`. Defaults to None.

        Returns:
            float | tuple[float, float]: The uncertainty of the sum, in the image's units, and, with ``return_pct``,
            the uncertainty in percent of the sum.
        """
        V, final_recon = self._compute_V(mask, data_storage_callback, subiteration_number, include_additive_term, post_recon_filter)
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

    def _compute_V(
        self,
        mask: torch.Tensor,
        data_storage_callback: DataStorageCallback,
        subiteration_number: int | None = None,
        include_additive_term: bool = False,
        post_recon_filter: Transform | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Computes the derivative of the counts in ``mask`` at subiteration ``subiteration_number`` with respect to the projections, :math:`[V_y^k]'\xi` in the uncertainty paper (and with respect to the additive term, :math:`[V_s^k]'\xi`, if ``include_additive_term``); this is a helper function for ``compute_uncertainty``. The variance of the counts is then :math:`y\cdot([V_y^k]'\xi)^2` (plus the additive term's contribution).

        Args:
            mask (torch.Tensor): Masked region of the reconstructed object.
            data_storage_callback (Callback): Callback that has been used in a reconstruction algorithm.
            subiteration_number (int | None, optional): Subiteration number. If None, then the last subiteration. Defaults to None.
            include_additive_term (bool, optional): Whether or not to also compute the derivative with respect to the additive term. Defaults to False.
            post_recon_filter (Transform | None, optional): Filter applied to the reconstruction. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The derivatives, a tensor of shape ``[1, ...]`` (``[2, ...]`` if ``include_additive_term``) where ``...`` is the shape of the projections, and the reconstructed object at subiteration ``subiteration_number`` (filtered, if a filter is given).
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
        return V, final_recon

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
        dim_idx = len(FP_current_update.shape) - len(self.likelihood.system_matrix.proj_meta.shape) # the subset dimension is always 0 unless dual photopeak is used, in which case it may be dimension 1
        extra_dims = [FP_current_update.shape[i] for i in range(dim_idx)]
        if include_additive_term:
            output_additive = self.likelihood.compute_gradient_sf(object_current_update, FP_current_update, subset_idx)(output)
            output_total = torch.zeros((
                2,
                *extra_dims,
                *self.likelihood.system_matrix.proj_meta.shape,
            )).to(pytomography.device)            
            output_total[0].index_copy_(dim_idx, subset_indices_array.to(pytomography.device), output_primary)
            output_total[1].index_copy_(dim_idx, subset_indices_array.to(pytomography.device), output_additive)
            return output_total
        else:
            output_total = torch.zeros((
                1,
                *extra_dims,
                *self.likelihood.system_matrix.proj_meta.shape,
            )).to(pytomography.device)
            output_total[0].index_copy_(dim_idx, subset_indices_array.to(pytomography.device), output_primary)
        return output_total
        
class OSEM(LinearPreconditionedGradientAscentAlgorithm):
    r"""Ordered subset expectation maximisation: :math:`f^{n+1} = f^{n} + \frac{f^n}{H_m^T 1} \nabla_{f} L(g_m|f^{n})`.

    With :class:`~pytomography.likelihoods.PoissonLogLikelihood` this is the multiplicative OSEM update
    :math:`f^{n+1} = \frac{f^n}{H_m^T 1} H_m^T \frac{g_m}{H_m f^n + s_m}`.

    Args:
        likelihood (Likelihood): The likelihood :math:`L`.
        object_initial (torch.Tensor | None, optional): The initial object, [Lx, Ly, Lz]. Defaults to None: the system
            matrix's initial object.

    Example:
        .. code-block:: python

            recon = OSEM(likelihood)(n_iters=4, n_subsets=8)
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
    r"""Ordered subset maximum a posteriori one step late (OSMAPOSL), OSEM with a prior.

    The update is :math:`f^{n+1} = f^{n} + \frac{f^n}{H_m^T 1+\beta\nabla_f V(f^n)} \left[ \nabla_{f} L(g_m|f^{n}) -
    \beta\nabla_f V(f^n) \right]`.

    Args:
        likelihood (Likelihood): The likelihood :math:`L`.
        object_initial (torch.Tensor | None, optional): The initial object, [Lx, Ly, Lz]. Defaults to None: the system
            matrix's initial object.
        prior (Prior | None, optional): The prior :math:`V` (with its weight :math:`\beta`). Defaults to None.
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
    r"""Rescaled block iterative expectation maximisation (RBI-EM).

    The preconditioner is :math:`C^n(f^n) = f^n / (r_m \sum_k H_k^T 1)`, with :math:`r_m` the largest ratio over the
    voxels of :math:`H_m^T 1` to :math:`\sum_k H_k^T 1`: each subset's step is scaled to the whole data's
    sensitivity.

    Args:
        likelihood (Likelihood): The likelihood :math:`L`.
        object_initial (torch.Tensor | None, optional): The initial object, [Lx, Ly, Lz]. Defaults to None: the system
            matrix's initial object.
        prior (Prior | None, optional): The prior :math:`V`; RBI-EM's preconditioner does not include it (see
            :class:`RBIMAP`). Defaults to None.
    """
    _sums_normalisation = True

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
        r"""The preconditioner :math:`C^n(f^n) = f^n / (r_m \sum_k H_k^T 1)`.

        Args:
            object (torch.Tensor): The object :math:`f^n`.
            n_iter (int): Iteration number.
            n_subset (int): Subset index :math:`m`.

        Returns:
            torch.Tensor: The preconditioner, [Lx, Ly, Lz].
        """
        norm_BP = self.likelihood._get_normBP(n_subset)
        norm_BP_allsubsets = self.likelihood._get_normBP(n_subset, return_sum=True)
        rm = torch.max(norm_BP / (norm_BP_allsubsets + pytomography.delta))
        return object/(norm_BP_allsubsets*rm + pytomography.delta)
    
class RBIMAP(PreconditionedGradientAscentAlgorithm):
    r"""Rescaled block iterative maximum a posteriori (RBI-MAP), RBI-EM with a prior.

    The preconditioner is :math:`C^n(f^n) = f^n / (r_m \sum_k H_k^T 1 + \beta\nabla_f V(f^n))`, with :math:`r_m` the
    largest ratio over the voxels of :math:`H_m^T 1 + \beta\nabla_f V` to :math:`\sum_k H_k^T 1 + \beta\nabla_f V`.

    Args:
        likelihood (Likelihood): The likelihood :math:`L`.
        object_initial (torch.Tensor | None, optional): The initial object, [Lx, Ly, Lz]. Defaults to None: the system
            matrix's initial object.
        prior (Prior | None, optional): The prior :math:`V` (with its weight :math:`\beta`). Defaults to None.
    """
    _sums_normalisation = True

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
        r"""The preconditioner :math:`C^n(f^n) = f^n / (r_m \sum_k H_k^T 1 + \beta\nabla_f V(f^n))`.

        Args:
            object (torch.Tensor): The object :math:`f^n`.
            n_iter (int): Iteration number.
            n_subset (int): Subset index :math:`m`.

        Returns:
            torch.Tensor: The preconditioner, [Lx, Ly, Lz].
        """
        norm_BP = self.likelihood._get_normBP(n_subset)
        norm_BP_allsubsets = self.likelihood._get_normBP(n_subset, return_sum=True)
        rm = torch.max((norm_BP + self.prior_gradient) / (norm_BP_allsubsets + self.prior_gradient + pytomography.delta))
        return object/(norm_BP_allsubsets*rm + self.prior_gradient + pytomography.delta)
    
class BSREM(LinearPreconditionedGradientAscentAlgorithm):
    r"""Block sequential regularized expectation maximisation (BSREM).

    The update is :math:`f^{n+1} = f^{n} + \frac{\alpha(n) f^n}{\omega_m \sum_k H_k^T 1} \left[\nabla_{f} L(g_m|f^{n})
    - \beta\nabla_f V(f^n) \right]`, with :math:`\omega_m` the share of the data in subset :math:`m` and
    :math:`\alpha` the relaxation sequence.

    Args:
        likelihood (Likelihood): The likelihood :math:`L`.
        object_initial (torch.Tensor | None, optional): The initial object, [Lx, Ly, Lz]. Defaults to None: the system
            matrix's initial object.
        prior (Prior | None, optional): The prior :math:`V` (with its weight :math:`\beta`). Defaults to None.
        relaxation_sequence (Callable, optional): The relaxation :math:`\alpha(n)`, a function of the iteration number
            (counted from 0; every subiteration of an iteration uses the same value). Defaults to 1 for every
            iteration.
        addition_after_iteration (float, optional): A floor for the object: after every update, voxels at or below
            this value are set to it, so that voxels cannot get "locked" at 0. Defaults to 1e-4.
    """
    _sums_normalisation = True

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
    r"""The kernel method: OSEM of the coefficients :math:`\alpha` of an image :math:`f = K\alpha`.

    The update is :math:`\alpha^{n+1} = \alpha^{n} + \frac{\alpha^n}{\tilde{H}_m^T 1} \nabla_{\alpha}
    L(g_m|\alpha^{n})`, with :math:`\tilde{H} = HK` the system matrix of the likelihood, a
    :class:`~pytomography.projectors.shared.KEMSystemMatrix` (which holds the kernel :math:`K`). The image
    :math:`f = K\alpha` is returned, and given to the callbacks.

    Args:
        likelihood (Likelihood): The likelihood :math:`L`, whose system matrix is a KEMSystemMatrix.
        object_initial (torch.Tensor | None, optional): The initial coefficients, [Lx, Ly, Lz]. Defaults to None: the
            system matrix's initial object.
    """
    def _compute_callback(self, n_iter: int, n_subset: int):
        r"""Runs the callback on the image :math:`f^n = K \hat{\alpha}^{n}` rather than on the coefficients.

        Args:
            n_iter (int): Iteration number.
            n_subset (int): Subset index.
        """
        self.callback.run(self.likelihood.system_matrix.kem_transform.forward(self.object_prediction), n_iter, n_subset)
    def __call__(
        self, *args, **kwargs
    ):
        r"""Reconstructs the coefficients as OSEM does, and returns the image :math:`K\hat{\alpha}`.

        Args:
            *args: As for :class:`PreconditionedGradientAscentAlgorithm` (``n_iters``, ``n_subsets``...).
            **kwargs: As for :class:`PreconditionedGradientAscentAlgorithm`.

        Returns:
            torch.Tensor: The image :math:`f = K\hat{\alpha}`, [Lx, Ly, Lz] (not the coefficients).
        """
        object_prediction = super(KEM, self).__call__(*args, **kwargs)
        return self.likelihood.system_matrix.kem_transform.forward(object_prediction)
    
class MLEM(OSEM):
    r"""Maximum likelihood expectation maximisation (MLEM): OSEM with one subset.

    The update is :math:`f^{n+1} = f^{n} + \frac{f^n}{H^T 1} \nabla_{f} L(g|f^{n})`.

    Args:
        likelihood (Likelihood): The likelihood :math:`L`.
        object_initial (torch.Tensor | None, optional): The initial object, [Lx, Ly, Lz]. Defaults to None: the system
            matrix's initial object.
    """
    _memory_fixed = {'n_subsets': 1}

    def __call__(self, n_iters, callback=None):
        """Reconstructs: ``n_iters`` iterations with all the data.

        Args:
            n_iters (int): Number of iterations.
            callback (Callback, optional): Run after each iteration. Defaults to None.

        Returns:
            torch.Tensor: The reconstructed object, [Lx, Ly, Lz].
        """
        return super(MLEM, self).__call__(n_iters, n_subsets=1, callback=callback)

class SART(PreconditionedGradientAscentAlgorithm):
    r"""Simultaneous algebraic reconstruction technique (SART), from a system matrix and projections.

    The update (equation 3 of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8506772/) is
    :math:`f^{n+1} = f^{n} + \frac{\alpha(n)}{H_m^T 1} H_m^T \frac{g_m - (H_m f^n + s_m)}{H_m 1}`; it makes its own
    likelihood (:class:`~pytomography.likelihoods.SARTWeightedNegativeMSELikelihood`).

    Args:
        system_matrix (SystemMatrix): The system matrix :math:`H`.
        projections (torch.Tensor): The measured projections :math:`g` (for CT, line integrals).
        additive_term (torch.Tensor | None, optional): The additive term :math:`s`. Defaults to None.
        object_initial (torch.Tensor | None, optional): The initial object, [Lx, Ly, Lz]. Defaults to None: the system
            matrix's initial object.
        relaxation_sequence (Callable, optional): The relaxation :math:`\alpha(n)`, a function of the iteration number
            (counted from 0). Defaults to 1 for every iteration.
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
        r"""The preconditioner :math:`C^n = \alpha(n) / H_m^T 1`.

        Args:
            object (torch.Tensor): The object :math:`f^n` (not used).
            n_iter (int): Iteration number.
            n_subset (int): Subset index :math:`m`.

        Returns:
            torch.Tensor: The preconditioner, [Lx, Ly, Lz].
        """
        relaxation_factor = self.relaxation_sequence(n_iter)
        return relaxation_factor/(self.likelihood._get_normBP(n_subset) + pytomography.delta)
    
class PGAAMultiBedSPECT(PreconditionedGradientAscentAlgorithm):
    """Reconstructs several SPECT bed positions together, one algorithm per bed, and stitches their images.

    Each subiteration runs one subiteration of every bed's algorithm and stitches the beds' images
    (:func:`~pytomography.io.SPECT.stitch_beds`), each bed placed by the ``object_meta`` of its system matrix (as
    :func:`~pytomography.io.SPECT.read_dicom_beds` gives them). After a reconstruction, ``object_meta`` is the grid of
    the stitched image; masks for :meth:`compute_uncertainty` go on it.

    Args:
        reconstruction_algorithms (Sequence[PreconditionedGradientAscentAlgorithm]): The algorithm of each bed, e.g.
            an :class:`OSEM` per bed, each with its own likelihood and system matrix.

    Example:
        .. code-block:: python

            algorithms = []
            for projections, object_meta, proj_meta in io.SPECT.read_dicom_beds(files_NM):
                system_matrix = SPECTSystemMatrix([], [], object_meta, proj_meta)   # with each bed's transforms
                algorithms.append(OSEM(PoissonLogLikelihood(system_matrix, projections[peak])))
            recon = PGAAMultiBedSPECT(algorithms)(n_iters=4, n_subsets=8)
    """
    def __init__(
        self,
        reconstruction_algorithms: Sequence[object],
        ) -> None:
        self.reconstruction_algorithms = reconstruction_algorithms
        self.object_metas = [a.likelihood.system_matrix.object_meta for a in reconstruction_algorithms]
        self.object_meta = None

    def _memory_title(self) -> str:
        return f"{type(self).__name__} ({len(self.reconstruction_algorithms)} beds)"

    def _bed_parts(self, settings: dict) -> list[MemoryPart]:
        """The arrays of every bed's reconstruction with ``settings``: what each holds for the whole run (they all stay
        alive, one bed at a time is updated), the arrays of a subset and of a projection of the bed that needs most,
        and the beds' images, stacked, and the stitched image (three images per bed, rounded up)."""
        estimates = []
        for algorithm in self.reconstruction_algorithms:
            system_matrix = algorithm.likelihood.system_matrix
            estimates.append(system_matrix._memory_estimate(settings, held_parts(algorithm._memory_held()), alternatives=False,
                                                            extra=algorithm._memory_extra))
        parts = []
        for k, estimate in enumerate(estimates, start=1):
            held = [p for p in estimate.parts if p.scope == 'held']
            parts.append(MemoryPart(f'bed {k}: what its reconstruction holds', sum(p.ram_bytes for p in held), sum(p.gpu_bytes for p in held)))
        largest = max(estimates, key=lambda e: sum(p.ram_bytes + p.gpu_bytes for p in e.parts if p.scope in ('subset', 'chunk')))
        parts += [p for p in largest.parts if p.scope in ('subset', 'chunk')]
        image = 4 * math.prod(self.reconstruction_algorithms[0].likelihood.system_matrix.object_meta.shape)
        parts.append(MemoryPart("the beds' images, stacked, and the stitched image",
                                **on_device(pytomography.device, 3 * len(estimates) * image), scope='subset'))
        return parts

    def _bed_estimate(self, settings: dict, alternatives: bool = True) -> MemoryEstimate:
        n = settings['n_subsets']
        sm = self.reconstruction_algorithms[0].likelihood.system_matrix
        return memory_estimate(f"{self._memory_title()}: {n} subset{'' if n == 1 else 's'}", self._bed_parts(settings), settings=settings,
                               cuda_projector=getattr(sm, 'memory_cuda_projector', False))

    def estimate_memory(self, n_subsets: int | None = None, N_splits: int | None = None) -> MemoryEstimate:
        """The predicted peak RAM and GPU memory of the reconstruction of every bed, for ``n_subsets`` subsets: what each
        bed's reconstruction holds (see :meth:`PreconditionedGradientAscentAlgorithm.estimate_memory`), the arrays of a
        subset of the bed that needs most, and the images stitched together.

        Args:
            n_subsets (int | None, optional): Number of subsets. Defaults to None: those of the last call, or 1.
            N_splits (int | None, optional): Projector splits, for system matrices that take them. Defaults to None.

        Returns:
            MemoryEstimate: The estimate.
        """
        first = self.reconstruction_algorithms[0]
        settings = {name: present for name, (present, _) in first.likelihood.system_matrix._memory_settings().items()}
        settings['n_subsets'] = n_subsets if n_subsets is not None else getattr(first, 'n_subsets', 1)
        if N_splits is not None and 'N_splits' in settings:
            settings['N_splits'] = N_splits
        return self._bed_estimate(settings)

    def plan_memory(self, ram_gb: float | None = None, gpu_gb: float | None = None, **fixed) -> MemoryPlan:
        """The fastest settings of the reconstruction of every bed that fit in ``ram_gb`` of RAM and ``gpu_gb`` of GPU
        memory (see :meth:`PreconditionedGradientAscentAlgorithm.plan_memory`), counted as in :meth:`estimate_memory`.

        Args:
            ram_gb (float | None, optional): RAM to fit in, in GB. Defaults to None: what is available now.
            gpu_gb (float | None, optional): GPU memory to fit in, in GB. Defaults to None: what is available now.
            **fixed: Settings to keep, by name, e.g. ``n_subsets``.

        Returns:
            MemoryPlan: The plan.
        """
        settings = self.reconstruction_algorithms[0].likelihood.system_matrix._memory_settings()
        return plan_settings(self._memory_title(), settings, self._bed_estimate, fixed, ram_gb, gpu_gb)

    def __call__(
        self,
        n_iters: int,
        n_subsets: int,
        callback: Callback | Sequence[Callback] | None = None,
        ) -> torch.Tensor:
        """Reconstructs every bed for ``n_iters`` iterations of ``n_subsets`` subsets, and returns the stitched image.

        Args:
            n_iters (int): Number of iterations.
            n_subsets (int): Number of subsets.
            callback (Callback | list[Callback] | None, optional): A callback run on the stitched image after each
                subiteration, or a list of one callback per bed, each run on its bed's image (e.g. a
                :class:`~pytomography.callbacks.DataStorageCallback` per bed, for :meth:`compute_uncertainty`).
                Defaults to None.

        Returns:
            torch.Tensor: The stitched image, [Lx, Ly, Lz'], on the grid ``object_meta``.
        """
        self.callback = callback
        for i in range(n_iters):
            for j in range(n_subsets):
                self.recons = []
                for recon_algo in self.reconstruction_algorithms:
                    self.recons.append(recon_algo(1, n_subsets, n_subset_specific=j))
                self.object_prediction, self.object_meta = stitch_beds(torch.stack(self.recons), self.object_metas)
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
                self.object_prediction, self.object_meta = stitch_beds(torch.stack(self.recons), self.object_metas)
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
        r"""Estimates the uncertainty (standard deviation) of the sum of the stitched image over a region.

        The reconstruction must have been run with a list of one :class:`~pytomography.callbacks.DataStorageCallback`
        per bed, and the beds' algorithms must estimate uncertainty (their preconditioner linear in the object, e.g.
        OSEM). The counts in the mask are :math:`S = \sum_k (w_k\xi)\cdot x_k`: each bed :math:`k` contributes its
        reconstruction :math:`x_k` in the part of the mask :math:`\xi` the stitched image takes from it (its stitching
        weights :math:`w_k`, see :func:`~pytomography.io.SPECT.stitch_beds`). With :math:`v_k = [V_k]'(w_k\xi)` the
        derivative of bed :math:`k`'s part with respect to its projections :math:`y_k`, the variance is

        .. math::

            \text{Var}(S) = \sum_k y_k\cdot v_k^2 + 2\sum_{k<l} \sum_{j\in k\cap l} y_j\, v_{k,j}\, v_{l,j}

        where the second sum runs over the projection data that beds :math:`k` and :math:`l` share.
        :func:`~pytomography.io.SPECT.read_dicom_beds` fills the overlap of each bed's projections with its neighbour's
        counts, so neighbouring beds share all of their overlap and their estimates there are correlated; beds
        reconstructed from their own projections share nothing, and their variances add. A slice of projections (all
        angles and radial positions at one :math:`z`) is taken to be shared by two beds where the two beds'
        projections are equal there. The additive term (if ``include_additive_term``) is treated the same way, its data
        taken to be shared in the same slices as the projections (:func:`~pytomography.io.SPECT.read_dicom_beds` fills
        every energy window) and with the variance estimate of the lower bed; this is exact when the additive term's
        variance estimate is that of an unsmoothed energy window estimate, and approximate within a few kernel widths
        of a bed's edge when it is smoothed.

        Args:
            mask (torch.Tensor): The region, a boolean tensor on the grid ``object_meta`` of the stitched image (e.g.
                ``io.load_mask(...).on(object_meta, label=..., threshold=0.5)``).
            data_storage_callbacks (Sequence[DataStorageCallback]): The callbacks the reconstruction was run with, one
                per bed, in the order of ``reconstruction_algorithms``.
            subiteration_number (int | None, optional): The subiteration whose images are used. Defaults to None: the
                last.
            return_pct (bool, optional): Also return the uncertainty in percent of the sum. Defaults to False.
            include_additive_term (bool, optional): Include the uncertainty of the additive term; this needs the
                likelihoods' ``additive_term_variance_estimate``. Defaults to False.

        Returns:
            float | tuple[float, float]: The uncertainty of the sum, in the image's units, and, with ``return_pct``,
            the uncertainty in percent of the sum.
        """
        if subiteration_number is None:
            subiteration_number = len(data_storage_callbacks[0].objects) - 1
        recons = [data_storage_callback.objects[subiteration_number].to(pytomography.device) for data_storage_callback in data_storage_callbacks]
        stitching_weights, zs = stitch_beds(torch.stack(recons), self.object_metas, return_stitching_weights=True)
        Lz = recons[0].shape[-1]
        Vs = [None] * len(recons)
        variance = 0
        total_counts = 0
        for k, algorithm in enumerate(self.reconstruction_algorithms):
            # The part of the mask the stitched image takes from this bed
            mask_k = mask[:,:,zs[k]:zs[k]+Lz] * stitching_weights[k]
            if mask_k.sum()==0:
                continue
            V, _ = algorithm._compute_V(mask_k, data_storage_callbacks[k], subiteration_number, include_additive_term=include_additive_term)
            Vs[k] = V
            projections = algorithm.likelihood.projections.to(V.device)
            variance += torch.sum(V[0] * projections * V[0])
            if include_additive_term:
                variance += torch.sum(algorithm.likelihood.additive_term_variance_estimate(V[1]))
            total_counts += (recons[k]*mask_k).sum().item()
        # Covariance between beds, from the projection data they share
        for k in range(len(recons)):
            for l in range(k+1, len(recons)):
                if Vs[k] is None or Vs[l] is None:
                    continue
                # make k the lower bed: its additive term variance estimate is used below
                lower, upper = (k, l) if zs[k] <= zs[l] else (l, k)
                slices_lower, slices_upper = self._shared_projection_slices(lower, upper, zs, Lz)
                if len(slices_lower)==0:
                    continue
                V_lower, V_upper = Vs[lower], Vs[upper]
                slices_lower, slices_upper = slices_lower.to(V_lower.device), slices_upper.to(V_lower.device)
                projections = self.reconstruction_algorithms[lower].likelihood.projections.to(V_lower.device)
                variance += 2 * torch.sum(V_lower[0][...,slices_lower] * projections[...,slices_lower] * V_upper[0][...,slices_upper])
                if include_additive_term:
                    # The variance estimate is a quadratic form a -> (Ka)*var*(Ka) (elementwise): (F(a+b)-F(a)-F(b))/2 = (Ka)*var*(Kb)
                    F = self.reconstruction_algorithms[lower].likelihood.additive_term_variance_estimate
                    a = V_lower[1]
                    b = torch.zeros_like(a)
                    b[...,slices_lower] = V_upper[1][...,slices_upper]
                    cross = (F(a+b) - F(a) - F(b)) / 2
                    variance += 2 * torch.sum(cross[...,slices_lower])
        uncertainty_abs_total = torch.sqrt(variance).item() if torch.is_tensor(variance) else 0.0
        if return_pct:
            uncertainty_pct = uncertainty_abs_total / total_counts * 100
            return uncertainty_abs_total, uncertainty_pct
        else:
            return uncertainty_abs_total

    def _shared_projection_slices(
        self,
        k: int,
        l: int,
        zs: Sequence[int],
        Lz: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Finds the slices of projections (all angles and radial positions at one z) that beds ``k`` and ``l`` share: those where their projections overlap in z and are equal (the same counts, as :func:`~pytomography.io.SPECT.read_dicom_beds` gives neighbouring beds in their overlap).

        Args:
            k (int): Index of the first bed (in ``reconstruction_algorithms``)
            l (int): Index of the second bed
            zs (Sequence[int]): Index of each bed's first slice in the stitched image
            Lz (int): Number of slices of each bed

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The shared slices, as indices of bed ``k``'s slices and of bed ``l``'s slices.
        """
        z_start, z_end = max(zs[k], zs[l]), min(zs[k], zs[l]) + Lz
        if z_end <= z_start:
            empty = torch.zeros(0, dtype=torch.long)
            return empty, empty
        slices_k = torch.arange(z_start - zs[k], z_end - zs[k])
        slices_l = torch.arange(z_start - zs[l], z_end - zs[l])
        projections_k = self.reconstruction_algorithms[k].likelihood.projections
        projections_l = self.reconstruction_algorithms[l].likelihood.projections.to(projections_k.device)
        equal = projections_k[...,slices_k.to(projections_k.device)] == projections_l[...,slices_l.to(projections_k.device)]
        shared = equal.reshape(-1, len(slices_k)).all(dim=0).cpu()
        return slices_k[shared], slices_l[shared]
