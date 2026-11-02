"""Conjugate gradients for least-squares reconstruction, weighted or not."""
from __future__ import annotations

import torch

import pytomography
from pytomography.likelihoods import NegativeMSELikelihood


class CGLS:
    r"""Minimise :math:`\frac12\|W^{1/2}(Hf+s-g)\|_2^2` without constraining the sign of the image.

    This is conjugate gradients on the normal equations, implemented using forward and back projections without
    forming :math:`H^T H`. :math:`W` is the diagonal of the likelihood's ``weights`` (the identity without them):
    weights of one over each bin's variance give weighted least squares, closer to the Poisson likelihood for counts
    (:func:`~pytomography.likelihoods.poisson_weights` makes them from the data).
    An optional fixed positive diagonal inverse preconditioner applies preconditioned CG to
    the same equations. It changes the iterations, not the full-rank least-squares solution.

    ``system_matrix.backward`` must be the adjoint of ``forward``. An approximate backprojector, such as the
    opposite-angle interpolating rotation in the current SPECT projector, does not provide the CGLS convergence
    guarantee. Check the adjoint relation before using a new projector. There are no ordered subsets or positivity
    clipping. Least squares, even weighted, is not a Poisson likelihood; early stopping can regularise an
    ill-conditioned problem.

    With the default zero initial image and no preconditioner, a rank-deficient problem approaches the minimum-norm
    least-squares solution. A nonzero initial image retains its null-space component; a preconditioner can change
    which solution is selected when the solution is not unique.

    Args:
        likelihood (NegativeMSELikelihood): Least-squares data, system matrix, additive background and weights.
            Its positive ``scaling_constant`` does not change the minimiser and is omitted from the recurrence.
            Projections must be a dense tensor.
        object_initial (torch.Tensor | None, optional): Initial image, [Lx, Ly, Lz], in the reconstructed image's
            units. Copied to ``pytomography.device`` and ``pytomography.dtype``. Defaults to a zero image.
        preconditioner (torch.Tensor | None, optional): Fixed positive diagonal inverse preconditioner, with the
            image's shape, on any device. For Jacobi preconditioning use the reciprocal diagonal of :math:`H^T H`.
            Defaults to the identity. With weights, for Jacobi preconditioning use the reciprocal diagonal of
            :math:`H^T W H`, e.g. ``1 / H.backward(weights * H.forward(ones))`` for a non-negative :math:`H`.
    """

    def __init__(
        self,
        likelihood: NegativeMSELikelihood,
        object_initial: torch.Tensor | None = None,
        preconditioner: torch.Tensor | None = None,
    ) -> None:
        self.likelihood = likelihood
        self.system_matrix = likelihood.system_matrix
        if likelihood.scaling_constant <= 0:
            raise ValueError("CGLS requires a positive least-squares scaling constant")
        if object_initial is None:
            self.object_prediction = torch.zeros(self.system_matrix.object_meta.shape,
                                                 device=pytomography.device, dtype=pytomography.dtype)
        else:
            self.object_prediction = object_initial.to(device=pytomography.device, dtype=pytomography.dtype).clone()
        self.preconditioner = None
        if preconditioner is not None:
            self.preconditioner = preconditioner.to(self.object_prediction).clone()
            if self.preconditioner.shape != self.object_prediction.shape or not torch.all(
                torch.isfinite(self.preconditioner) & (self.preconditioner > 0)
            ):
                raise ValueError("preconditioner must be finite, positive and have the image's shape")
        self.normal_residual_norms: list[float] = []
        self.converged = False
        self.iterations = 0

    def _precondition(self, residual: torch.Tensor) -> torch.Tensor:
        return residual if self.preconditioner is None else self.preconditioner * residual

    @torch.no_grad()
    def __call__(self, n_iters: int, rtol: float = 1e-6, atol: float = 0.0) -> torch.Tensor:
        r"""Reconstruct, restarting conjugate directions from the current image on each call.

        Stops when :math:`\|H^TW(g-s-Hf)\|_2 \leq \max(\mathrm{rtol}\|H^TW(g-s)\|_2,\mathrm{atol})` or after
        ``n_iters`` updates. ``normal_residual_norms`` contains the initial norm and the norms after each update.
        ``iterations`` and ``converged`` describe this call. The final residual is recomputed from the image before
        declaring convergence, since recursively updated residuals can drift in finite precision.

        Args:
            n_iters (int): Maximum number of updates, non-negative.
            rtol (float, optional): Relative normal-residual tolerance. Defaults to 1e-6.
            atol (float, optional): Absolute normal-residual tolerance, in backprojected-data units. Defaults to 0.

        Returns:
            torch.Tensor: Image [Lx, Ly, Lz], on ``pytomography.device``. Negative estimates are retained.
        """
        if n_iters < 0 or rtol < 0 or atol < 0:
            raise ValueError("iteration count and tolerances must be non-negative")
        forward, backward = self.system_matrix.forward, self.system_matrix.backward
        predicted = forward(self.object_prediction)
        data = self.likelihood.projections.to(predicted)
        if self.likelihood.additive_term is not None:
            data = data - self.likelihood.additive_term.to(predicted)
        weights = self.likelihood.weights
        weigh = (lambda y: y) if weights is None else (lambda y, w=weights.to(predicted): w * y)
        residual = data - predicted
        gradient = backward(weigh(residual))
        tolerance = max(atol, rtol * float(torch.linalg.vector_norm(backward(weigh(data)))))
        norm = float(torch.linalg.vector_norm(gradient))
        self.normal_residual_norms = [norm]
        self.iterations = 0
        self.converged = norm <= tolerance
        z = self._precondition(gradient)
        direction = z.clone()
        gamma = torch.sum(gradient * z)

        for _ in range(n_iters):
            if self.converged:
                break
            projected = forward(direction)
            denominator = torch.sum(weigh(projected) * projected)
            if not torch.isfinite(denominator) or denominator <= 0:
                raise RuntimeError("CGLS direction has no finite positive projection norm; check the adjoint pair")
            alpha = gamma / denominator.to(gamma.device)
            self.object_prediction.add_(direction * alpha)
            residual.sub_(projected * alpha.to(projected.device))
            gradient = backward(weigh(residual))
            norm = float(torch.linalg.vector_norm(gradient))
            self.iterations += 1
            if norm <= tolerance:
                residual = data - forward(self.object_prediction)
                gradient = backward(weigh(residual))
                norm = float(torch.linalg.vector_norm(gradient))
                self.converged = norm <= tolerance
                # A residual replacement requires a restart to retain conjugacy assumptions.
                z = self._precondition(gradient)
                direction = z.clone()
                gamma = torch.sum(gradient * z)
            else:
                z = self._precondition(gradient)
                gamma_next = torch.sum(gradient * z)
                direction = z + (gamma_next / gamma) * direction
                gamma = gamma_next
            self.normal_residual_norms.append(norm)
        return self.object_prediction
