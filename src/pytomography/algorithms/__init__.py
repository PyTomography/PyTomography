"""Reconstruction algorithms.

Statistical algorithms take a likelihood (:mod:`pytomography.likelihoods`), and some a prior (:mod:`pytomography.priors`);
calling one with the number of iterations and subsets reconstructs:

* :class:`OSEM` and :class:`MLEM`: ordered subset expectation maximisation;
* :class:`OSMAPOSL`, :class:`BSREM`, :class:`RBIMAP`: maximum a posteriori, with a prior;
* :class:`RBIEM`: rescaled block iterative EM;
* :class:`KEM`: the kernel method (with :class:`~pytomography.projectors.shared.KEMSystemMatrix`);
* :class:`SART`: simultaneous algebraic reconstruction (least squares, e.g. CT);
* :class:`PGAAMultiBedSPECT`: several SPECT bed positions reconstructed together and stitched;
* :class:`PreconditionedGradientAscentAlgorithm`: the base class of all of them;
* :class:`DIPRecon`: the deep image prior.

:class:`FilteredBackProjection` is the analytic reconstruction of SPECT and CT projections.

Example:
    .. code-block:: python

        likelihood = PoissonLogLikelihood(system_matrix, projections)
        recon = OSEM(likelihood)(n_iters=4, n_subsets=8)
"""
from .preconditioned_gradient_ascent import PreconditionedGradientAscentAlgorithm, OSEM, OSMAPOSL, BSREM, KEM, RBIEM, RBIMAP, SART, PGAAMultiBedSPECT, MLEM
from .fbp import FilteredBackProjection
from .dip_recon import DIPRecon
