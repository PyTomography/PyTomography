"""Likelihoods: how well an object explains the measured data, which reconstruction algorithms maximise.

* :class:`PoissonLogLikelihood`: emission data (SPECT, PET), with an additive term for scatter and randoms;
* :class:`MonteCarloHybridSPECTPoissonLogLikelihood`: the same, with a Monte Carlo forward projection;
* :class:`NegativeMSELikelihood` and :class:`SARTWeightedNegativeMSELikelihood`: least squares (CT, SART);
* :class:`Likelihood`: the base class.
"""
from .mse_objective import NegativeMSELikelihood, SARTWeightedNegativeMSELikelihood
from .poisson_log_likelihood import PoissonLogLikelihood, MonteCarloHybridSPECTPoissonLogLikelihood
from .likelihood import Likelihood