r"""Priors (penalties) for maximum a posteriori reconstruction.

With a prior :math:`V`, the reconstruction maximises :math:`\ln L(g|f) - \beta V(f)` instead of the log-likelihood
alone. The priors here sum over the 26 nearest neighbours :math:`s` of each voxel :math:`r`,
:math:`V(f) = \sum_{r,s} w_{r,s} \phi(f_r,f_s)`, with weights :math:`w_{r,s}`:

* :class:`RelativeDifferencePrior`, :class:`QuadraticPrior`, :class:`LogCoshPrior`: the function :math:`\phi`;
* :class:`EuclideanNeighbourWeight`, :class:`AnatomyNeighbourWeight`, :class:`TopNAnatomyNeighbourWeight`: the
  weights;
* :class:`Prior`, :class:`NearestNeighbourPrior`, :class:`NeighbourWeight`: the base classes.

Give a prior to :class:`~pytomography.algorithms.OSMAPOSL`, :class:`~pytomography.algorithms.BSREM` or
:class:`~pytomography.algorithms.RBIMAP`.
"""

from .prior import Prior
from .nearest_neighbour import NearestNeighbourPrior, QuadraticPrior, LogCoshPrior, RelativeDifferencePrior, NeighbourWeight, EuclideanNeighbourWeight, AnatomyNeighbourWeight, TopNAnatomyNeighbourWeight