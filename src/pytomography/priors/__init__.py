r"""Under the modification :math:`L(\tilde{f}, f) \to L(\tilde{f}, f)e^{-\beta V(f)}`, the log-liklihood becomes :math:`\ln L(\tilde{f},f) - \beta V(f)`. Typically, the prior has a form :math:`V(f) = \sum_{r,s} w_{r,s} \phi(f_r,f_s)`. In this expression, :math:`r` represents a voxel in the object, :math:`s` represents a neighbouring voxel to voxel :math:`r`, and :math:`w_{r,s}` is a weight that adjusts for the Euclidean distance between the voxels (set to unity for neighbouring voxels). For all priors implemented here, the neighbouring voxels considered are those surrounding a given voxel, so :math:`\sum_s` is a sum over 26 points."""

from .smoothness import QuadraticPrior, LogCoshPrior
from .relative_difference import RelativeDifferencePrior
from .prior import Prior

