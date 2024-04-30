:py:mod:`pytomography.utils.scatter`
====================================

.. py:module:: pytomography.utils.scatter


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.utils.scatter.compute_EW_scatter



.. py:function:: compute_EW_scatter(projection_lower, projection_upper, width_lower, width_upper, width_peak, weighting_lower = 0.5, weighting_upper = 0.5, return_scatter_variance_estimate = False)

   Computes triple energy window estimate from lower and upper scatter projections as well as window widths

   :param projection_lower: Projection data corresponding to lower energy window
   :type projection_lower: torch.Tensor
   :param projection_upper: Projection data corresponding to upper energy window
   :type projection_upper: torch.Tensor
   :param width_lower: Width of lower energy window
   :type width_lower: float
   :param width_upper: Width of upper energy window
   :type width_upper: float
   :param width_peak: Width of peak energy window
   :type width_peak: float
   :param return_scatter_variance_estimate: Return scatter variance estimate. Defaults to False.
   :type return_scatter_variance_estimate: bool, optional

   :returns: Scatter estimate (and scatter variance estimate if specified)
   :rtype: torch.Tensor | Sequence[torch.Tensor]


