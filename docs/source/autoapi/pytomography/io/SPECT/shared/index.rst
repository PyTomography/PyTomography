:py:mod:`pytomography.io.SPECT.shared`
======================================

.. py:module:: pytomography.io.SPECT.shared


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.SPECT.shared.subsample_projections
   pytomography.io.SPECT.shared.subsample_projections_and_modify_metadata
   pytomography.io.SPECT.shared.subsample_amap



.. py:function:: subsample_projections(projections, N_pixel, N_angle, N_angle_start)

   Subsamples SPECT projections by averaging over N_pixel x N_pixel pixel regions and by removing certain angles

   :param projections: Projections to subsample
   :type projections: torch.Tensor
   :param N_pixel: Pixel reduction factor (1 means no reduction)
   :type N_pixel: int
   :param N_angle: Angle reduction factor (1 means no reduction)
   :type N_angle: int
   :param N_angle_start: Angle index to start at
   :type N_angle_start: int

   :returns: Subsampled projections
   :rtype: torch.Tensor


.. py:function:: subsample_projections_and_modify_metadata(object_meta, proj_meta, projections, N_pixel = 1, N_angle = 1, N_angle_start = 0)

   Subsamples SPECT projection and modifies metadata accordingly

   :param object_meta: Object metadata
   :type object_meta: ObjectMeta
   :param proj_meta: Projection metadata
   :type proj_meta: SPECTProjMeta
   :param projections: Projections to subsample
   :type projections: torch.Tensor
   :param N_pixel: Pixel reduction factor (1 means no reduction). Defaults to 1.
   :type N_pixel: int
   :param N_angle: Angle reduction factor (1 means no reduction). Defaults to 1.
   :type N_angle: int
   :param N_angle_start: Angle index to start at. Defaults to 0.
   :type N_angle_start: int

   :returns: Modified object metadata, modified projection metadata, subsampled projections
   :rtype: Sequence


.. py:function:: subsample_amap(amap, N)

   Subsamples 3D attenuation map by averaging over N x N x N regions

   :param amap: Original attenuation map
   :type amap: torch.Tensor
   :param N: Factor to reduce by
   :type N: int

   :returns: Subsampled attenuation map
   :rtype: torch.Tensor


