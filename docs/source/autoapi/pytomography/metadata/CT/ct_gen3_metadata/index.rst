:py:mod:`pytomography.metadata.CT.ct_gen3_metadata`
===================================================

.. py:module:: pytomography.metadata.CT.ct_gen3_metadata


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.metadata.CT.ct_gen3_metadata.CTGen3ProjMeta




.. py:class:: CTGen3ProjMeta(source_phis, source_rhos, source_zs, source_phi_offsets, source_rho_offsets, source_z_offsets, detector_centers_col_idx, detector_centers_row_idx, col_det_spacing, row_det_spacing, DSD, shape)

   Bases: :py:obj:`pytomography.metadata.ProjMeta`

   Parent class for all different types of Projection Space Metadata. Implementation and required parameters will differ significantly between different imaging modalities.


   .. py:method:: get_detector_coordinates(idxs)

      Obtain detector coordinates and the angles corresponding to idxs

      :param idxs: Angle indices
      :type idxs: torch.Tensor[int]

      :returns: Detector coordinates (in XYZ) at all angle indices.
      :rtype: torch.Tensor



