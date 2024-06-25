:py:mod:`pytomography.metadata.CT.ct_conebeam_flatpanel_metadata`
=================================================================

.. py:module:: pytomography.metadata.CT.ct_conebeam_flatpanel_metadata


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.metadata.CT.ct_conebeam_flatpanel_metadata.CTConeBeamFlatPanelProjMeta




.. py:class:: CTConeBeamFlatPanelProjMeta(angles, z_locations, detector_radius, beam_radius, shape, dr, COR=None)

   Bases: :py:obj:`pytomography.metadata.ProjMeta`

   Parent class for all different types of Projection Space Metadata. Implementation and required parameters will differ significantly between different imaging modalities.


   .. py:method:: _get_CORs()


   .. py:method:: _get_detector_pixel_s_v(device=None)


   .. py:method:: _get_detector_coordinates(idx)



