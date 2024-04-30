:py:mod:`pytomography.io.SPECT.simind`
======================================

.. py:module:: pytomography.io.SPECT.simind


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.SPECT.simind.get_metadata
   pytomography.io.SPECT.simind._get_projections_from_single_file
   pytomography.io.SPECT.simind.get_projections
   pytomography.io.SPECT.simind.get_energy_window_width
   pytomography.io.SPECT.simind.combine_projection_data
   pytomography.io.SPECT.simind.get_attenuation_map
   pytomography.io.SPECT.simind.get_psfmeta_from_header



Attributes
~~~~~~~~~~

.. autoapisummary::

   pytomography.io.SPECT.simind.relation_dict


.. py:data:: relation_dict

   

.. py:function:: get_metadata(headerfile, distance = 'cm', corrfile = None)

   Obtains required metadata from a SIMIND header file.

   :param headerfile: Path to the header file
   :type headerfile: str
   :param distance: The units of measurements in the SIMIND file (this is required as input, since SIMIND uses mm/cm but doesn't specify). Defaults to 'cm'.
   :type distance: str, optional
   :param corrfile: .cor file used in SIMIND to specify radial positions for non-circular orbits. This needs to be provided for non-standard orbits.
   :type corrfile: str, optional

   :returns: Required information for reconstruction in PyTomography.
   :rtype: (SPECTObjectMeta, SPECTProjMeta, torch.Tensor[1, Ltheta, Lr, Lz])


.. py:function:: _get_projections_from_single_file(headerfile)

   Gets projection data from a SIMIND header file.

   :param headerfile: Path to the header file
   :type headerfile: str
   :param distance: The units of measurements in the SIMIND file (this is required as input, since SIMIND uses mm/cm but doesn't specify). Defaults to 'cm'.
   :type distance: str, optional

   :returns: Simulated SPECT projection data.
   :rtype: (torch.Tensor[1, Ltheta, Lr, Lz])


.. py:function:: get_projections(headerfiles, weights = None)

   Gets projection data from a SIMIND header file.

   :param headerfile: Path to the header file
   :type headerfile: str
   :param distance: The units of measurements in the SIMIND file (this is required as input, since SIMIND uses mm/cm but doesn't specify). Defaults to 'cm'.
   :type distance: str, optional

   :returns: Simulated SPECT projection data.
   :rtype: (torch.Tensor[1, Ltheta, Lr, Lz])


.. py:function:: get_energy_window_width(headerfile)

   Computes the energy window width from a SIMIND header file

   :param headerfile: Headerfile corresponding to SIMIND data
   :type headerfile: str

   :returns: Energy window width
   :rtype: float


.. py:function:: combine_projection_data(headerfiles, weights)

   Takes in a list of SIMIND headerfiles corresponding to different simulated regions and adds the projection data together based on the `weights`.

   :param headerfiles: List of filepaths corresponding to the SIMIND header files of different simulated regions
   :type headerfiles: Sequence[str]
   :param weights: Amount by which to weight each projection relative.
   :type weights: Sequence[str]

   :returns: Returns necessary object/projections metadata along with the projection data
   :rtype: (SPECTObjectMeta, SPECTProjMeta, torch.Tensor)


.. py:function:: get_attenuation_map(headerfile)

   Opens attenuation data from SIMIND output

   :param headerfile: Path to header file
   :type headerfile: str

   :returns: Tensor containing attenuation map required for attenuation correction in SPECT/PET imaging.
   :rtype: torch.Tensor[batch_size, Lx, Ly, Lz]


.. py:function:: get_psfmeta_from_header(headerfile)

   Obtains the SPECTPSFMeta data corresponding to a SIMIND simulation scan from the headerfile

   :param headerfile: SIMIND headerfile.
   :type headerfile: str

   :returns: SPECT PSF metadata required for PSF modeling in reconstruction.
   :rtype: SPECTPSFMeta


