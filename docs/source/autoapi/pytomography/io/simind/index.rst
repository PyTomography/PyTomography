:py:mod:`pytomography.io.simind`
================================

.. py:module:: pytomography.io.simind


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.simind.find_first_entry_containing_header
   pytomography.io.simind.simind_projections_to_data
   pytomography.io.simind.simind_MEW_to_data
   pytomography.io.simind.simind_CT_to_data
   pytomography.io.simind.get_SPECT_recon_algorithm_simind



Attributes
~~~~~~~~~~

.. autoapisummary::

   pytomography.io.simind.relation_dict


.. py:data:: relation_dict

   

.. py:function:: find_first_entry_containing_header(list_of_attributes, header, dtype = np.float32)

   Finds the first entry in a SIMIND Interfile output corresponding to the header (header).

   :param list_of_attributes: Simind data file, as a list of lines.
   :type list_of_attributes: list[str]
   :param header: The header looked for
   :type header: str
   :param dtype: The data type to be returned corresponding to the value of the header. Defaults to np.float32.
   :type dtype: type, optional

   :returns: The value corresponding to the header (header).
   :rtype: float|str|int


.. py:function:: simind_projections_to_data(headerfile, distance = 'cm')

   Obtains ObjectMeta, ImageMeta, and projections from a SIMIND header file.

   :param headerfile: Path to the header file
   :type headerfile: str
   :param distance: The units of measurements in the SIMIND file (this is required as input, since SIMIND uses mm/cm but doesn't specify). Defaults to 'cm'.
   :type distance: str, optional

   :returns: Required information for reconstruction in PyTomography.
   :rtype: (ObjectMeta, ImageMeta, torch.Tensor[1, Ltheta, Lr, Lz])


.. py:function:: simind_MEW_to_data(headerfiles, distance = 'cm')

   Opens multiple projection files corresponding to the primary, lower scatter, and upper scatter windows

   :param headerfiles: List of file paths to required files. Must be in order of: 1. Primary, 2. Lower Scatter, 3. Upper scatter
   :type headerfiles: list[str]
   :param distance: The units of measurements in the SIMIND file (this is required as input, since SIMIND uses mm/cm but doesn't specify). Defaults to 'cm'.
   :type distance: str, optional

   :returns: Required information for reconstruction in PyTomography. First returned tensor contains primary data, and second returned tensor returns estimated scatter using the triple energy window method.
   :rtype: (ObjectMeta, ImageMeta, torch.Tensor[1, Ltheta, Lr, Lz], torch.Tensor[1, Ltheta, Lr, Lz])


.. py:function:: simind_CT_to_data(headerfile)

   Opens attenuation data from SIMIND output

   :param headerfile: Path to header file
   :type headerfile: str

   :returns: Tensor containing CT data.
   :rtype: torch.tensor[Lx,Ly,Lz]


.. py:function:: get_SPECT_recon_algorithm_simind(projections_header, scatter_headers = None, CT_header = None, psf_meta = None, prior = None, object_initial = None, recon_algorithm_class = OSEMOSL)


