:py:mod:`pytomography.io.shared.interfile`
==========================================

.. py:module:: pytomography.io.shared.interfile


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.shared.interfile.get_header_value
   pytomography.io.shared.interfile.get_attenuation_map_interfile



.. py:function:: get_header_value(list_of_attributes, header, dtype = np.float32, split_substr=':=', split_idx=-1, return_all=False)

   Finds the first entry in an Interfile with the string ``header``

   :param list_of_attributes: Simind data file, as a list of lines.
   :type list_of_attributes: list[str]
   :param header: The header looked for
   :type header: str
   :param dtype: The data type to be returned corresponding to the value of the header. Defaults to np.float32.
   :type dtype: type, optional

   :returns: The value corresponding to the header (header).
   :rtype: float|str|int


.. py:function:: get_attenuation_map_interfile(headerfile)

   Opens attenuation data from SIMIND output

   :param headerfile: Path to header file
   :type headerfile: str

   :returns: Tensor containing attenuation map required for attenuation correction in SPECT/PET imaging.
   :rtype: torch.Tensor[batch_size, Lx, Ly, Lz]


