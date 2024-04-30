:py:mod:`pytomography.io.PET.petsird`
=====================================

.. py:module:: pytomography.io.PET.petsird


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.PET.petsird.get_detector_ids
   pytomography.io.PET.petsird.get_scanner_LUT_from_header
   pytomography.io.PET.petsird.get_TOF_meta_from_header



.. py:function:: get_detector_ids(petsird_file, read_tof = None, read_energy = None, time_block_ids = None, return_header = False)

   Read all time blocks of a PETSIRD listmode file

   :param petsird_file: the PETSIRD listmode file
   :type petsird_file: str
   :param read_tof: read the TOF bin information of every event
                    default None means that is is auto determined
                    based on the scanner information (length of tof bin edges)
   :type read_tof: bool | None, optional
   :param read_energy: read the energy information of every event
                       default None means that is is auto determined
                       based on the scanner information (length of energy bin edges)
   :type read_energy: bool | None, optional

   :returns: PRD listmode file header, 2D array containing all event attributes
   :rtype: tuple[prd.types.Header, torch.Tensor]


.. py:function:: get_scanner_LUT_from_header(header)

   Obtains the scanner lookup table (relating detector IDs to physical coordinates) from a PETSIRD header.

   :param header: PETSIRD header
   :type header: prd.Header

   :returns: scanner lookup table.
   :rtype: torch.Tensor


.. py:function:: get_TOF_meta_from_header(header, n_sigmas = 3.0)

   Obtain time of flight metadata from a PETSIRD header

   :param header: PETSIRD header
   :type header: prd.Header
   :param n_sigmas: Number of sigmas to consider when performing TOF projection. Defaults to 3..
   :type n_sigmas: float, optional

   :returns: Time of flight metadata.
   :rtype: PETTOFMeta


