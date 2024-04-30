:py:mod:`pytomography.io.PET.clinical`
======================================

.. py:module:: pytomography.io.PET.clinical


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.PET.clinical.get_detector_info
   pytomography.io.PET.clinical.get_tof_meta
   pytomography.io.PET.clinical.modify_tof_events
   pytomography.io.PET.clinical.get_detector_ids_hdf5
   pytomography.io.PET.clinical.get_weights_hdf5
   pytomography.io.PET.clinical.get_additive_term_hdf5
   pytomography.io.PET.clinical.get_sensitivity_ids_hdf5
   pytomography.io.PET.clinical.get_sensitivity_ids_and_weights_hdf5



.. py:function:: get_detector_info(scanner_name)

   Obtains the PET geometry information for a given scanner.

   :param scanner_name: Name of the scanner
   :type scanner_name: str

   :returns: PET geometry dictionary required for obtaining lookup table
   :rtype: dict


.. py:function:: get_tof_meta(scanner_name)

   Obtains the PET TOF metadata for a given scanner

   :param scanner_name: Name of the scanner
   :type scanner_name: str

   :returns: PET TOF metadata
   :rtype: PETTOFMeta


.. py:function:: modify_tof_events(TOF_ids, scanner_name)

   Modifies TOF indices based on the scanner name

   :param TOF_ids: 1D tensor of TOF indices
   :type TOF_ids: torch.Tensor
   :param scanner_name: Name of scanner
   :type scanner_name: str

   :returns: Modified TOF indices
   :rtype: torch.Tensor


.. py:function:: get_detector_ids_hdf5(listmode_file, scanner_name)

   Returns the detector indices obtained from an HDF5 listmode file

   :param listmode_file: Path to the listmode file
   :type listmode_file: str
   :param scanner_name: Name of the PET scanner
   :type scanner_name: str

   :returns: Listmode form of the detector IDS for each event
   :rtype: torch.Tensor


.. py:function:: get_weights_hdf5(correction_file)

   Obtain the multiplicative weights from an HDF5 file that correct for attenuation and sensitivty effects for each of the detected listmode events.

   :param correction_file: Path to the correction file
   :type correction_file: str

   :returns: 1D tensor that contains the weights for each listmode event.
   :rtype: torch.Tensor


.. py:function:: get_additive_term_hdf5(correction_file)

   Obtain the additive term from an HDF5 file that corrects for random and scatte effects for each of the detected listmode events.

   :param correction_file: Path to the correction file
   :type correction_file: str

   :returns: 1D tensor that contains the additive term for each listmode event.
   :rtype: torch.Tensor


.. py:function:: get_sensitivity_ids_hdf5(corrections_file, scanner_name)

   Obtain the detector indices corresponding to all valid detector pairs (nonTOF): this is used to obtain the sensitivity weights for all detector pairs when computing the normalization factor.

   :param corrections_file: Path to the correction file
   :type corrections_file: str
   :param scanner_name: Name of the scanner
   :type scanner_name: str

   :returns: Tensor yielding all valid detector pairs
   :rtype: torch.Tensor[2,N_events]


.. py:function:: get_sensitivity_ids_and_weights_hdf5(corrections_file, scanner_name)

   Obtain the detector indices and corresponding detector weights for all valid detector pairs (nonTOF).

   :param corrections_file: Path to the correction file
   :type corrections_file: str
   :param scanner_name: Name of the scanner
   :type scanner_name: str

   :returns: Tensor yielding all valid detector pairs and tensor yielding corresponding weights.
   :rtype: torch.Tensor[2,N_events], torch.Tensor[N_events]


