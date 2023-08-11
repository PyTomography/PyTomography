:py:mod:`pytomography.io.CT.dicom`
==================================

.. py:module:: pytomography.io.CT.dicom

.. autoapi-nested-parse::

   This module contains functionality for opening CT images.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.CT.dicom.open_CT_file
   pytomography.io.CT.dicom.compute_max_slice_loc_CT



.. py:function:: open_CT_file(files_CT)

   Given a list of seperate DICOM files, opens them up and stacks them together into a single CT image.

   :param files_CT: List of DICOM files corresponding to a particular scan
   :type files_CT: Sequence[str]

   :returns: CT scan in units of Hounsfield Units at the effective CT energy.
   :rtype: np.array


.. py:function:: compute_max_slice_loc_CT(files_CT)


