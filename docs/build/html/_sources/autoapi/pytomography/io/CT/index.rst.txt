:py:mod:`pytomography.io.CT`
============================

.. py:module:: pytomography.io.CT

.. autoapi-nested-parse::

   Input/output functions for the CT imaging modality. Currently, the data types supported are DICOM files.



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   attenuation_map/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.CT.get_HU2mu_conversion



.. py:function:: get_HU2mu_conversion(files_CT, CT_kvp, E_SPECT)

   Obtains the HU to mu conversion function that converts CT data to the required linear attenuation value in units of 1/cm required for attenuation correction in SPECT/PET imaging.

   :param files_CT: CT data files
   :type files_CT: Sequence[str]
   :param CT_kvp: kVp value for CT scan
   :type CT_kvp: float
   :param E_SPECT: Energy of photopeak in SPECT scan
   :type E_SPECT: float

   :returns: Conversion function from HU to mu.
   :rtype: function


