:py:mod:`pytomography.metadata.PET.pet_tof_metadata`
====================================================

.. py:module:: pytomography.metadata.PET.pet_tof_metadata


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   pytomography.metadata.PET.pet_tof_metadata.PETTOFMeta




.. py:class:: PETTOFMeta(num_bins, tof_range, fwhm, n_sigmas = 3, bin_type = 'symmetric')

   Class for PET time of flight metadata. Contains information such as spatial binning and resolution.

   :param num_bins: Number of bins used to discretize time of flight data
   :type num_bins: int
   :param tof_range: Total range accross all TOF bins in mm.
   :type tof_range: float
   :param fwhm: FWHM corresponding to TOF uncertainty in mm.
   :type fwhm: float
   :param n_sigmas: Number of sigmas to consider when using TOF projection. Defaults to 3.
   :type n_sigmas: float
   :param bin_type: How the bins are arranged. Currently, the only option is symmetry, which means the bins are distributed symmetrically (evenly on each side) between the center of all LOR pairs. Defaults to 'symmetric'.
   :type bin_type: str, optional


