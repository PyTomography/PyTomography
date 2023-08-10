:py:mod:`pytomography.utils.nist_data`
======================================

.. py:module:: pytomography.utils.nist_data


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.utils.nist_data.dual_sqrt_exponential
   pytomography.utils.nist_data.get_E_mu_data_from_datasheet
   pytomography.utils.nist_data.get_mu_from_spectrum_interp



.. py:function:: dual_sqrt_exponential(energy, c1, c2, d1, d2)

   Function used for curve fitting of linear attenuation coefficient vs. photon energy curves from NIST. It's given by the functional form :math:`f(x) = c_1e^{-d_1\sqrt{x}} + c_2e^{-d_2\sqrt{x}}`. It was chosen purely because it gave good fit results.

   :param energy: Energy of photon
   :type energy: float
   :param c1: Fit parameter 1
   :type c1: float
   :param c2: Fit parameter 2
   :type c2: float
   :param d1: Fit parameter 3
   :type d1: float
   :param d2: Fit parameter 4
   :type d2: float

   :returns: _description_
   :rtype: float


.. py:function:: get_E_mu_data_from_datasheet(file)

   Return energy and linear attenuation data from NIST datafiles of mass attenuation coefficients between 50keV and 511keV.

   :param file: Location of NIST data file. Corresponds to a particular element/material.
   :type file: str

   :returns: Energy and linear attenuation values.
   :rtype: tuple[np.array, np.array]


.. py:function:: get_mu_from_spectrum_interp(file, energy)

   Gets linear attenuation corresponding to a given energy in in the data from ``file``.

   :param file: Filepath of the mu-energy data
   :type file: str
   :param energy: Energy at which mu is computed
   :type energy: float

   :returns: Linear attenuation coefficient (in 1/cm) at the desired energies.
   :rtype: np.array


