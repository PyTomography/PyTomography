:py:mod:`pytomography.io.CT.attenuation_map`
============================================

.. py:module:: pytomography.io.CT.attenuation_map

.. autoapi-nested-parse::

   This module is used to create attenuation maps from CT images required for SPECT/PET attenuation correction.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.CT.attenuation_map.bilinear_transform
   pytomography.io.CT.attenuation_map.get_HU_from_spectrum_interp
   pytomography.io.CT.attenuation_map.get_HU_mu_curve
   pytomography.io.CT.attenuation_map.HU_to_mu
   pytomography.io.CT.attenuation_map.get_HU_corticalbone
   pytomography.io.CT.attenuation_map.get_ECT_from_corticalbone_HU
   pytomography.io.CT.attenuation_map.get_HU2mu_conversion



Attributes
~~~~~~~~~~

.. autoapisummary::

   pytomography.io.CT.attenuation_map.module_path
   pytomography.io.CT.attenuation_map.DATADIR
   pytomography.io.CT.attenuation_map.FILE_WATER
   pytomography.io.CT.attenuation_map.FILE_AIR
   pytomography.io.CT.attenuation_map.FILE_CBONE


.. py:data:: module_path

   

.. py:data:: DATADIR

   

.. py:data:: FILE_WATER

   

.. py:data:: FILE_AIR

   

.. py:data:: FILE_CBONE

   

.. py:function:: bilinear_transform(HU, a1, a2, b1, b2)

   Function used to convert between Hounsfield Units at an effective CT energy and linear attenuation coefficient at a given SPECT radionuclide energy. It consists of two distinct linear curves in regions :math:`HU<0` and :math:`HU \geq 0`.

   :param HU: Hounsfield units at CT energy
   :type HU: float
   :param a1: Fit parameter 1
   :type a1: float
   :param a2: Fit parameter 2
   :type a2: float
   :param b1: Fit parameter 3
   :type b1: float
   :param b2: Fit parameter 4
   :type b2: float

   :returns: Linear attenuation coefficient at SPECT energy
   :rtype: float


.. py:function:: get_HU_from_spectrum_interp(file, energy)

   Obtains the Hounsfield Units of some material at a given energy

   :param file: Filepath of material
   :type file: str
   :param energy: Energy at which HU is desired
   :type energy: float

   :returns: HU at the desired energies.
   :rtype: np.array


.. py:function:: get_HU_mu_curve(files_CT, CT_kvp, E_SPECT)

   Gets Housnfield Unit vs. linear attenuation coefficient for air, water, and cortical bone data points

   :param files_CT: Filepaths of all CT slices
   :type files_CT: Sequence[str]
   :param CT_kvp: Value of kVp for the CT scan
   :type CT_kvp: float
   :param E_SPECT: Photopeak energy of the SPECT scan
   :type E_SPECT: float

   :rtype: tuple[np.array, np.array]


.. py:function:: HU_to_mu(HU, E, p_water_opt, p_air_opt)

   Converts hounsfield units to linear attenuation coefficient

   :param HU: Hounsfield Unit value
   :type HU: float
   :param E: Effective CT energy
   :type E: float
   :param p_water_opt: Optimal fit parameters for mu vs. E data for water
   :type p_water_opt: Sequence[float]
   :param p_air_opt: Optimal fit parameters for mu vs. E data for air
   :type p_air_opt: Sequence[float]

   :returns: _description_
   :rtype: _type_


.. py:function:: get_HU_corticalbone(files_CT)

   Obtains the Hounsfield Unit corresponding to cortical bone from a CT scan.

   :param files_CT: CT data files
   :type files_CT: Sequence[str]

   :returns: Hounsfield unit of bone. If not found, then returns ``None``.
   :rtype: float | None


.. py:function:: get_ECT_from_corticalbone_HU(HU)

   Finds the effective CT energy that gives a cortical bone Hounsfield Unit value corresponding to ``HU``.

   :param HU: Hounsfield Unit of Cortical bone at effective CT energy
   :type HU: float

   :returns: Effective CT energy
   :rtype: float


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


