:py:mod:`pytomography.io.dicom`
===============================

.. py:module:: pytomography.io.dicom


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.dicom.get_radii_and_angles
   pytomography.io.dicom.dicom_projections_to_data
   pytomography.io.dicom.HU_to_mu
   pytomography.io.dicom.get_affine_spect
   pytomography.io.dicom.get_affine_CT
   pytomography.io.dicom.dicom_CT_to_data



Attributes
~~~~~~~~~~

.. autoapisummary::

   pytomography.io.dicom.a1
   pytomography.io.dicom.b1
   pytomography.io.dicom.a2
   pytomography.io.dicom.b2


.. py:function:: get_radii_and_angles(ds)


.. py:function:: dicom_projections_to_data(file)


.. py:data:: a1
   :value: 0.00014376

   

.. py:data:: b1
   :value: 0.1352

   

.. py:data:: a2
   :value: 8.787e-05

   

.. py:data:: b2
   :value: 0.1352

   

.. py:function:: HU_to_mu(HU)


.. py:function:: get_affine_spect(ds)


.. py:function:: get_affine_CT(ds, max_z)


.. py:function:: dicom_CT_to_data(files_CT, file_NM=None)


