:py:mod:`pytomography.io.shared.dicom`
======================================

.. py:module:: pytomography.io.shared.dicom


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.shared.dicom._get_affine_multifile
   pytomography.io.shared.dicom.open_multifile
   pytomography.io.shared.dicom.compute_max_slice_loc_multifile
   pytomography.io.shared.dicom.compute_slice_thickness_multifile
   pytomography.io.shared.dicom.align_images_affine



.. py:function:: _get_affine_multifile(files)

   Computes an affine matrix corresponding the coordinate system of a CT DICOM file. Note that since CT scans consist of many independent DICOM files, ds corresponds to an individual one of these files. This is why the maximum z value is also required (across all seperate independent DICOM files).

   :param ds: DICOM dataset of CT data
   :type ds: Dataset
   :param max_z: Maximum value of z across all axial slices that make up the CT scan
   :type max_z: float

   :returns: Affine matrix corresponding to CT scan.
   :rtype: np.array


.. py:function:: open_multifile(files)

   Given a list of seperate DICOM files, opens them up and stacks them together into a single CT image.

   :param files: List of CT DICOM filepaths corresponding to different z slices of the same scan.
   :type files: Sequence[str]

   :returns: CT scan in units of Hounsfield Units at the effective CT energy.
   :rtype: np.array


.. py:function:: compute_max_slice_loc_multifile(files)

   Obtains the maximum z-location from a list of DICOM slice files

   :param files: List of DICOM filepaths corresponding to different z slices of the same scan.
   :type files: Sequence[str]

   :returns: Maximum z location
   :rtype: float


.. py:function:: compute_slice_thickness_multifile(files)

   Compute the slice thickness for files that make up a scan. Though this information is often contained in the DICOM file, it is sometimes inconsistent with the ImagePositionPatient attribute, which gives the true location of the slices.

   :param files: List of DICOM filepaths corresponding to different z slices of the same scan.
   :type files: Sequence[str]

   :returns: Slice thickness of the scan
   :rtype: float


.. py:function:: align_images_affine(im_fixed, im_moving, affine_fixed, affine_moving, cval=0)


