:py:mod:`pytomography.io.dicom`
===============================

.. py:module:: pytomography.io.dicom

.. autoapi-nested-parse::

   Note: This module is still being built and is not yet finished.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   pytomography.io.dicom.get_radii_and_angles
   pytomography.io.dicom.dicom_projections_to_data
   pytomography.io.dicom.dicom_MEW_to_data
   pytomography.io.dicom.get_HU2mu_coefficients
   pytomography.io.dicom.bilinear_transform
   pytomography.io.dicom.get_affine_spect
   pytomography.io.dicom.get_affine_CT
   pytomography.io.dicom.dicom_CT_to_data
   pytomography.io.dicom.get_SPECT_recon_algorithm_dicom



.. py:function:: get_radii_and_angles(ds)

   Gets projections with corresponding radii and angles corresponding to projection data from a DICOM file.

   :param ds: pydicom dataset object.
   :type ds: Dataset

   :returns: Required image data for reconstruction.
   :rtype: (torch.tensor[1,Ltheta, Lr, Lz], np.array, np.array)


.. py:function:: dicom_projections_to_data(file)

   Obtains ObjectMeta, ImageMeta, and projections from a .dcm file.

   :param file: Path to the .dcm file
   :type file: str

   :returns: Required information for reconstruction in PyTomography.
   :rtype: (ObjectMeta, ImageMeta, torch.Tensor[1, Ltheta, Lr, Lz])


.. py:function:: dicom_MEW_to_data(file, type='DEW')


.. py:function:: get_HU2mu_coefficients(ds, photopeak_window_index = 0)

   Obtains the four coefficients required for the bilinear transformation between Hounsfield Units and linear attenuation coefficient at the photon energy corresponding to the primary window of the given dataset.

   :param ds: DICOM data set of projection data
   :type ds: Dataset
   :param primary_window_index: The energy window corresponding to the photopeak. Defaults to 0.
   :type primary_window_index: int, optional

   :returns: Array of length 4 containins the 4 coefficients required for the bilinear transformation.
   :rtype: np.array


.. py:function:: bilinear_transform(arr, a1, b1, a2, b2)

   Converts an array of Hounsfield Units into linear attenuation coefficient using the bilinear transformation :math:`f(x)=a_1x+b_1` for positive :math:`x` and :math:`f(x)=a_2x+b_2` for negative :math:`x`.

   :param arr: Array to be transformed using bilinear transformation
   :type arr: np.array
   :param a1: Bilinear slope for negative input values
   :type a1: float
   :param b1: Bilinear intercept for negative input values
   :type b1: float
   :param a2: Bilinear slope for positive input values
   :type a2: float
   :param b2: Bilinear intercept for positive input values
   :type b2: float

   :returns: Transformed array.
   :rtype: np.array


.. py:function:: get_affine_spect(ds)

   Computes an affine matrix corresponding the coordinate system of a SPECT DICOM file.

   :param ds: DICOM dataset of projection data
   :type ds: Dataset

   :returns: Affine matrix.
   :rtype: np.array


.. py:function:: get_affine_CT(ds, max_z)

   Computes an affine matrix corresponding the coordinate system of a CT DICOM file. Note that since CT scans consist of many independent DICOM files, ds corresponds to an individual one of these files. This is why the maximum z value is also required (across all seperate independent DICOM files).

   :param ds: DICOM dataset of CT data
   :type ds: Dataset
   :param max_z: Maximum value of z across all axial slices that make up the CT scan
   :type max_z: float

   :returns: Affine matrix corresponding to CT scan.
   :rtype: np.array


.. py:function:: dicom_CT_to_data(files_CT, file_NM, photopeak_window_index = 0)

   Converts a sequence of DICOM CT files (corresponding to a single scan) into a torch.Tensor object usable as an attenuation map in PyTomography. This is primarily intended for opening pre-reconstructed CT data such that it can be used as an attenuation map during PET/SPECT reconstruction.

   :param files_CT: List of all files corresponding to an individual CT scan
   :type files_CT: Sequence[str]
   :param file_NM: File corresponding to raw PET/SPECT data (required to align CT with projections)
   :type file_NM: str
   :param photopeak_window_index: Index corresponding to photopeak in projection data. Defaults to 0.
   :type photopeak_window_index: int, optional

   :returns: Tensor of shape [Lx, Ly, Lz] corresponding to attenuation map.
   :rtype: torch.Tensor


.. py:function:: get_SPECT_recon_algorithm_dicom(projections_file, atteunation_files = None, use_psf = False, scatter_type = None, prior = None, recon_algorithm_class = OSEMOSL, object_initial = None)

   Helper function to quickly create reconstruction algorithm given SPECT DICOM files and CT dicom files.

   :param projections_file: DICOM filepath corresponding to SPECT data.
   :type projections_file: str
   :param atteunation_files: DICOM filepaths corresponding to CT data. If None, then atteunation correction is not used. Defaults to None.
   :type atteunation_files: Sequence[str], optional
   :param use_psf: Whether or not to use PSF modeling. Defaults to False.
   :type use_psf: bool, optional
   :param scatter_type: Type of scatter correction used in reconstruction. Defaults to None.
   :type scatter_type: str | None, optional
   :param prior: Bayesian Prior used in reconstruction algorithm. Defaults to None.
   :type prior: Prior, optional
   :param recon_algorithm_class: Type of reconstruction algorithm used. Defaults to OSEMOSL.
   :type recon_algorithm_class: nn.Module, optional
   :param object_initial: Initial object used in reconstruction. If None, defaults to all ones. Defaults to None.
   :type object_initial: torch.Tensor | None, optional

   :raises Exception: If not able to compute relevant PSF parameters from DICOM data and corresponding data tables.

   :returns: Reconstruction algorithm used.
   :rtype: OSML


