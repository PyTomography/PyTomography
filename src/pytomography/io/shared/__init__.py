"""The image writers, shared by every modality.

* :func:`~pytomography.io.save_dicom`: a DICOM series (NM, PT or CT);
* :func:`~pytomography.io.save_nifti`: a NIfTI-1 file.

Both are also in :mod:`pytomography.io`, where they are usually imported from; they live in
:mod:`pytomography.io.shared.output`.
"""

from .output import save_nifti, save_dicom
