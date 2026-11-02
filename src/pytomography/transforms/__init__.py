r"""Transforms: the operations in object space and in projection space that system matrices are built from.

* :mod:`pytomography.transforms.SPECT`: attenuation, the collimator-detector response (PSF), a projection mask and
  an additive term, for :class:`~pytomography.projectors.SPECT.SPECTSystemMatrix`;
* :mod:`pytomography.transforms.shared`: a Gaussian filter (the PET resolution model), rotation, the kernel method
  and motion, for any modality.

:class:`Transform` is the base class of all of them.
"""
from .transform import Transform
