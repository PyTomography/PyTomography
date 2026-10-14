"""Object-to-object transforms for any modality.

* :class:`GaussianFilter`: a Gaussian blur of a given FWHM in mm (the PET resolution model);
* :class:`RotationTransform`: rotation about the z axis;
* :class:`KEMTransform`: the kernel matrix of the kernel method;
* :class:`DVFMotionTransform`: motion by a deformation vector field, for motion correction.
"""
from .kem import KEMTransform
from .filters import GaussianFilter
from .spatial import RotationTransform
from .motion import DVFMotionTransform