r"""This module contains transforms; operators that map within the same vector space; such operators are used to build the system matrix. For example, the ``SPECTAttenuationTransform`` is an operator :math:`A:\mathbb{U} \to \mathbb{U}` which adjusts each voxel in an image based on the probability of it being attenuated before it reaches a detector at the :math:`+x` axis. As another example, the ``CutOffTransform`` is an operator :math:`B:\mathbb{V} \to \mathbb{V}` which sets all pixels in image space equal to zero which exist beyond the detector boundaries. Since operators are often used in reconstruction algorithms, their transpose also must be implemented """
from .transform import Transform
from .SPECT.atteunation import SPECTAttenuationTransform
from .SPECT.psf import SPECTPSFTransform
from .SPECT.cutoff import CutOffTransform
from .PET.attenuation import PETAttenuationTransform
from .PET.psf import PETPSFTransform
