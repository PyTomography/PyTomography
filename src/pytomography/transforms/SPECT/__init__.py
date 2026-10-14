"""The transforms of :class:`~pytomography.projectors.SPECT.SPECTSystemMatrix`.

Object-to-object (applied to the object rotated for each view):

* :class:`SPECTAttenuationTransform`: attenuation towards the detector, from a map in 1/mm;
* :class:`SPECTPSFTransform`: the distance-dependent collimator-detector response.

Projection-to-projection:

* :class:`CutOffTransform`: zero outside a mask of the detector's field of view;
* :class:`AdditiveTermTransform`: adds a term to the projections.
"""
from .attenuation import SPECTAttenuationTransform
from .psf import SPECTPSFTransform
from .cutoff import CutOffTransform
from .additive_term import AdditiveTermTransform