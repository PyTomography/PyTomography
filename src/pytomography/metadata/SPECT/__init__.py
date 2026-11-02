"""SPECT metadata, all lengths in mm.

* :class:`SPECTObjectMeta`: the object grid, padded for the rotate-and-sum projectors;
* :class:`SPECTProjMeta`: the projection space of a parallel-hole camera (angles, radii, energy windows...);
* :class:`StarGuideProjMeta`: the projection space of the GE StarGuide;
* :class:`EnergyWindow`: an energy window, with its photon energy;
* :class:`SPECTPSFMeta`: the collimator-detector response, for
  :class:`~pytomography.transforms.SPECT.SPECTPSFTransform`.
"""
from .spect_metadata import EnergyWindow, SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta
from .starguide_metadata import StarGuideProjMeta
