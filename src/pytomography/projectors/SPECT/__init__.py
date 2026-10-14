"""SPECT system matrices, which project by rotating the object and summing it along the detector's normal.

* :class:`SPECTSystemMatrix`: parallel-hole cameras (dual-head and others), with attenuation, PSF and other
  transforms;
* :class:`MonteCarloHybridSPECTSystemMatrix`: a SPECTSystemMatrix whose forward projection is a Monte Carlo
  simulation;
* :class:`StarGuideSystemMatrix`: the GE StarGuide's 12 swivelling heads.
"""
from .dualhead_system_matrix import SPECTSystemMatrix, MonteCarloHybridSPECTSystemMatrix
from .starguide_system_matrix import StarGuideSystemMatrix