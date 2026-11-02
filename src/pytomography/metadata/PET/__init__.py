"""PET metadata, all lengths in mm. The object grid is a plain :class:`~pytomography.metadata.ObjectMeta`.

* :class:`PETLMProjMeta`: list mode events, the scanner and the sensitivity weights;
* :class:`PETSinogramPolygonProjMeta`: sinograms of a polygonal scanner;
* :class:`PETTOFMeta`: time of flight binning.
"""
from .petlm_metadata import PETLMProjMeta
from .pet_tof_metadata import PETTOFMeta
from .pet_sinogram_metadata import PETSinogramPolygonProjMeta