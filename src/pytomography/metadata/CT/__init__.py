"""Projection metadata of CT scans: the geometry of every view.

* :class:`~pytomography.metadata.CT.CTGen3ProjMeta`: a third generation (fan beam, cylindrical detector) scan,
  helical or axial, as :func:`~pytomography.io.CT.read_dicom_ct_pd` reads it from DICOM-CT-PD.
* :class:`~pytomography.metadata.CT.CTConeBeamFlatPanelProjMeta`: a cone-beam scan with a flat panel detector, as
  :func:`~pytomography.io.CT.read_nikon` reads it from a Nikon XTek scan.

The object grid is an :class:`~pytomography.metadata.ObjectMeta`, as for every modality.
"""
from .ct_conebeam_flatpanel_metadata import CTConeBeamFlatPanelProjMeta
from .ct_gen3_metadata import CTGen3ProjMeta