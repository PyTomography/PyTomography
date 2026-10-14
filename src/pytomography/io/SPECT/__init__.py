"""Reading SPECT data into PyTomography's standard form.

Every reader returns ``(projections, object_meta, proj_meta)``: projections [W, θ, r, z] (W energy windows, z foot
to head), an object grid (:class:`~pytomography.metadata.SPECT.SPECTObjectMeta`, mm; placed in the patient's frame
when the data says where the patient is, else centred on the scanner) and the projection metadata
(:class:`~pytomography.metadata.SPECT.SPECTProjMeta`) with every energy window, the frame duration, the start time and
the nuclide.

Readers:

* :func:`read_dicom`: a DICOM NM file of a dual-head camera (Siemens, GE, Mediso...);
* :func:`read_dicom_beds`: several bed positions, and :func:`stitch_beds` to join their reconstructions;
* :func:`read_starguide`: the head files of a GE StarGuide acquisition;
* :func:`read_simind`: SIMIND Monte Carlo projections.

For the system matrix and the corrections:

* :func:`collimator_psf` and :func:`simind_psf`: the collimator's resolution, for
  :class:`~pytomography.transforms.SPECT.SPECTPSFTransform`;
* :func:`energy_window_scatter`: a dual or triple energy window scatter estimate;
* :func:`fov_mask`: the detector's field of view in the projections.

Attenuation maps and masks come from :func:`pytomography.io.attenuation_map` and :func:`pytomography.io.load_mask`.
"""

from .dicom import read_dicom, read_dicom_beds, stitch_beds
from .starguide import read_starguide
from .simind import read_simind, simind_psf
from ._energy import energy_window_scatter, collimator_psf, fov_mask
