"""Reading SPECT data into PyTomography's standard form: projections [W, θ, r, z] (z foot to head), an object grid
in the patient's frame (``SPECTObjectMeta``, mm) and the projection metadata with every energy window, the frame
duration, the start time and the nuclide."""

from .dicom import read_dicom, read_dicom_beds, stitch_beds
from .starguide import read_starguide
from .simind import read_simind, simind_psf
from ._energy import energy_window_scatter, collimator_psf, fov_mask
