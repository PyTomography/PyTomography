"""Input/output functions for the SPECT imaging modality. Currently, the data types supported are SIMIND and DICOM files.
"""

from .simind import get_projections, get_attenuation_map
from .dicom import get_projections, get_attenuation_map_from_file, get_attenuation_map_from_CT_slices, get_energy_window_scatter_estimate, get_psfmeta_from_scanner_params, CT_to_mumap
from .shared import subsample_amap, subsample_projections_and_modify_metadata