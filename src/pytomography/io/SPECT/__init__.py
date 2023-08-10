"""Input/output functions for the SPECT imaging modality. Currently, the data types supported are SIMIND and DICOM files.
"""

from .simind import get_projections, get_atteuation_map, get_scatter_from_TEW
from .dicom import get_projections, get_attenuation_map_from_file, get_attenuation_map_from_CT_slices, get_scatter_from_TEW, get_blank_below_above, get_psfmeta_from_scanner_params, open_CT_file, get_affine_matrix, CT_to_mumap