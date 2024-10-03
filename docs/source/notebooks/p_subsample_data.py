from pytomography.io.SPECT import simind
from pytomography.io.SPECT.shared import subsample_metadata, subsample_amap, subsample_projections
import os

# CHANGE THIS TO WHERE YOU DOWNLOADED THE TUTORIAL DATA
PATH = '/disk1/pytomography_tutorial_data'

data_path = os.path.join(PATH, 'simind_tutorial', 'lu177_SYME_jaszak')
photopeak_path = os.path.join(data_path,'tot_w4.h00')
attenuation_path = os.path.join(data_path, 'amap.h00')

object_meta, proj_meta = simind.get_metadata(photopeak_path)
photopeak = simind.get_projections(photopeak_path)
amap = simind.get_attenuation_map(attenuation_path)

# Projection data can be subsampled in pixels (via average pooling) and angles via
photopeak_subsampled = subsample_projections(
    projections = photopeak,
    N_pixel = 2, # from 128x128 to 64x64
    N_angle = 2, # every 2nd angle
    N_angle_start = 1, # angle to start at
)

# If you subsample the photopeak, you also have to adjust the metadata:
object_meta_subsampled, proj_meta_subsampled = subsample_metadata(
    object_meta = object_meta,
    proj_meta = proj_meta,
    N_pixel = 2,
    N_angle = 2,
    N_angle_start = 1,
)

# And the attenuation map (if you use attenuation correction):
attenuation_map = subsample_amap(
    amap = amap,
    N_pixel = 2,
)

# You can then proceed to use the updated metadata/projections to build the system matrix and likelihood as normal