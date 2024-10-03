import os
from pytomography.io.SPECT import simind

path = '/disk1/pytomography_tutorial_data/simind_tutorial/'

# SIMIND simulation files for simulation of only a liver region
photopeak_path_liver = os.path.join(path, 'multi_projections', 'liver', 'photopeak.h00')
upperscatter_path_liver = os.path.join(path, 'multi_projections', 'liver', 'lowerscatter.h00')
lowerscatter_path_liver = os.path.join(path, 'multi_projections', 'liver', 'upperscatter.h00')

# SIMIND simulation files for simulation of all anatomical regions
organs = ['bkg', 'liver', 'l_lung', 'r_lung', 'l_kidney', 'r_kidney','salivary', 'bladder']
headerfiles = [os.path.join(path, 'multi_projections', organ, 'photopeak.h00') for organ in organs]
headerfiles_lower = [os.path.join(path, 'multi_projections', organ, 'lowerscatter.h00') for organ in organs]
headerfiles_upper = [os.path.join(path, 'multi_projections', organ, 'upperscatter.h00') for organ in organs]

# 1. To load multiple energy windows for one region, we provide the paths as a list
photopeak_liver = simind.get_projections([photopeak_path_liver, upperscatter_path_liver, lowerscatter_path_liver])
print(photopeak_liver.shape)

# 2. To load a single energy window and combine multiple regions, we provide the list as [[<list>]]. This will scale each SIMIND set of projections by the corresponding activities (combining projections together):
activities = [2500, 450, 7, 7, 100, 100, 20, 90] # MBq
photopeak_allregions = simind.get_projections([headerfiles], weights=activities)
print(photopeak_allregions.shape)

# 3. To load multiple energy windows for multiple regions, we provide the list as [[<list>], [<list>], [<list>]]:
activities = [2500, 450, 7, 7, 100, 100, 20, 90] # MBq
projections_allregions = simind.get_projections([headerfiles, headerfiles_lower, headerfiles_upper], weights=activities)
print(projections_allregions.shape)