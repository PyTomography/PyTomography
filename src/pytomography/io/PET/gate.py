from __future__ import annotations
from collections.abc import Sequence
import pytomography
from pytomography.metadata import ObjectMeta
import torch
import numpy as np
import numpy.linalg as npl
import uproot
import nibabel as nib
from scipy.ndimage import affine_transform
from ..shared import get_header_value, get_attenuation_map_interfile
from .shared import listmode_to_sinogram, sinogram_to_listmode, get_detector_ids_from_trans_axial_ids, get_axial_trans_ids_from_info, get_scanner_LUT, smooth_randoms_sinogram, randoms_sinogram_to_sinogramTOF

def get_aligned_attenuation_map(
    headerfile: str,
    object_meta: ObjectMeta
    ) -> torch.tensor:
    """Returns an aligned attenuation map in units of inverse mm for reconstruction. This assumes that the attenuation map shares the same center point with the reconstruction space.

    Args:
        headerfile (str): Filepath to the header file of the attenuation map
        object_meta (ObjectMeta): Object metadata providing spatial information about the reconstructed dimensions.

    Returns:
        torch.Tensor: Aligned attenuation map
    """
    amap = get_attenuation_map_interfile(headerfile).cpu().numpy()
    # Load metadata
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    dx = get_header_value(headerdata, 'scaling factor (mm/pixel) [1]', np.float32)
    dy = get_header_value(headerdata, 'scaling factor (mm/pixel) [2]', np.float32)
    dz = get_header_value(headerdata, 'scaling factor (mm/pixel) [3]', np.float32)
    dr_amap = (dx, dy, dz)
    shape_amap = amap.shape
    object_origin_amap = (- np.array(shape_amap) / 2 + 0.5) * (np.array(dr_amap))
    dr = object_meta.dr
    shape = object_meta.shape
    object_origin = object_origin = (- np.array(shape) / 2 + 0.5) * (np.array(dr))
    M_PET = np.array([
        [dr[0],0,0,object_origin[0]],
        [0,dr[1],0,object_origin[1]],
        [0,0,dr[2],object_origin[2]],
        [0,0,0,1]
    ])
    M_CT = np.array([
        [dr_amap[0],0,0,object_origin_amap[0]],
        [0,dr_amap[1],0,object_origin_amap[1]],
        [0,0,dr_amap[2],object_origin_amap[2]],
        [0,0,0,1]
    ])
    amap = affine_transform(amap, npl.inv(M_CT)@M_PET, output_shape = shape, order=1)
    amap = torch.tensor(amap, device=pytomography.device) / 10 # to mm^-1
    return amap

def get_detector_info(
    path: str,
    init_volume_name: str = 'crystal',
    mean_interaction_depth: float = 0,
    min_rsector_difference: int = 0
    ) -> dict:
    """Generates detector geometry information dictionary from GATE macro file

    Args:
        path (str): Path to GATE macro file that defines geometry: should end in ".mac"
        init_volume_name (str, optional): Initial volume name in the GATE file. Defaults to 'crystal'.
        mean_interaction_depth (float, optional): Mean interaction depth of photons within crystal. Defaults to 0.
        min_rsector_difference (int, optional): Minimum r_sector difference for retained events. Defaults to 0.

    Returns:
        dict: PET geometry information dictionary
    """
    with open(path) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    parents = ['crystal', 'submodule', 'module', 'rsector', 'world']
    positions = []
    for parent in parents:
        try:
            x = get_header_value(headerdata, f'/gate/{parent}/placement/setTranslation', split_substr=None, split_idx=1)
            y = get_header_value(headerdata, f'/gate/{parent}/placement/setTranslation', split_substr=None, split_idx=2)
            z = get_header_value(headerdata, f'/gate/{parent}/placement/setTranslation', split_substr=None, split_idx=3)
        except:
            x = y = z = 0
        positions.append([x,y,z])
    positions = np.array(positions)
    x_crystal, y_crystal, z_crystal = positions.sum(axis=0)
    x_crystal = np.array([x_crystal])
    y_crystal = np.array([y_crystal])
    z_crystal = np.array([z_crystal])
    # Get edges of crystal (assume original in +X) TODO: fix
    info = {}
    info['min_rsector_difference'] = min_rsector_difference
    info['crystal_length'] = get_header_value(headerdata, f'/gate/{init_volume_name}/geometry/setXLength', split_substr=None, split_idx=1) 
    info['radius'] = x_crystal[0] - info['crystal_length']/2 + mean_interaction_depth
    for parent in parents:
        repeaters = get_header_value(headerdata, f'/gate/{parent}/repeaters/insert', split_substr=None, split_idx=1, dtype=str, return_all=True)
        if not(repeaters):
            if parent=='submodule':
                info['submoduleAxialNr'] = 1
                info['submoduleAxialSpacing'] = 0
                info['submoduleTransNr'] = 1
                info['submoduleTransSpacing'] = 0
            continue
        for repeater in repeaters:
            if repeater=='cubicArray':
                repeat_numbers = np.array([get_header_value(headerdata, f'/gate/{parent}/{repeater}/setRepeatNumber{coord}', split_substr=None, split_idx=1) for coord in ['X', 'Y', 'Z']])
                repeat_vector = np.array([get_header_value(headerdata, f'/gate/{parent}/{repeater}/setRepeatVector', split_substr=None, split_idx=i) for i in range(1,4)])
                idx_trans = np.argmax(repeat_numbers[:2])
                info[f'{parent}TransNr'] = int(repeat_numbers[idx_trans])
                info[f'{parent}TransSpacing'] = repeat_vector[idx_trans]
                info[f'{parent}AxialNr'] = int(repeat_numbers[2])
                info[f'{parent}AxialSpacing'] = repeat_vector[2]
            elif repeater=='linear':
                repeat_number = get_header_value(headerdata, f'/gate/{parent}/{repeater}/setRepeatNumber', split_substr=None, split_idx=1)
                repeat_vector = [get_header_value(headerdata, f'/gate/{parent}/{repeater}/setRepeatVector', split_substr=None, split_idx=i) for i in range(1,4)]
                # append to axial/trans information
                info[f'{parent}AxialNr'] = int(repeat_number)
                info[f'{parent}AxialSpacing'] = repeat_vector[2]
            elif repeater=='ring':
                repeat_number = get_header_value(headerdata, f'/gate/{parent}/{repeater}/setRepeatNumber', split_substr=None, split_idx=1)
                # Repeat number for a ring is in the axial direction
                info[f'{parent}TransNr'] = int(repeat_number)
                info[f'{parent}AxialNr'] = 1
    info['NrCrystalsPerRing'] = info['crystalTransNr'] * info['moduleTransNr'] * info['submoduleTransNr'] * info['rsectorTransNr']
    info['NrRings'] = info['crystalAxialNr'] * info['submoduleAxialNr'] * info['moduleAxialNr'] * info['rsectorAxialNr']
    info['firstCrystalAxis'] = 1
    return info

def get_axial_trans_ids_from_ROOT(
    f: object,
    info: dict,
    j: int = None,
    substr: str = 'Coincidences') -> Sequence[torch.Tensor]:
    """Obtain transaxial and axial IDS (for crystals, submodules, modules, and rsectors) corresponding to each listmode event in an opened ROOT file

    Args:
        f (object): Opened ROOT file    
        info (dict): PET geometry information dictionary
        j (int, optional): Which of the detectors to consider in a coincidence event OR which detector to consider for a single (None). Defaults to None.
        substr (str, optional): Whether to consider coincidences or singles. Defaults to 'Coincidences'.

    Returns:
        Sequence[torch.Tensor]: Sequence of IDs (transaxial/axial) for all components (crystals, submodules, modules, and rsectors)
    """
    if j is None:
        idx_str = ''
    else:
        idx_str = f'{j+1}'
    ids_rsector = torch.tensor(f[substr][f'rsectorID{idx_str}'].array(library="np"))
    ids_module = torch.tensor(f[substr][f'moduleID{idx_str}'].array(library="np"))
    ids_submodule = torch.tensor(f[substr][f'submoduleID{idx_str}'].array(library="np"))
    ids_crystal = torch.tensor(f[substr][f'crystalID{idx_str}'].array(library="np"))
    ids_trans_rsector = ids_rsector % info['rsectorTransNr']
    ids_axial_rsector = ids_rsector // info['rsectorTransNr']
    ids_trans_module = ids_module % info['moduleTransNr']
    ids_axial_module = ids_module // info['moduleTransNr']
    ids_trans_submodule = ids_submodule % info['submoduleTransNr']
    ids_axial_submodule = ids_submodule // info['submoduleTransNr']
    ids_trans_crystal = ids_crystal % info['crystalTransNr']
    ids_axial_crystal = ids_crystal // info['crystalTransNr']
    return ids_trans_crystal, ids_axial_crystal, ids_trans_submodule, ids_axial_submodule, ids_trans_module, ids_axial_module, ids_trans_rsector, ids_axial_rsector

def get_detector_ids_from_root(
    paths: Sequence[str],
    info: dict,
    tof_meta = None,
    substr: str = 'Coincidences',
    include_randoms: bool = True,
    include_scatters: bool = True,
    randoms_only: bool = False,
    scatters_only: bool = False
    ) -> torch.Tensor:
    """Obtain detector IDs corresponding to each listmode event in a set of ROOT files

    Args:
        paths (Sequence[str]): List of ROOT files to consider
        info (dict): PET geometry information dictionary
        tof_meta (PETTOFMeta, optional): PET time of flight metadata for binning. If none, then TOF is not considered Defaults to None.
        substr (str, optional): Name of events to consider in the ROOT file. Defaults to 'Coincidences'.
        include_randoms (bool, optional): Whether or not to include random events in the returned listmode events. Defaults to True.
        include_scatters (bool, optional): Whether or not to include scatter events in the returned listmode events. Defaults to True.
        randoms_only (bool, optional): Flag to return only random events. Defaults to False.
        scatters_only (bool, optional): Flag to return only scatter events. Defaults to False.

    Returns:
        torch.Tensor: Tensor of shape [N_events,2] (non-TOF) or [N_events,3] (TOF)
    """
    if tof_meta is not None:
        TOF_bin_edges = tof_meta.bin_edges
    detector_ids_trio = [[],[],[]]
    for i,path in enumerate(paths):
        print(i)
        with uproot.open(path) as f:
            N_events = f[substr]['sourcePosX1'].array(library='np').shape[0]
            valid_indices = torch.ones(N_events).to(torch.bool)
            if not(include_randoms) or randoms_only or not(include_scatters) or scatters_only:
                xs1 = torch.tensor(f[substr]['sourcePosX1'].array(library='np'))
                xs2 = torch.tensor(f[substr]['sourcePosX2'].array(library='np'))
                ys1 = torch.tensor(f[substr]['sourcePosY1'].array(library='np'))
                ys2 = torch.tensor(f[substr]['sourcePosY2'].array(library='np'))
                zs1 = torch.tensor(f[substr]['sourcePosZ1'].array(library='np'))
                zs2 = torch.tensor(f[substr]['sourcePosZ2'].array(library='np'))
                random_indices = ~((xs1==xs2)*(ys1==ys2)*(zs1==zs2))
            if not(include_scatters) or scatters_only:
                scatter_raleigh_1 = torch.tensor(f['Coincidences']['RayleighPhantom1'].array(library='np'))
                scatter_raleigh_2 = torch.tensor(f['Coincidences']['RayleighPhantom2'].array(library='np'))
                scatter_compton_1 = torch.tensor(f['Coincidences']['comptonPhantom1'].array(library='np'))
                scatter_compton_2 = torch.tensor(f['Coincidences']['comptonPhantom2'].array(library='np'))
                scatter_indices = (scatter_raleigh_1+scatter_raleigh_2+scatter_compton_1+scatter_compton_2).to(torch.bool)
                
            # Adjust indices we're looking for based on the events we want
            if randoms_only:
                valid_indices *= random_indices
            elif scatters_only:
                # Only include scatter events that arent from randoms
                valid_indices *= (scatter_indices)*(~random_indices)
            else:
                if not(include_randoms):
                    valid_indices *= ~random_indices
                if not(include_scatters):
                    valid_indices *= ~scatter_indices
            for j in range(2):
                ids_trans_crystal, ids_axial_crystal, ids_trans_submodule, ids_axial_submodule, ids_trans_module, ids_axial_module, ids_trans_rsector, ids_axial_rsector = get_axial_trans_ids_from_ROOT(f, info, j, substr)
                detector_ids = get_detector_ids_from_trans_axial_ids(ids_trans_crystal, ids_trans_submodule, ids_trans_module, ids_trans_rsector, ids_axial_crystal, ids_axial_submodule, ids_axial_module, ids_axial_rsector, info)
                detector_ids = detector_ids[valid_indices]
                detector_ids_trio[j].append(detector_ids.to(torch.int32))
            if tof_meta is not None:
                t1 = f[substr]['time1'].array(library='np')
                t2 = f[substr]['time2'].array(library='np')
                tof_pos = 1e12*(t2 - t1) * 0.15 # ps to mm
                detector_id = np.digitize(-tof_pos, TOF_bin_edges) - 1
                # First see if only binning scatters/randoms
                detector_id = detector_id[valid_indices]
                detector_ids_trio[2].append(torch.tensor(detector_id))
    
    if tof_meta is not None:
        return torch.vstack([
            torch.concatenate(detector_ids_trio[0]),
            torch.concatenate(detector_ids_trio[1]),
            torch.concatenate(detector_ids_trio[2])]).T
    else:
        return torch.vstack([
            torch.concatenate(detector_ids_trio[0]),
            torch.concatenate(detector_ids_trio[1])]).T
        
def get_symmetry_histogram_from_ROOTfile(
    f: object,
    info: dict,
    substr: str = 'Coincidences',
    include_randoms: bool = True
    ) -> torch.Tensor:
    """Obtains a histogram that exploits symmetries when computing normalization factors from calibration ROOT scans

    Args:
        f (object): Opened ROOT file
        info (dict): PET geometry information dictionary
        substr (str, optional): Name of events to consider in ROOT file. Defaults to 'Coincidences'.
        include_randoms (bool, optional): Whether or not to include random events from data. Defaults to True.

    Returns:
        torch.Tensor: Symmetry histogram
    """
    ids1_trans_crystal, ids1_axial_crystal, ids1_trans_submodule, ids1_axial_submodule, ids1_trans_module, ids1_axial_module, ids1_trans_rsector, ids1_axial_rsector = get_axial_trans_ids_from_ROOT(f, info, 0, substr= substr)
    ids2_trans_crystal, ids2_axial_crystal, ids2_trans_submodule, ids2_axial_submodule, ids2_trans_module, ids2_axial_module, ids2_trans_rsector, ids2_axial_rsector = get_axial_trans_ids_from_ROOT(f, info, 1, substr= substr)
    ids_trans_crystal = torch.vstack([ids1_trans_crystal, ids2_trans_crystal])
    ids_axial_crystal = torch.vstack([ids1_axial_crystal, ids2_axial_crystal])
    ids_axial_submodule = torch.vstack([ids1_axial_submodule, ids2_axial_submodule])
    ids_axial_module = torch.vstack([ids1_axial_module, ids2_axial_module])
    ids_trans_rsector = torch.vstack([ids1_trans_rsector, ids2_trans_rsector])
    # Make sure smallest detector ID always comes first
    detector_ids1 = get_detector_ids_from_trans_axial_ids(ids1_trans_crystal, ids1_trans_submodule, ids1_trans_module, ids1_trans_rsector, ids1_axial_crystal, ids1_axial_submodule, ids1_axial_module, ids1_axial_rsector, info)
    detector_ids2 = get_detector_ids_from_trans_axial_ids(ids2_trans_crystal, ids2_trans_submodule, ids2_trans_module, ids2_trans_rsector, ids2_axial_crystal, ids2_axial_submodule, ids2_axial_module, ids2_axial_rsector, info)
    detector_ids = torch.vstack([detector_ids1, detector_ids2])
    idx_min = detector_ids.min(axis=0).indices
    idx_max = detector_ids.max(axis=0).indices
    # Compute histogram quantities
    ids_delta_axial_submodule =  (
        torch.take_along_dim(ids_axial_submodule, idx_max.unsqueeze(0), 0) \
      - torch.take_along_dim(ids_axial_submodule, idx_min.unsqueeze(0), 0) )\
      + (info['submoduleAxialNr'] - 1)
    ids_delta_axial_module =  (
        torch.take_along_dim(ids_axial_module, idx_max.unsqueeze(0), 0) \
      - torch.take_along_dim(ids_axial_module, idx_min.unsqueeze(0), 0) )\
      + (info['moduleAxialNr'] - 1)
    ids_delta_trans_rsector =  (
        torch.take_along_dim(ids_trans_rsector, idx_max.unsqueeze(0), 0) \
      - torch.take_along_dim(ids_trans_rsector, idx_min.unsqueeze(0), 0) )\
      % info['rsectorTransNr'] 
    histo = torch.vstack([
        torch.take_along_dim(ids_axial_crystal, idx_min.unsqueeze(0), 0),
        torch.take_along_dim(ids_axial_crystal, idx_max.unsqueeze(0), 0),
        torch.take_along_dim(ids_trans_crystal, idx_min.unsqueeze(0), 0),
        torch.take_along_dim(ids_trans_crystal, idx_max.unsqueeze(0), 0),
        ids_delta_axial_submodule,
        ids_delta_axial_module,
        ids_delta_trans_rsector
    ]).T
    if include_randoms:
        xs1 = torch.tensor(f[substr]['sourcePosX1'].array(library='np'))
        xs2 = torch.tensor(f[substr]['sourcePosX2'].array(library='np'))
        ys1 = torch.tensor(f[substr]['sourcePosY1'].array(library='np'))
        ys2 = torch.tensor(f[substr]['sourcePosY2'].array(library='np'))
        zs1 = torch.tensor(f[substr]['sourcePosZ1'].array(library='np'))
        zs2 = torch.tensor(f[substr]['sourcePosZ2'].array(library='np'))
        same_location_idxs = (xs1==xs2)*(ys1==ys2)*(zs1==zs2)
        return histo[same_location_idxs]
    else:
        return histo

def get_symmetry_histogram_all_combos(info: dict) -> torch.Tensor:
    """Obtains the symmetry histogram for detector sensitivity corresponding to all possible detector pair combinations

    Args:
        info (dict): PET geometry information dictionary

    Returns:
        torch.Tensor: Histogram corresponding to all possible detector pair combinations. This simply counts the number of detector pairs in each bin of the histogram.
    """
    ids_trans_crystal, ids_axial_crystal, ids_trans_submodule, ids_axial_submodule, ids_trans_module, ids_axial_module, ids_trans_rsector, ids_axial_rsector = get_axial_trans_ids_from_info(info, return_combinations=True, sort_by_detector_ids=True)
    ids_delta_axial_submodule = (ids_axial_submodule[:,1] - ids_axial_submodule[:,0]) + (info['submoduleAxialNr'] - 1)
    ids_delta_axial_module = (ids_axial_module[:,1] - ids_axial_module[:,0]) + (info['moduleAxialNr'] - 1)
    ids_delta_trans_rsector = (ids_trans_rsector[:,1] - ids_trans_rsector[:,0]) % info['rsectorTransNr'] # because of circle
    return torch.vstack([ids_axial_crystal[:,0], ids_axial_crystal[:,1], ids_trans_crystal[:,0], ids_trans_crystal[:,1], ids_delta_axial_submodule, ids_delta_axial_module, ids_delta_trans_rsector]).T

def get_normalization_weights_cylinder_calibration(
    paths: Sequence[str],
    info: dict,
    cylinder_radius: float,
    include_randoms: bool = True,
    ) -> torch.tensor:
    """Function to get sensitivty factor from a cylindrical calibration phantom

    Args:
        paths (Sequence[str]): List of paths corresponding to calibration scan
        info (dict): PET geometry information dictionary
        cylinder_radius (float): Radius of cylindrical phantom used in scan
        include_randoms (bool, optional): Whether or not to include random events from the cylinder calibration. Defaults to True.


    Returns:
        torch.tensor: Sensitivty factor for all possible detector combinations
    """
    # Part 1: Geometry correction factor for non-unform exposure from cylindrical shell
    scanner_LUT = get_scanner_LUT(info)
    all_LOR_ids = torch.combinations(torch.arange(scanner_LUT.shape[0]).to(torch.int32), 2)
    geometric_correction_factor = 1/(torch.sqrt(1-(torch.abs(get_radius(all_LOR_ids, scanner_LUT)) / cylinder_radius )**2) + pytomography.delta)
    # Part 2: Detector sensitivity correction factor (exploits symmetries)
    Nr_crystal_axial_bins = info['crystalAxialNr']
    Nr_crystal_trans_bins = info['crystalTransNr']
    Nr_delta_submodule_axial_bins = info['submoduleAxialNr'] * 2 - 1
    Nr_delta_module_axial_bins = info['moduleAxialNr'] * 2 - 1
    Nr_delta_rsector_trans_bins = info['rsectorTransNr'] # b/c circle
    histo = torch.zeros([Nr_crystal_axial_bins, Nr_crystal_axial_bins, Nr_crystal_trans_bins, Nr_crystal_trans_bins, Nr_delta_submodule_axial_bins, Nr_delta_module_axial_bins, Nr_delta_rsector_trans_bins])
    bin_edges = [torch.arange(x).to(torch.float32)-0.5 for x in [Nr_crystal_axial_bins+1, Nr_crystal_axial_bins+1, Nr_crystal_trans_bins+1, Nr_crystal_trans_bins+1, Nr_delta_submodule_axial_bins+1, Nr_delta_module_axial_bins+1, Nr_delta_rsector_trans_bins+1]]
    for path in paths:
        with uproot.open(path) as f:
            vals = get_symmetry_histogram_from_ROOTfile(f, info, include_randoms=include_randoms)
            histo += torch.histogramdd(vals.to(torch.float32), bin_edges)[0]
    vals_all_pairs = get_symmetry_histogram_all_combos(info)
    N_bins = torch.histogramdd(vals_all_pairs.to(torch.float32), bin_edges)[0]
    # exploits the fact that vals_all_pairs is in order of ascending detector ids
    return (histo/N_bins)[vals_all_pairs[:,0], vals_all_pairs[:,1], vals_all_pairs[:,2], vals_all_pairs[:,3], vals_all_pairs[:,4], vals_all_pairs[:,5], vals_all_pairs[:,6]] * geometric_correction_factor

def get_norm_sinogram_from_listmode_data(
    weights_sensitivity: torch.Tensor,
    info: dict
) -> torch.Tensor:
    """Obtains normalization "sensitivty" sinogram from listmode data

    Args:
        weights_sensitivity (torch.Tensor): Sensitivty weight corresponding to all possible detector pairs
        info (dict): PET geometry information dictionary

    Returns:
        torch.Tensor: PET sinogram
    """
    scanner_LUT = get_scanner_LUT(info)
    all_LOR_ids = torch.combinations(torch.arange(scanner_LUT.shape[0]).to(torch.int32), 2)
    return listmode_to_sinogram(all_LOR_ids, info, weights=weights_sensitivity, normalization=True)

def get_norm_sinogram_from_root_data(
    normalization_paths: Sequence[str],
    info: dict,
    cylinder_radius: float,
    include_randoms: bool =True,
) -> torch.Tensor:
    """Obtain normalization "sensitivity" sinogram directly from ROOT files

    Args:
        normalization_paths (Sequence[str]): Paths to all ROOT files corresponding to calibration scan
        info (dict): PET geometry information dictionary
        cylinder_radius (float): Radius of cylinder used in calibration scan
        include_randoms (bool, optional): Whether or not to include randoms in loaded data. Defaults to True.

    Returns:
        torch.Tensor: PET sinogram
    """
    eta = get_normalization_weights_cylinder_calibration(
        normalization_paths,
        info,
        include_randoms=include_randoms,
        cylinder_radius = cylinder_radius
    )
    return get_norm_sinogram_from_listmode_data(eta, info)


def get_sinogram_from_root_data(
    paths: Sequence[str],
    info: dict,
    include_randoms: bool = True,
    include_scatters: bool = True,
    randoms_only: bool = False,
    scatters_only: bool = False
) -> torch.Tensor:
    """Get PET sinogram directly from ROOT data

    Args:
        paths (Sequence[str]): GATE generated ROOT files
        info (dict): PET geometry information dictionary
        include_randoms (bool, optional): Whether or not to include random events in the sinogram. Defaults to True.
        include_scatters (bool, optional): Whether or not to include scatter events in the sinogram. Defaults to True.
        randoms_only (bool, optional): Flag for only binning randoms. Defaults to False.
        scatters_only (bool, optional): Flag for only binning scatters. Defaults to False.

    Returns:
        torch.Tensor: PET sinogram
    """
    detector_ids = get_detector_ids_from_root(
        paths,
        info,
        include_randoms=include_randoms,
        include_scatters=include_scatters,
        randoms_only=randoms_only,
        scatters_only=scatters_only,
        TOF=False)
    return listmode_to_sinogram(detector_ids, info)

def get_radius(detector_ids: torch.tensor, scanner_LUT: torch.tensor) -> torch.tensor:
    """Gets the radial position of all LORs

    Args:
        detector_ids (torch.tensor): Detector ID pairs corresponding to LORs
        scanner_LUT (torch.tensor): scanner look up table

    Returns:
        torch.tensor: radii of all detector ID pairs provided
    """
    x1, y1, z1 = scanner_LUT[detector_ids[:,0]].T
    x2, y2, z2 = scanner_LUT[detector_ids[:,1]].T
    return torch.where(
        (x1==x2)*(y1==y2),
        torch.sqrt(x1**2+y1**2),
        (x1*y2-y1*x2)/torch.sqrt((x1-x2)**2+(y1-y2)**2)
    )
    
def get_angle(detector_ids: torch.tensor, scanner_LUT: torch.tensor) -> torch.tensor:
    """Gets the angular position of all LORs

    Args:
        detector_ids (torch.tensor): Detector ID pairs corresponding to LORs
        scanner_LUT (torch.tensor): scanner look up table

    Returns:
        torch.tensor: angle of all detector ID pairs provided
    """
    x1, y1, z1 = scanner_LUT[detector_ids[:,0]].T
    x2, y2, z2 = scanner_LUT[detector_ids[:,1]].T
    return torch.where(
        (x1==x2)*(y1==y2),
        torch.inf,
        torch.arccos(torch.abs(x1-x2)/torch.sqrt((x1-x2)**2+(y1-y2)**2))
    )

# Removes all LORs not intersecting with reconstruction cube
def remove_events_out_of_bounds(
    detector_ids: torch.tensor,
    scanner_LUT: torch.tensor,
    object_meta: ObjectMeta
    ) -> torch.tensor:
    r"""Removes all detected LORs outside of the reconstruced volume given by ``object_meta``.

    Args:
        detector_ids (torch.tensor): :math:`N \times 2` (non-TOF) or :math:`N \times 3` (TOF) tensor that provides detector ID pairs (and TOF bin) for coincidence events.
        scanner_LUT (torch.tensor): scanner lookup table that provides spatial coordinates for all detector ID pairs
        object_meta (ObjectMeta): object metadata providing the region of reconstruction

    Returns:
        torch.tensor: all detector ID pairs corresponding to coincidence events
    """
    bmin = -torch.tensor(object_meta.shape) * torch.tensor(object_meta.dr) / 2
    bmax = torch.tensor(object_meta.shape) * torch.tensor(object_meta.dr) / 2
    bmin = bmin.to(detector_ids.device); bmax=bmax.to(detector_ids.device)
    origin = scanner_LUT[detector_ids[:,0]]
    direction = scanner_LUT[detector_ids[:,1]] - origin
    t1 = torch.where(
        direction>=0,
        (bmin - origin) / direction,
        (bmax - origin) / direction
    )
    t2 = torch.where(
        direction>=0,
        (bmax - origin) / direction,
        (bmin - origin) / direction
    )
    intersect = (t1[:,0]>t2[:,1])+(t1[:,1]>t2[:,0])+((t1[:,0]>t2[:,2]))+(t1[:,2]>t2[:,0])
    return detector_ids[~intersect]

def get_attenuation_map_nifti(path, object_meta):
    # If img is none, extract data from path
    data = nib.load(path)
    img = data.get_fdata()
    Sx, Sy, Sz = -(np.array(img.shape)-1) / 2
    dx, dy, dz = data.header['pixdim'][1:4]
    # Convert from RAS to LPS space for DICOM
    dx*=-1; dy*=-1
    M_highres = np.zeros((4,4))
    M_highres[0] = np.array([dx, 0, 0, Sx*dx])
    M_highres[1] = np.array([0, dy, 0, Sy*dy])
    M_highres[2] = np.array([0, 0, dz, Sz*dz])
    M_highres[3] = np.array([0, 0, 0, 1])
    dx, dy, dz = object_meta.dr
    Sx, Sy, Sz = -(np.array(object_meta.shape)-1) / 2
    M_pet = np.zeros((4,4))
    M_pet[0] = np.array([dx, 0, 0, Sx*dx])
    M_pet[1] = np.array([0, dy, 0, Sy*dy])
    M_pet[2] = np.array([0, 0, dz, Sz*dz])
    M_pet[3] = np.array([0, 0, 0, 1])
    M = npl.inv(M_highres) @ M_pet
    return torch.tensor(affine_transform(img, M, output_shape=object_meta.shape, mode='constant', order=1)) / 10
