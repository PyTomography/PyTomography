from __future__ import annotations
from collections.abc import Sequence
import pytomography
from pytomography.metadata import ObjectMeta
import torch
import numpy as np
import numpy.linalg as npl
import uproot
from scipy.ndimage import affine_transform
from ..shared import get_header_value, get_attenuation_map_interfile
from .shared import listmode_to_sinogram,  get_detector_ids_from_trans_axial_ids, get_axial_trans_ids_from_info, get_scanner_LUT

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
    amap = get_attenuation_map_interfile(headerfile)[0].cpu().numpy()
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
    amap = torch.tensor(amap, device=pytomography.device).unsqueeze(0) / 10 # to mm^-1
    return amap

def get_detector_info(
    path: str,
    init_volume_name: str = 'crystal',
    final_volume_name: str = 'world',
    mean_interaction_depth = 0,
    min_rsector_difference = 0
    ) -> np.array:
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
    return info



def get_axial_trans_ids_from_ROOT(f, j, info, substr: str = 'Coincidences', sort_by_detector_ids=False):
    ids_rsector = torch.tensor(f[substr][f'rsectorID{j+1}'].array(library="np"))
    ids_module = torch.tensor(f[substr][f'moduleID{j+1}'].array(library="np"))
    ids_submodule = torch.tensor(f[substr][f'submoduleID{j+1}'].array(library="np"))
    ids_crystal = torch.tensor(f[substr][f'crystalID{j+1}'].array(library="np"))
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
    paths,
    mac_file: str,
    TOF: bool = False,
    TOF_bin_edges: np.array = None,
    substr: str = 'Coincidences',
    include_randoms: bool = True,
    include_scatters: bool = True,
    randoms_only: bool = False,
    scatters_only: bool = False
    ) -> np.array:
    if TOF:
        if TOF_bin_edges is None:
            Exception('If using TOF, must provide TOF bin edges for binning')
    info = get_detector_info(mac_file)
    detector_ids_trio = [[],[],[]]
    for i,path in enumerate(paths):
        with uproot.open(path) as f:
            N_events = f['Coincidences']['sourcePosX1'].array(library='np').shape[0]
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
                ids_trans_crystal, ids_axial_crystal, ids_trans_submodule, ids_axial_submodule, ids_trans_module, ids_axial_module, ids_trans_rsector, ids_axial_rsector = get_axial_trans_ids_from_ROOT(f, j, info, substr)
                detector_ids = get_detector_ids_from_trans_axial_ids(ids_trans_crystal, ids_trans_submodule, ids_trans_module, ids_trans_rsector, ids_axial_crystal, ids_axial_submodule, ids_axial_module, ids_axial_rsector, info)
                detector_ids = detector_ids[valid_indices]
                detector_ids_trio[j].append(detector_ids.to(torch.int16))
            if TOF:
                t1 = f[substr]['time1'].array(library='np')
                t2 = f[substr]['time2'].array(library='np')
                tof_pos = 1e12*(t2 - t1) * 0.15 # ps to mm
                detector_id = np.digitize(-tof_pos, TOF_bin_edges) - 1
                # First see if only binning scatters/randoms
                detector_id = detector_id[valid_indices]
                detector_ids_trio[2].append(torch.tensor(detector_id))
    
    if TOF:
        return torch.vstack([
            torch.concatenate(detector_ids_trio[0]),
            torch.concatenate(detector_ids_trio[1]),
            torch.concatenate(detector_ids_trio[2])]).T
    else:
        return torch.vstack([
            torch.concatenate(detector_ids_trio[0]),
            torch.concatenate(detector_ids_trio[1])]).T
        
def get_symmetry_histogram_from_ROOTfile(f, info, substr='Coincidences', include_randoms=True) -> torch.tensor:
    ids1_trans_crystal, ids1_axial_crystal, ids1_trans_submodule, ids1_axial_submodule, ids1_trans_module, ids1_axial_module, ids1_trans_rsector, ids1_axial_rsector = get_axial_trans_ids_from_ROOT(f, 0, info, substr= substr)
    ids2_trans_crystal, ids2_axial_crystal, ids2_trans_submodule, ids2_axial_submodule, ids2_trans_module, ids2_axial_module, ids2_trans_rsector, ids2_axial_rsector = get_axial_trans_ids_from_ROOT(f, 1, info, substr= substr)
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

def get_symmetry_histogram_all_combos(info) -> torch.tensor:
    ids_trans_crystal, ids_axial_crystal, ids_trans_submodule, ids_axial_submodule, ids_trans_module, ids_axial_module, ids_trans_rsector, ids_axial_rsector = get_axial_trans_ids_from_info(info, return_combinations=True, sort_by_detector_ids=True)
    ids_delta_axial_submodule = (ids_axial_submodule[:,1] - ids_axial_submodule[:,0]) + (info['submoduleAxialNr'] - 1)
    ids_delta_axial_module = (ids_axial_module[:,1] - ids_axial_module[:,0]) + (info['moduleAxialNr'] - 1)
    ids_delta_trans_rsector = (ids_trans_rsector[:,1] - ids_trans_rsector[:,0]) % info['rsectorTransNr'] # because of circle
    return torch.vstack([ids_axial_crystal[:,0], ids_axial_crystal[:,1], ids_trans_crystal[:,0], ids_trans_crystal[:,1], ids_delta_axial_submodule, ids_delta_axial_module, ids_delta_trans_rsector]).T

def get_eta_cylinder_calibration(
    paths,
    mac_file: str,
    cylinder_radius: float,
    include_randoms: bool = True,
    mean_interaction_depth: float = 0
    ) -> torch.tensor:
    info = get_detector_info(mac_file, mean_interaction_depth=mean_interaction_depth)
    # Part 1: Geometry correction factor for non-unform exposure from cylindrical shell
    scanner_LUT = torch.tensor(get_scanner_LUT(info))
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
    weights_sensitivity,
    macro_path
):
    info = get_detector_info(macro_path)
    scanner_LUT = torch.tensor(get_scanner_LUT(info))
    all_LOR_ids = torch.combinations(torch.arange(scanner_LUT.shape[0]).to(torch.int32), 2)
    return listmode_to_sinogram(all_LOR_ids, info, weights=weights_sensitivity)

def get_norm_sinogram_from_root_data(
    normalization_paths,
    macro_path,
    cylinder_radius,
    include_randoms=True,
    mean_interaction_depth=0
):
    eta = get_eta_cylinder_calibration(
        normalization_paths,
        macro_path,
        cylinder_radius,
        include_randoms=include_randoms,
        mean_interaction_depth=mean_interaction_depth
    )
    return get_norm_sinogram_from_listmode_data(eta, macro_path)

def get_sinogram_from_listmode_data(
    detector_ids,
    macro_path
):
    # Gate specific function
    info = get_detector_info(macro_path)
    return listmode_to_sinogram(detector_ids, info)

def get_sinogram_from_root_data(
    paths,
    macro_path,
    include_randoms=True
):
    detector_ids = get_detector_ids_from_root(
        paths,
        macro_path,
        include_randoms=include_randoms,
        TOF=False)
    return get_sinogram_from_listmode_data(detector_ids, macro_path)

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
