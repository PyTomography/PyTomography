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

def get_scanner_LUT(
    path: str,
    init_volume_name: str = 'crystal',
    final_volume_name: str = 'world',
    mean_interaction_depth: float = 0,
    return_info: bool = False
    ) -> np.array:
    """Returns the scanner lookup table. The three values at a particular index in the lookup table correspond to the x, y, and z positions of the detector id correpsonding to that index.

    Args:
        path (str): Path to .mac file where the scanner geometry is defined in FATE
        init_volume_name (str, optional): Volume name corresponding the lowest level element in the GATE geometry. Defaults to 'crystal'.
        final_volume_name (str, optional): Volume name corresponding the highest level element in the GATE geometry. Defaults to 'world'.
        mean_interaction_depth (float, optional): Average interaction depth of photons in the crystals in mm. Defaults to 0.
        return_info (bool, optional): Returns information about the scanner geometry. Defaults to False.

    Returns:
        np.array: Scanner lookup table.
    """
    with open(path) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    # Recursively get names of all volumes
    volume = init_volume_name
    parents = [volume]
    while volume!=final_volume_name:
        volume = get_header_value(headerdata, f'/daughters/name.*{volume}', split_substr='/', split_idx=2, dtype=str)
        if not(volume):
            return None
        parents.append(volume)
    # Get initial positions of crystal
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
    x_crystal = x_crystal - get_header_value(headerdata, f'/gate/{init_volume_name}/geometry/setXLength', split_substr=None, split_idx=1) / 2 + mean_interaction_depth
    # Generate positions of all crystals using repeaters
    #parents.reverse()
    repeats = []
    for parent in parents:
        repeaters = get_header_value(headerdata, f'/gate/{parent}/repeaters/insert', split_substr=None, split_idx=1, dtype=str, return_all=True)
        if not(repeaters):
            continue
        for repeater in repeaters:
            if repeater=='cubicArray':
                repeat_numbers = [get_header_value(headerdata, f'/gate/{parent}/{repeater}/setRepeatNumber{coord}', split_substr=None, split_idx=1) for coord in ['X', 'Y', 'Z']]
                repeat_vector = [get_header_value(headerdata, f'/gate/{parent}/{repeater}/setRepeatVector', split_substr=None, split_idx=i) for i in range(1,4)]
                # GATE convention is that z changes first, so meshgrid is done like this
                xv, zv, yv = np.meshgrid(
                    repeat_vector[0] * (np.arange(0,repeat_numbers[0]) - (repeat_numbers[0]-1)/2),
                    repeat_vector[2] * (np.arange(0,repeat_numbers[2]) - (repeat_numbers[2]-1)/2),
                    repeat_vector[1] * (np.arange(0,repeat_numbers[1]) - (repeat_numbers[1]-1)/2),
                        )
                len_repeat = xv.ravel().shape[0]
                x_crystal_repeated = np.repeat(x_crystal[:,np.newaxis], len_repeat, axis=1)
                y_crystal_repeated = np.repeat(y_crystal[:,np.newaxis], len_repeat, axis=1)
                z_crystal_repeated = np.repeat(z_crystal[:,np.newaxis], len_repeat, axis=1)
                x_crystal = (x_crystal_repeated+xv.ravel()).ravel()
                y_crystal = (y_crystal_repeated+yv.ravel()).ravel()
                z_crystal = (z_crystal_repeated+zv.ravel()).ravel()
                repeats.append(int(np.prod(repeat_numbers)))
            elif repeater=='linear':
                repeat_number = get_header_value(headerdata, f'/gate/{parent}/{repeater}/setRepeatNumber', split_substr=None, split_idx=1)
                repeat_vector = [get_header_value(headerdata, f'/gate/{parent}/{repeater}/setRepeatVector', split_substr=None, split_idx=i) for i in range(1,4)]
                xr = repeat_vector[0] * (np.arange(0,repeat_number) - (repeat_number-1)/2)
                yr = repeat_vector[1] * (np.arange(0,repeat_number) - (repeat_number-1)/2)
                zr = repeat_vector[2] * (np.arange(0,repeat_number) - (repeat_number-1)/2)
                len_repeat = xr.shape[0]
                x_crystal_repeated = np.repeat(x_crystal[:,np.newaxis], len_repeat, axis=1)
                y_crystal_repeated = np.repeat(y_crystal[:,np.newaxis], len_repeat, axis=1)
                z_crystal_repeated = np.repeat(z_crystal[:,np.newaxis], len_repeat, axis=1)
                x_crystal = (x_crystal_repeated+xr).ravel()
                y_crystal = (y_crystal_repeated+yr).ravel()
                z_crystal = (z_crystal_repeated+zr).ravel() 
                repeats.append(int(repeat_number))
            elif repeater=='ring':
                repeat_number = get_header_value(headerdata, f'/gate/{parent}/{repeater}/setRepeatNumber', split_substr=None, split_idx=1)
                first_angle = get_header_value(headerdata, f'/gate/{parent}/{repeater}/setFirstAngle', split_substr=None, split_idx=1) * np.pi / 180
                if not first_angle:
                    first_angle = 0
                phi = np.linspace(first_angle, first_angle+2*np.pi, int(repeat_number), endpoint=False)
                x_crystal_repeated = np.repeat(x_crystal[:,np.newaxis], len(phi), axis=1)
                y_crystal_repeated = np.repeat(y_crystal[:,np.newaxis], len(phi), axis=1)
                z_crystal_repeated = np.repeat(z_crystal[:,np.newaxis], len(phi), axis=1)
                x_crystal = (np.cos(phi)*x_crystal_repeated - np.sin(phi)*y_crystal_repeated).ravel()
                y_crystal = (np.sin(phi)*x_crystal_repeated + np.cos(phi)*y_crystal_repeated).ravel()
                z_crystal = (z_crystal_repeated).ravel()
                repeats.append(int(repeat_number))
    info = dict(zip(parents, repeats))
    if return_info:
        return torch.tensor(-np.vstack((x_crystal,y_crystal,z_crystal)).T), info
    else:
        return torch.tensor(-np.vstack((x_crystal,y_crystal,z_crystal)).T)
    
def get_N_components(mac_file: str) -> tuple:
    """Obtains the number of gantrys, rsectors, modules, submodules, and crystals per level from a GATE macro file.

    Args:
        mac_file (str): Path to the gate macro file

    Returns:
        tuple: number of gantrys, rsectors, modules, submodules, and crystals
    """
    geom_info = get_scanner_LUT(mac_file, return_info=True)[1]
    N_gantry = 1
    N_module = geom_info['module']
    try:
        N_submodule = geom_info['submodule']
    except:
        N_submodule = 1
    N_rsector = geom_info['rsector']
    N_crystal = geom_info['crystal']
    return N_gantry, N_rsector, N_module, N_submodule, N_crystal

def get_detector_ids(
    paths: Sequence[str],
    mac_file: str,
    TOF: bool = False,
    TOF_bin_edges: np.array = None,
    substr: str = 'Coincidences',
    same_source_pos: bool = False
    ) -> np.array:
    """Obtains the detector IDs from a sequence of ROOT files

    Args:
        paths (Sequence[str]): sequence of root file paths
        mac_file (str): GATE geometry macro file
        TOF (bool, optional): Whether or not to get TOF binning information. Defaults to False.
        TOF_bin_edges (np.array, optional): TOF bin edges; required if TOF is True. Defaults to None.
        substr (str, optional): Substring to index for in ROOT files. Defaults to 'Coincidences'.
        same_source_pos (bool, optional): Only include coincidences that correspond to the same source position. This can be used to filter randoms. Defaults to False.

    Returns:
        np.array: Array of all detector ID pairs corresponding to all detected LORs.
    """
    if TOF:
        if TOF_bin_edges is None:
            Exception('If using TOF, must provide TOF bin edges for binning')
    N_gantry, N_rsector, N_module, N_submodule, N_crystal = get_N_components(mac_file)
    detector_ids = [[],[],[]]
    for i,path in enumerate(paths):
        with uproot.open(path) as f:
            if same_source_pos:
                xs1 = f[substr]['sourcePosX1'].array(library='np')
                xs2 = f[substr]['sourcePosX2'].array(library='np')
                ys1 = f[substr]['sourcePosY1'].array(library='np')
                ys2 = f[substr]['sourcePosY2'].array(library='np')
                zs1 = f[substr]['sourcePosZ1'].array(library='np')
                zs2 = f[substr]['sourcePosZ2'].array(library='np')
                same_location_idxs = (xs1==xs2)*(ys1==ys2)*(zs1==zs2)
            for j in range(2):
                gantry_id = f[substr][f'gantryID{j+1}'].array(library="np")
                rsector_id = f[substr][f'rsectorID{j+1}'].array(library="np")
                module_id = f[substr][f'moduleID{j+1}'].array(library="np")
                submodule_id = f[substr][f'submoduleID{j+1}'].array(library="np")
                crystal_id = f[substr][f'crystalID{j+1}'].array(library="np")
                detector_id = crystal_id * N_submodule * N_module * N_rsector * N_gantry \
                    + submodule_id * N_module * N_rsector * N_gantry \
                    + module_id * N_rsector * N_gantry \
                    + rsector_id * N_gantry \
                    + gantry_id
                if same_source_pos:
                    detector_id = detector_id[same_location_idxs]
                detector_ids[j].append(detector_id.astype(np.int16))
            if TOF:
                t1 = f[substr]['time1'].array(library='np')
                t2 = f[substr]['time2'].array(library='np')
                tof_pos = 1e12*(t2 - t1) * 0.15 # ps to mm
                detector_id = np.digitize(-tof_pos, TOF_bin_edges) - 1
                if same_source_pos:
                    detector_id = detector_id[same_location_idxs]
                detector_ids[2].append(detector_id) 
    if TOF:
        return np.array([
            np.concatenate(detector_ids[0]),
            np.concatenate(detector_ids[1]),
            np.concatenate(detector_ids[2])]).T
    else:
        return np.array([
            np.concatenate(detector_ids[0]),
            np.concatenate(detector_ids[1])]).T

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
        torch.abs(x1*y2-y1*x2)/torch.sqrt((x1-x2)**2+(y1-y2)**2)
    )

def get_table(det_ids: torch.tensor, mac_file: str) -> torch.tensor:
    r"""Obtains a table of crystal1ID, crystal2ID, submoduleID, :math:`\Delta`moduleID, :math:`\Delta`rsectorID corresponding to each of the detector id pairs provided. Useful fo symmetries when computing normalization :math:`\eta`.

    Args:
        det_ids (torch.tensor): :math:`N \times 2` (non-TOF) or :math:`N \times 3` (TOF) tensor that provides detector ID pairs (and TOF bin) for coincidence events.
        mac_file (str): GATE macro file that defines detector geometry

    Returns:
        torch.tensor: A 2D tensor that lists crystal1ID, crystal2ID, submoduleID, :math:`\Delta`moduleID, :math:`\Delta`rsectorID for each LOR.
    """
    N_gantry, N_rsector, N_module, N_submodule, N_crystal = get_N_components(mac_file)
    # Larger ID comes second
    det_ids = torch.vstack([det_ids.min(axis=1)[0], det_ids.max(axis=1)[0]]).T
    cry_ids = det_ids // (N_submodule * N_module * N_rsector * N_gantry)
    subM_ids = det_ids % (N_submodule * N_module * N_rsector * N_gantry) // (N_module * N_rsector * N_gantry)
    M_ids = det_ids % (N_module * N_rsector * N_gantry) // (N_rsector * N_gantry)
    R_ids = det_ids % (N_rsector * N_gantry) // N_gantry
    deltaM_ids = torch.abs(torch.diff(M_ids, axis=1))
    deltaR_ids = torch.abs(torch.diff(R_ids, axis=1))
    return torch.tensor(torch.concatenate([cry_ids, subM_ids, deltaM_ids, deltaR_ids], axis=1))

def get_eta_cylinder_calibration(
    paths: Sequence[str],
    mac_file: str,
    cylinder_radius: float,
    same_source_pos: bool = False,
    mean_interaction_depth: float = 0
    ) -> torch.tensor:
    """Obtain normalization :math:`\eta` from a calibration scan consisting of a cylindrical shell

    Args:
        paths (Sequence[str]): paths of all ROOT files containing data
        mac_file (str): GATE macro file that defines scanner geometry
        cylinder_radius (float): The radius of the cylindrical shell used in calibration
        same_source_pos (bool, optional): Only include coincidence events with same source position; can be used to filter out randoms. Defaults to False.
        mean_interaction_depth (float, optional): Mean interaction depth of photons in detector crystals. Defaults to 0.

    Returns:
        torch.tensor: Tensor corresponding to :math:`eta`.
    """
    N_gantry, N_rsector, N_module, N_submodule, N_crystal = get_N_components(mac_file)
    N_detectors = N_gantry* N_rsector * N_module * N_submodule * N_crystal
    # Geometry correction factor for non-unform exposure from cylindrical shell
    scanner_LUT = torch.tensor(get_scanner_LUT(mac_file, mean_interaction_depth=mean_interaction_depth))
    all_LOR_ids = torch.combinations(torch.arange(N_detectors).to(torch.int32), 2)
    geometric_correction_factor = 1/(torch.sqrt(1-(get_radius(all_LOR_ids, scanner_LUT) / cylinder_radius )**2) + pytomography.delta)
    # Detector correction factor (exploits symmetries)
    H = torch.zeros(N_crystal, N_crystal, N_submodule, N_submodule, N_module, N_rsector)
    for path in paths:
        det_ids = torch.tensor(get_detector_ids([path], mac_file, same_source_pos=same_source_pos))
        vals = get_table(det_ids, mac_file)
        bins = [torch.arange(x).to(torch.float32)-0.5 for x in [N_crystal+1, N_crystal+1, N_submodule+1, N_submodule+1, N_module+1, N_rsector+1]]
        H += torch.histogramdd(vals.to(torch.float32), bins)[0]
    vals_all_pairs = get_table(torch.combinations(torch.arange(N_detectors).to(torch.int32), 2), mac_file)
    N_bins = torch.histogramdd(vals_all_pairs.to(torch.float32), bins)[0]
    # If you want to test this later, also return H and N_bins seperately
    return (H/N_bins)[vals_all_pairs[:,0], vals_all_pairs[:,1], vals_all_pairs[:,2], vals_all_pairs[:,3], vals_all_pairs[:,4], vals_all_pairs[:,5]] * geometric_correction_factor

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