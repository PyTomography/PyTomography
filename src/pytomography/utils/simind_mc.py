from __future__ import annotations
from typing import Sequence
import os
import time
import subprocess
import tempfile
from copy import copy
import numpy as np
from pytomography.metadata import ObjectMeta
from pytomography.metadata.SPECT import SPECTProjMeta
import pydicom
import torch
from pytomography.transforms import Transform

ENERGY_RESOLUTION_MODELS = ['siemens']

#: The simind.ini changes every simulation is run with (SIMIND's ``/in:`` switch; each change needs a switch of its own,
#: since SIMIND applies only the first change of ``in:x2,6x21,0x``):
#:
#: * ``in:x2,6x``: entry 2, the maximum number of scatter orders in the phantom, is 6.
#: * ``in:x21,0x``: entry 21, the density above which a voxel of the phantom is made of the second phantom material
#:   (1170, 1000 * g/cm3, by default, meant for bone), is 0, so that every voxel with any density is made of the second
#:   material. With the default, SIMIND V8.0 (the Windows build of 2025-01-28) leaves the photons emitted in soft tissue
#:   or water (the first material) unattenuated on their way to the detector: a point source in the centre of a 10 cm
#:   radius water cylinder gives as many unscattered counts as in air, with the cylinder's scatter on top. The forward
#:   projections of :class:`~pytomography.projectors.SPECT.MonteCarloHybridSPECTSystemMatrix` were then about three
#:   times too large and lacked attenuation, so its reconstructions were too low and brighter at the edge of the body.
#:   With every voxel in the second material, the attenuation agrees with PyTomography's attenuated projector to a few
#:   percent. Both phantom materials are water in the ``simind.smc`` file used here, so nothing else changes.
SIMIND_INI_CHANGES = ('in:x2,6x', 'in:x21,0x')

def save_attenuation_map(
    attenuation_map: torch.Tensor,
    dx: float,
    temp_path: str
):
    """Save attenuation map as binary file to temporary directory for subsequent use by Monte Carlo scatter simulation.
    SIMIND reads the attenuation per voxel (mu times the voxel size), which does not depend on the length unit.

    Args:
        attenuation_map (torch.Tensor): Attenuation map to save (1/mm)
        dx (float): Spacing of attenuation map (mm)
        temp_path (str): Temporary folder to save to
    """
    d = (attenuation_map).cpu().numpy().astype(np.float32) * dx
    d_flat = d.swapaxes(0,2).ravel()
    d_flat.tofile(os.path.join(temp_path, f'phantom140_atn_av.bin'))
    
def save_source_map(
    source_map: torch.Tensor,
    temp_path: str,
    scaling: float,
    vmax: float = 1e6
):
    """Save source map as binary file to temporary directory for subsequent use by Monte Carlo scatter simulation.

    Args:
        source_map (torch.Tensor): Source map to save
        temp_path (str): Temporary folder to save to
        vmax (float, optional): Maximum value of source map, prevents divergence at early iterations. Defaults to 1e6.
    """
    source_map = source_map.clamp(0, vmax)
    d = source_map.cpu().numpy().astype(np.float32)
    d *= scaling / d.sum()
    d_flat = d.swapaxes(0,2).ravel()
    d_flat.tofile(os.path.join(temp_path, f'source_act_av.bin'))

def get_simind_params_from_metadata(
    object_meta: ObjectMeta,
    proj_meta: SPECTProjMeta
) -> dict:
    """Obtain dictionary of SIMIND parameters from object and projection metadata. PyTomography's lengths are in mm;
    SIMIND's indices 02-10, 28 and 31 are in cm.

    Args:
        object_meta (ObjectMeta): SPECT object metadata used in reconstruction
        proj_meta (SPECTProjMeta): SPECT projection metadata used in reconstruction

    Returns:
        dict: Dictionary of SIMIND parameters obtainable from object and projection metadata
    """
    num_angles = len(proj_meta.angles)
    # save radii to corr file
    if proj_meta.angles[1] - proj_meta.angles[0] > 0:
        index_30 = 0 # clockwise
    else:
        index_30 = 2 # counter-clockwise
    cm = 0.1                                    # mm to cm
    index_dict = {
        '28': proj_meta.dr[0] * cm, # pixel spacing
        '29': num_angles, # number of projetions
        '30': index_30,
        '41': -proj_meta.angles[0].item(), # first angle
        '76': proj_meta.shape[-2], # number of pixels in projection (X)
        '77': proj_meta.shape[-1], # number of pixels in projection (Y)
        '78': object_meta.shape[0], # X voxels (source)
        '79': object_meta.shape[0], # X voxels (phantom)
        '81': object_meta.shape[1], # Y voxels (source)
        '82': object_meta.shape[1], # Y voxels (phantom)
        '31': object_meta.dr[0] * cm, # size of source/CT slices
        '34': object_meta.shape[2], # number of source/CT slices (in this case aligned with SPECT)
        '02': object_meta.dr[2]*object_meta.shape[2]/2 * cm, # size of source phantom
        '03': object_meta.dr[0]*object_meta.shape[0]/2 * cm, # size of source phantom
        '04': object_meta.dr[1]*object_meta.shape[1]/2 * cm, # size of source phantom
        '05': object_meta.dr[2]*object_meta.shape[2]/2 * cm, # size of CT phantom
        '06': object_meta.dr[0]*object_meta.shape[0]/2 * cm, # size of CT phantom
        '07': object_meta.dr[1]*object_meta.shape[1]/2 * cm, # size of CT phantom
        '08': object_meta.dr[0]*object_meta.shape[0]/2 * cm,
        '10': object_meta.dr[2]*object_meta.shape[2]/2 * cm,
    }
    return index_dict

#: The detector thicknesses' keywords before PyTomography 4, in cm; they are now in mm, with ``_mm`` in their names.
_THICKNESSES_IN_CM = ('crystal_thickness', 'cover_thickness', 'backscatter_thickness')


def refuse_thicknesses_in_cm(kwargs: dict) -> None:
    """Raise a TypeError that explains the rename when ``kwargs`` holds a detector thickness under its old name (cm)."""
    for name in _THICKNESSES_IN_CM:
        if name in kwargs:
            raise TypeError(f'{name} (cm) is now {name}_mm, in mm like every length in PyTomography 4: '
                            f'{name}={kwargs[name]} is {name}_mm={10 * kwargs[name]:g}')
    if kwargs:
        raise TypeError(f'unexpected keyword arguments {sorted(kwargs)}')


def get_simind_isotope_detector_params(
    isotope_name: str,
    collimator_type: str,
    cover_thickness_mm: float | None = None,
    backscatter_thickness_mm: float | None = None,
    crystal_thickness_mm: float | None = None,
    energy_resolution_140keV: float = 0,
    advanced_energy_resolution_model: str | None = None,
    advanced_collimator_modeling: bool = False,
    random_collimator_movement: bool = False,
    **removed,
) -> dict:
    """Obtain SIMIND parameter dictionary from isotope and detector parameters. The thicknesses are in mm, like every
    length in PyTomography; SIMIND's indices 09, 11 and 13 are in cm, so they are converted here.

    Args:
        isotope_name (str): Name of isotope used for Monte Carlo scatter simulation
        collimator_type (str): Collimator type used for Monte Carlo scatter simulation.
        cover_thickness_mm (float): Cover thickness (mm) used for simulation. Currently assumes aluminum is used.
        backscatter_thickness_mm (float): Equivalent backscatter thickness (mm) used for simulation. Currently assumes pyrex is used.
        crystal_thickness_mm (float): Crystal thickness (mm) used for simulation. Currently assumes NaI.
        energy_resolution_140keV (float): Energy resolution in percent of the detector at 140keV. Currently uses the relationship that resolution is proportional to sqrt(E) for E in keV.
        advanced_collimator_modeling (bool, optional): Whether or not to use advanced collimator modeling that can be used to model septal penetration and scatter. Defaults to False.
        random_collimator_movement (bool, optional): Whether or not to include random collimator movement (e.g. holes are not fixed in place). Defaults to False.

    Returns:
        dict: Dictionary of SIMIND parameters obtainable from isotope and detector parameters
    """
    refuse_thicknesses_in_cm(removed)
    missing = [n for n, v in (('cover_thickness_mm', cover_thickness_mm), ('backscatter_thickness_mm', backscatter_thickness_mm),
                              ('crystal_thickness_mm', crystal_thickness_mm)) if v is None]
    if missing:
        raise TypeError(f'missing {", ".join(missing)}')
    per_cm = 10.0                               # mm per cm: SIMIND's unit for these indices is cm
    if advanced_energy_resolution_model is not None:
        if advanced_energy_resolution_model=='siemens':
            energy_resolution_140keV = 0
        else:
            raise ValueError(f'Advanced energy resolution model {advanced_energy_resolution_model} not recognized.')
    index_dict = {
        'fi': isotope_name,
        'cc': collimator_type,
        '22': energy_resolution_140keV, # TODO: add energy resolution arbitrary function as argument
        '53': int(advanced_collimator_modeling),
        '59': int(random_collimator_movement),
        '13': cover_thickness_mm / per_cm, # TODO: add material as argument, default aluminum
        '11': backscatter_thickness_mm / per_cm, # TODO: add material as argument, default pyrex
        '09': crystal_thickness_mm / per_cm
    }
    if advanced_energy_resolution_model is not None:
        index_dict['Fe'] = advanced_energy_resolution_model
    return index_dict

def get_energy_window_params_dicom(
    file_NM: str,
    idxs: Sequence[int] | None = None
) -> Sequence[str]:
    """Obtain energy window parameters from a DICOM file: this includes a list of strings which, when written to a file, correspond to a typical "scattwin.win" file used by SIMIND.

    Args:
        file_NM (str): DICOM projection file name
        idxs (Sequence[int]): Indices corresponding to the energy windows to extract. More than one index is provided in cases where multi-photopeak reconstruction is used and scatter needs to be obtained at all windows.

    Returns:
        Sequence[str]: Lines of the "scattwin.win" file corresponding to the energy windows specified by the indices.
    """
    from pytomography.io.SPECT.dicom import read_metadata
    windows = read_metadata(file_NM)[1].energy_windows
    if idxs is None:
        idxs = range(len(windows))
    return [f'{windows[idx].lower_keV},{windows[idx].upper_keV},0' for idx in idxs]
        
def get_energy_window_params_simind(headerfiles: Sequence[str]|str)-> Sequence[str]:
    """Obtain energy window parameters from a list of SIMIND header files: this includes a list of strings which, when written to a file, correspond to a typical "scattwin.win" file used by SIMIND.

    Args:
        headerfiles (Sequence[str]): SIMIND header files

    Returns:
        Sequence[str]: Lines of the "scattwin.win" file corresponding to the energy windows specified by the header files.
    """
    from pytomography.io.SPECT.simind import _energy_window, _header
    if isinstance(headerfiles, (str, os.PathLike)):
        headerfiles = [headerfiles]
    lines = []
    for headerfile in headerfiles:
        window = _energy_window(_header(headerfile))
        lines.append(f'{window.lower_keV},{window.upper_keV},0')
    return lines
        
def create_simind_command(index_dict: dict, parallel_idx: int) -> str:
    """Creates the terminal command to run SIMIND with the specified parameters: voxel source and attenuation maps
    (XCAT format), scatter and air images, and the simind.ini changes in :data:`SIMIND_INI_CHANGES`.

    Args:
        index_dict (dict): Dictionary of SIMIND parameters
        parallel_idx (int): Random seed used for simulation, used to differentiate between parallel simulations

    Returns:
        str: Terminal command to run SIMIND with the specified parameters
    """
    simind_command = f'temp_output{parallel_idx}/14:-7/15:-7/CA:1/RR:{parallel_idx}/01:-208'
    for change in SIMIND_INI_CHANGES:
        simind_command += f'/{change}'
    for key, value in index_dict.items():
        if isinstance(value, list):
            for v in value:
                simind_command += f'/{key}:{v}'
        else:
            simind_command += f'/{key}:{value}'
    return simind_command

def add_together(n_parallel: int, n_windows: int, temp_path: str):
    """Adds together all the parallel SIMIND simulations to obtain the final scatter and total projections

    Args:
        n_parallel (int): Number of parallel simulations
        n_windows (int): Number of energy windows used in the simulation
        temp_path (str): Temporary directory where files were saved
    """
    xscats = [0] * n_windows
    xtots = [0] * n_windows
    for i in range(n_parallel):
        for j in range(n_windows):
            w_scat = np.fromfile(os.path.join(temp_path, f'temp_output{i}_sca_w{j+1}.a00'), dtype=np.float32)
            xscats[j] += w_scat
            w_tot = np.fromfile(os.path.join(temp_path, f'temp_output{i}_tot_w{j+1}.a00'), dtype=np.float32)
            xtots[j] += w_tot 
    for i in range(n_windows):
        # Take mean
        xscat = xscats[i] / n_parallel
        xtot = xtots[i] / n_parallel
        xscat.tofile(os.path.join(temp_path, f'sca_w{i+1}.a00'))
        xtot.tofile(os.path.join(temp_path, f'tot_w{i+1}.a00'))
        # Create a header file for it
        subprocess.run(['mv', f'temp_output0_sca_w{i+1}.h00', f'sca_w{i+1}.h00'], cwd=temp_path)
        subprocess.run(['sed', '-i', f's/temp_output0_sca_w{i+1}.a00/sca_w{i+1}.a00/g', f'sca_w{i+1}.h00'], cwd=temp_path)
        # REMOVE THESE LATER >
        subprocess.run(['mv', f'temp_output0_tot_w{i+1}.h00', f'tot_w{i+1}.h00'], cwd=temp_path)
        subprocess.run(['sed', '-i', f's/temp_output0_tot_w{i+1}.a00/tot_w{i+1}.a00/g', f'tot_w{i+1}.h00'], cwd=temp_path)
                                   
def run_scatter_simulation(
    source_map: torch.Tensor,
    attenuation_map_140keV: torch.Tensor,
    object_meta: ObjectMeta,
    proj_meta: SPECTProjMeta,
    energy_window_params: list,
    simind_index_dict: dict,
    n_events: int,
    n_parallel: int = 1,
    return_total: bool = False
) -> torch.Tensor:
    """Runs a Monte Carlo scatter simulation using SIMIND

    Args:
        source_map (torch.Tensor): Source map used in the simulation
        attenuation_map_140keV (torch.Tensor): Attenuation map at 140keV used in the simulation
        object_meta (ObjectMeta): SPECT ObjectMeta used in reconstruction
        proj_meta (SPECTProjMeta): SPECT projection metadata used in reconstruction
        energy_window_params (list): List of strings which constitute a typical "scattwin.win" file used by SIMIND
        primary_window_idxs (Sequence[int]): Indices from the energy_window_params list corresponding to indices used as photopeak in reconstruction. For single photopeak reconstruction, this will be a list of length 1, while for multi-photopeak reconstruction, this will be a list of length > 1.
        simind_index_dict (dict): Dictionary of SIMIND parameters
        n_events (int): Number of events to simulate per projection angle
        n_parallel (int, optional): Number of simulations to perform in parallel, should not exceed number of CPU cores. Defaults to 1.
        return_total (bool, optional): Whether or not to also return the total projections. Defaults to False.

    Returns:
        torch.Tensor: Simulated projections
    """
    
    n_batches = 1
    
    temp_dir = tempfile.TemporaryDirectory()
    # Create window file
    with open(os.path.join(temp_dir.name, 'simind.win'), 'w') as f:
        f.write('\n'.join(energy_window_params))
    # Radial positions
    np.savetxt(os.path.join(temp_dir.name, f'radii_corfile.cor'), np.asarray(proj_meta.radii) / 10)   # SIMIND: cm
    # Save attenuation map and source map to TEMP directory
    save_attenuation_map(attenuation_map_140keV, object_meta.dr[0], temp_dir.name)
    save_source_map(source_map, temp_dir.name, scaling=n_events/n_parallel/n_batches)
    # Move simind.smc and energy_resolution.erf to TEMP directory
    module_path = os.path.dirname(os.path.abspath(__file__))
    smc_filepath = os.path.join(module_path, "../data/simind.smc")
    p = subprocess.Popen(['cp', smc_filepath, f'{temp_dir.name}/simind.smc']) 
    p.wait() # wait for copy to complete
    for energy_res_model in ENERGY_RESOLUTION_MODELS:
        e_res_filepath = os.path.join(module_path, f"../data/{energy_res_model}.erf")
        p = subprocess.Popen(['cp', e_res_filepath, f'{temp_dir.name}/{energy_res_model}.erf'])
    # Create simind commands and run simind in parallel
    simind_commands = [create_simind_command(simind_index_dict, i) for i in range(n_parallel*n_batches)]
    for batchIdx in range(n_batches):
        n0, n1 = batchIdx * n_parallel, (batchIdx + 1) * n_parallel
        procs = [subprocess.Popen([f'simind', 'simind', simind_command, 'radii_corfile.cor'], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=temp_dir.name) for simind_command in simind_commands[n0:n1]]
        for p in procs:
            p.wait()
            if p.returncode != 0:  # Check if the process exited with an error
                error_output = p.stderr.read().decode('utf-8')
                print(f"Error in process {p.args}:\n{error_output}")
    # Add together projection data from all seperate processes
    add_together(n_parallel*n_batches, len(energy_window_params), temp_dir.name)
    from pytomography.io.SPECT import read_simind
    radii_file = os.path.join(temp_dir.name, 'radii_corfile.cor')
    proj_simind_scatter = read_simind([f'{temp_dir.name}/sca_w{i+1}.h00' for i in range(len(energy_window_params))], radii_file=radii_file)[0]
    proj_simind_tot = read_simind([f'{temp_dir.name}/tot_w{i+1}.h00' for i in range(len(energy_window_params))], radii_file=radii_file)[0]
    # Remove data files from temporary directory
    temp_dir.cleanup()
    # Return data
    if return_total:
        return proj_simind_tot
    else:
        return proj_simind_scatter
    
    def __init__(
        self,
        object_meta: ObjectMeta,
        proj_meta: SPECTProjMeta,
        n_events: int,
        n_parallel: int,
        obj2obj_transforms: Sequence[Transform],
        proj2proj_transforms: Sequence[Transform],
        attenuation_map_140keV: torch.Tensor,
        energy_window_params: Sequence[str],
        primary_window_idx: int,
        isotope_names: Sequence[str],
        isotope_ratios: Sequence[float],
        collimator_type: str,
        crystal_thickness: float,
        cover_thickness: float,
        backscatter_thickness: float,
        energy_resolution_140keV: float = 0,
        advanced_energy_resolution_model: str | None = None,
        advanced_collimator_modeling: bool = False,
    ):
        """Monte Carlo Hybrid SPECT System Matrix class that uses SIMIND to simulate scatter and total projections.
        Args:
            object_meta (ObjectMeta): SPECT ObjectMeta used in reconstruction
            proj_meta (SPECTProjMeta): SPECT projection metadata used in reconstruction
            n_events (int): Number of photons to simulate per projection angle
            n_parallel (int): Number of simulations to perform in parallel, should not exceed number of CPU cores.
            obj2obj_transforms (Sequence[Transform]): List of object to object transforms for back projection
            proj2proj_transforms (Sequence[Transform]): List of projection to projection transforms for back projection
            attenuation_map_140keV (torch.Tensor): Attenuation map at 140keV (used in MC simulation)
            energy_window_params (Sequence[str]): List of strings which constitute a typical "scattwin.win" file used by SIMIND
            primary_window_idx (int): Index from the energy_window_params list corresponding to indices used as photopeak in reconstruction. For single photopeak reconstruction, this will be a list of length 1, while for multi-photopeak reconstruction, this will be a list of length > 1.
            isotope_names (Sequence[str]): List of isotope names used in the simulation
            isotope_ratios (Sequence[float]): Proportion of all isotopes.
            collimator_type (str): Collimator type used for Monte Carlo scatter simulation (should use SIMIND name).
            crystal_thickness (float): Crystal thickness used for Monte Carlo scatter simulation (currently assumes NaI)
            cover_thickness (float): Cover thickness used for simulation. Currently assumes aluminum is used.
            backscatter_thickness (float): Equivalent backscatter thickness used for simulation. Currently assumes pyrex is used.
            energy_resolution_140keV (float): Energy resolution in percent of the detector at 140keV. Currently uses the relationship that resolution is proportional to sqrt(E) for E in keV.
            advanced_energy_resolution_model (str | None, optional): Advanced energy resolution model to use. If provided, then ``energy_resolution_140keV`` is not used. Currently only 'siemens' is supported. Defaults to None.
            advanced_collimator_modeling (bool, optional): Whether or not to use advanced collimator modeling that can be used to model septal penetration and scatter. Defaults to False.
        """
        super().__init__(
            obj2obj_transforms,
            proj2proj_transforms,
            object_meta,
            proj_meta,
            object_initial_based_on_camera_path=True,
        )
        self.attenuation_map_140keV = attenuation_map_140keV
        self.energy_window_params = energy_window_params
        self.isotope_names = isotope_names
        self.isotope_ratios = isotope_ratios
        self.collimator_type = collimator_type
        self.cover_thickness = cover_thickness
        self.crystal_thickness = crystal_thickness  
        self.backscatter_thickness = backscatter_thickness
        self.energy_resolution_140keV = energy_resolution_140keV
        self.advanced_energy_resolution_model = advanced_energy_resolution_model
        self.advanced_collimator_modeling = advanced_collimator_modeling
        self.primary_window_idx = primary_window_idx
        self.n_events = n_events
        self.n_parallel = n_parallel
        
    def _get_proj_meta_subset(self, subset_idx: int) -> SPECTProjMeta:
        """Creates a new SPECTProjMeta that corresponds to a subset of projections
        Args:
            subset_idx (int): Index of the subset to use
        Returns:
            SPECTProjMeta: New SPECTProjMeta that corresponds to the subset of projections
        """
        indices_array = self.subset_indices_array[subset_idx]
        proj_meta_new = copy(self.proj_meta)
        proj_meta_new.angles = proj_meta_new.angles[indices_array]
        proj_meta_new.radii = proj_meta_new.radii[indices_array.cpu().numpy()]
        proj_meta_new.shape = (len(indices_array), proj_meta_new.shape[1], proj_meta_new.shape[2])
        proj_meta_new.padded_shape = (len(indices_array), proj_meta_new.padded_shape[1], proj_meta_new.padded_shape[2])
        proj_meta_new.num_projections = len(indices_array)
        return proj_meta_new
        
    def forward(self, object: torch.Tensor, subset_idx: int | None = None):
        """Runs the Monte Carlo scatter simulation using SIMIND and returns the simulated projections.
        Args:
            object (torch.Tensor): Object to simulate
            subset_idx (int | None, optional): Index of the subset to use. If None, then all projections are used. Defaults to None.
        Returns:
            torch.Tensor: Simulated projections
        """
        # subsample proj_meta
        if subset_idx is not None:
            proj_meta = self._get_proj_meta_subset(subset_idx)
        else:
            proj_meta = self.proj_meta
        projections_total = 0
        for isotope_name, isotope_ratio in zip(self.isotope_names, self.isotope_ratios):
            index_dict = get_simind_params_from_metadata(self.object_meta, proj_meta)
            index_dict.update(get_simind_isotope_detector_params(
                isotope_name = isotope_name,
                collimator_type= self.collimator_type,
                crystal_thickness=self.crystal_thickness,
                cover_thickness=self.cover_thickness,
                backscatter_thickness=self.backscatter_thickness,
                energy_resolution_140keV=self.energy_resolution_140keV,
                advanced_energy_resolution_model=self.advanced_energy_resolution_model,
                advanced_collimator_modeling=self.advanced_collimator_modeling
            ))
            projections = run_scatter_simulation(
                object,
                self.attenuation_map_140keV,
                self.object_meta,
                proj_meta,
                self.energy_window_params,
                index_dict,
                self.n_events,
                self.n_parallel,
                return_total=True,
            )[self.primary_window_idx]
            projections_total += projections * object.sum() * isotope_ratio
        # still apply proj2proj transforms since these are only additive term and cutoff
        for transform in self.proj2proj_transforms:
            projections_total = transform.forward(projections_total, padded=False)
        return projections_total