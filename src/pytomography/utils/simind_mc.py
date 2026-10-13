"""Running the SIMIND Monte Carlo program to simulate SPECT projections, scatter included.

:class:`~pytomography.projectors.SPECT.MonteCarloHybridSPECTSystemMatrix` uses these functions. SIMIND's parameters
come from the metadata (:func:`get_simind_params_from_metadata`) and from the isotope and the detector
(:func:`get_simind_isotope_detector_params`), its energy windows from a DICOM file
(:func:`get_energy_window_params_dicom`) or from SIMIND headers (:func:`get_energy_window_params_simind`), and
:func:`run_scatter_simulation` writes the maps, runs SIMIND in parallel processes and reads back the projections.

PyTomography's lengths are in mm and SIMIND's in cm; the conversion happens only in this module. SIMIND must be on the
PATH as ``simind``, and so must the ``cp``, ``mv`` and ``sed`` commands this module runs (on Windows they are not there
by default).
"""
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

#: The energy resolution models of SIMIND (``<name>.erf`` files in ``pytomography/data``) that
#: :func:`run_scatter_simulation` copies into SIMIND's folder, for ``advanced_energy_resolution_model``.
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
    """Writes an attenuation map into ``temp_path`` as the binary file SIMIND reads, ``phantom140_atn_av.bin``.

    SIMIND reads the attenuation per voxel (μ times the voxel size), which does not depend on the length unit. The
    file holds it as float32, x varying fastest and z slowest.

    Args:
        attenuation_map (torch.Tensor): Attenuation map [Lx, Ly, Lz], in 1/mm (at 140 keV, as
            :func:`run_scatter_simulation` takes it).
        dx (float): Voxel size, in mm.
        temp_path (str): Folder to write into.
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
    """Writes a source map into ``temp_path`` as the binary file SIMIND reads, ``source_act_av.bin``.

    The map is clipped to [0, ``vmax``], scaled to sum to ``scaling`` and written as float32, x varying fastest and z
    slowest.

    Args:
        source_map (torch.Tensor): Source map [Lx, Ly, Lz].
        temp_path (str): Folder to write into.
        scaling (float): Sum of the map written (:func:`run_scatter_simulation` gives each SIMIND process its share of
            the events).
        vmax (float, optional): Largest value kept, which prevents divergence at early iterations. Defaults to 1e6.
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
    """SIMIND's parameters that come from the object and projection metadata, by SIMIND index.

    They are the pixel size (28), the number of views (29), the direction of rotation (30: 0 when the angles increase, 2
    otherwise), the first angle (41, negated), the projection size (76, 77), the voxels of the source and density maps
    (78, 79, 81, 82, 34), their voxel size (31), the half-sizes of the source and density phantoms (02 to 07), and 08
    and 10 from the object's half-sizes along x and z. PyTomography's lengths are in mm; SIMIND's indices 02 to 10, 28
    and 31 are in cm, so they are converted here.

    Args:
        object_meta (ObjectMeta): Object metadata of the reconstruction.
        proj_meta (SPECTProjMeta): SPECT projection metadata of the reconstruction.

    Returns:
        dict: SIMIND index (a string, e.g. ``'28'``) to value.
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
    """Raises a TypeError for leftover keyword arguments, explaining the renaming of the detector thicknesses to mm.

    Args:
        kwargs (dict): The keyword arguments a function did not take (nothing happens when it is empty).

    Raises:
        TypeError: ``kwargs`` holds a detector thickness under its name before PyTomography 4, in cm
            (``crystal_thickness``, ``cover_thickness`` or ``backscatter_thickness``; the message gives the value in
            mm), or any other keyword.
    """
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
    """SIMIND's parameters that come from the isotope and the detector, by SIMIND index.

    They are the isotope (``fi``), the collimator (``cc``), the energy resolution at 140 keV (22), the collimator
    modelling (53, 59), the thicknesses of the cover (13), the backscatter layer (11) and the crystal (09), and the
    energy resolution model (``Fe``) if one is given. The thicknesses are in mm, like every length in PyTomography;
    SIMIND's indices 09, 11 and 13 are in cm, so they are converted here. The thickness keywords before PyTomography 4
    (``crystal_thickness`` and the others, in cm) raise a TypeError that gives the new value.

    Args:
        isotope_name (str): SIMIND's name of the isotope.
        collimator_type (str): SIMIND's code of the collimator (see
            :func:`~pytomography.utils.print_collimator_parameters`).
        cover_thickness_mm (float): Thickness of the detector's cover, in mm; the cover is taken to be aluminium.
            Required.
        backscatter_thickness_mm (float): Equivalent thickness of the backscattering material behind the crystal, in
            mm; it is taken to be pyrex. Required.
        crystal_thickness_mm (float): Thickness of the crystal, in mm; it is taken to be NaI. Required.
        energy_resolution_140keV (float, optional): Energy resolution (FWHM) of the detector at 140 keV, in percent;
            the FWHM in keV is taken to grow as the square root of the energy. Defaults to 0.
        advanced_energy_resolution_model (str | None, optional): Energy resolution model to use instead of
            ``energy_resolution_140keV`` (which is then set to 0); only ``'siemens'`` is supported. Defaults to None.
        advanced_collimator_modeling (bool, optional): Whether to use SIMIND's advanced collimator modelling, which
            models septal penetration and scatter. Defaults to False.
        random_collimator_movement (bool, optional): Whether the collimator moves at random (its holes are not fixed
            in place). Defaults to False.

    Returns:
        dict: SIMIND index (a string, e.g. ``'09'``) to value.

    Raises:
        TypeError: A thickness is missing, or a keyword is not one of these (:func:`refuse_thicknesses_in_cm`).
        ValueError: ``advanced_energy_resolution_model`` is not ``'siemens'``.
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
    """The energy windows of a SPECT DICOM file, as the lines of SIMIND's window file (``scattwin.win``).

    Args:
        file_NM (str): Path of the DICOM projection (NM) file.
        idxs (Sequence[int] | None, optional): Indices of the windows to keep; more than one for multi-photopeak
            reconstruction, where scatter is needed in every photopeak window. Defaults to None: every window.

    Returns:
        Sequence[str]: One line per window, ``'<lower keV>,<upper keV>,0'``.
    """
    from pytomography.io.SPECT.dicom import read_metadata
    windows = read_metadata(file_NM)[1].energy_windows
    if idxs is None:
        idxs = range(len(windows))
    return [f'{windows[idx].lower_keV},{windows[idx].upper_keV},0' for idx in idxs]
        
def get_energy_window_params_simind(headerfiles: Sequence[str]|str)-> Sequence[str]:
    """The energy windows of SIMIND header files, as the lines of SIMIND's window file (``scattwin.win``).

    Args:
        headerfiles (Sequence[str] | str): SIMIND header (``.h00``) files, one per window, or a single file.

    Returns:
        Sequence[str]: One line per header file, ``'<lower keV>,<upper keV>,0'``.
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
    """The output name and switches of one SIMIND run, as the argument of its command line.

    The switches select voxel source and attenuation maps (XCAT format), scatter and air images, and the simind.ini
    changes in :data:`SIMIND_INI_CHANGES`, followed by ``/<index>:<value>`` for each item of ``index_dict`` (one switch
    per element for a list).

    Args:
        index_dict (dict): SIMIND's parameters, by index.
        parallel_idx (int): Index of the run: SIMIND's random seed (``RR``) and the name of its output files
            (``temp_output<parallel_idx>``), which tell parallel runs apart.

    Returns:
        str: The argument, e.g. ``'temp_output0/14:-7/15:-7/...'``.
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
    """Averages the projections of parallel SIMIND runs, and writes them with headers.

    For each window ``w``, the scatter (``temp_output<i>_sca_w<w>.a00``) and total (``temp_output<i>_tot_w<w>.a00``)
    projections of the runs are averaged into ``sca_w<w>.a00`` and ``tot_w<w>.a00``, and the headers of run 0 are
    renamed and edited to match (``sca_w<w>.h00``, ``tot_w<w>.h00``) with the ``mv`` and ``sed`` commands.

    Args:
        n_parallel (int): Number of runs.
        n_windows (int): Number of energy windows.
        temp_path (str): Folder of the runs' files.
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
    """Runs SIMIND on a source map and an attenuation map, and returns the simulated projections of every window.

    In a temporary folder it writes the energy windows (``simind.win``), the detector distances of the views
    (``radii_corfile.cor``, in cm) and the two maps (:func:`save_attenuation_map`, :func:`save_source_map`), and copies
    ``simind.smc`` and the energy resolution models (:data:`ENERGY_RESOLUTION_MODELS`) from ``pytomography/data``. It
    then runs ``n_parallel`` SIMIND processes with different seeds (:func:`create_simind_command`), averages their
    projections (:func:`add_together`), reads them with :func:`~pytomography.io.SPECT.read_simind` and deletes the
    folder. A process that fails prints its error output; it does not raise.

    Args:
        source_map (torch.Tensor): Source map [Lx, Ly, Lz].
        attenuation_map_140keV (torch.Tensor): Attenuation map [Lx, Ly, Lz] at 140 keV, in 1/mm.
        object_meta (ObjectMeta): Object metadata of the maps (the voxel size along x scales the attenuation).
        proj_meta (SPECTProjMeta): SPECT projection metadata, for the detector distances ``radii`` (mm).
        energy_window_params (list): Lines of SIMIND's window file, one per window (from
            :func:`get_energy_window_params_dicom` or :func:`get_energy_window_params_simind`).
        simind_index_dict (dict): SIMIND's parameters (from :func:`get_simind_params_from_metadata` and
            :func:`get_simind_isotope_detector_params`).
        n_events (int): Number of photons to simulate per projection angle; the source map of each process sums to
            ``n_events / n_parallel``.
        n_parallel (int, optional): Number of SIMIND processes run at once; at most the number of CPU cores. Defaults
            to 1.
        return_total (bool, optional): Return the total projections instead of the scatter projections. Defaults to
            False.

    Returns:
        torch.Tensor: The simulated scatter projections, or the total projections with ``return_total``, [W, θ, r, z]
        (W windows), as :func:`~pytomography.io.SPECT.read_simind` reads them.
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