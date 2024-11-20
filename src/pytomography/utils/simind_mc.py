from __future__ import annotations
from typing import Sequence
import time
import subprocess
import tempfile
import numpy as np
from pytomography.io.SPECT import dicom, simind
import os
from pytomography.callbacks import Callback
from pytomography.likelihoods import Likelihood
from pytomography.metadata import ObjectMeta
from pytomography.metadata.SPECT import SPECTProjMeta
from pytomography.utils.scatter import get_smoothed_scatter
import torch

def save_attenuation_map(
    attenuation_map: torch.Tensor,
    dx: float,
    temp_path: str
):
    """Save attenuation map as binary file to temporary directory for subsequent use by Monte Carlo scatter simulation.

    Args:
        attenuation_map (torch.Tensor): Attenuation map to save
        dx (float): Spacing of attenuation map in cm
        temp_path (str): Temporary folder to save to
    """
    d = (attenuation_map).cpu().numpy().astype(np.float32) * dx
    d_flat = d.swapaxes(0,2).ravel()
    d_flat.tofile(os.path.join(temp_path, f'phantom140_atn_av.bin'))
    
def save_source_map(
    source_map: torch.Tensor,
    temp_path: str
):
    """Save source map as binary file to temporary directory for subsequent use by Monte Carlo scatter simulation.

    Args:
        source_map (torch.Tensor): Source map to save
        temp_path (str): Temporary folder to save to
    """
    d = source_map.cpu().numpy().astype(np.float32)
    d *= 1e3 / d.sum()
    d_flat = d.swapaxes(0,2).ravel()
    d_flat.tofile(os.path.join(temp_path, f'source_act_av.bin'))

def get_simind_params_from_metadata(
    object_meta: ObjectMeta,
    proj_meta: SPECTProjMeta
) -> dict:
    """Obtain dictionary of SIMIND parameters from object and projection metadata

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
    index_dict = {
        '28': proj_meta.dr[0], # pixel spacing
        '29': num_angles, # number of projetions
        '30': index_30,
        '41': proj_meta.angles[0].item(), # first angle
        '76': proj_meta.shape[-2], # number of pixels in projection (X)
        '77': proj_meta.shape[-1], # number of pixels in projection (Y)
        '78': object_meta.shape[0], # X voxels (source)
        '79': object_meta.shape[0], # X voxels (phantom)
        '81': object_meta.shape[1], # Y voxels (source)
        '82': object_meta.shape[1], # Y voxels (phantom)
        '31': object_meta.dr[0], # size of source/CT slices
        '34': object_meta.shape[2], # number of source/CT slices (in this case aligned with SPECT)
        '02': object_meta.dr[2]*object_meta.shape[2]/2, # size of source phantom
        '03': object_meta.dr[0]*object_meta.shape[0]/2, # size of source phantom
        '04': object_meta.dr[1]*object_meta.shape[1]/2, # size of source phantom
        '05': object_meta.dr[2]*object_meta.shape[2]/2, # size of CT phantom
        '06': object_meta.dr[0]*object_meta.shape[0]/2, # size of CT phantom
        '07': object_meta.dr[1]*object_meta.shape[1]/2, # size of CT phantom
    }
    return index_dict

def get_simind_isotope_detector_params(
    isotope_name: str,
    collimator_type: str,
    cover_thickness: float,
    backscatter_thickness: float,
    crystal_thickness: float,
    energy_resolution_140keV: float,
    advanced_collimator_modeling: bool = False,
    random_collimator_movement: bool = False,
) -> dict:
    """Obtain SIMIND parameter dictionary from isotope and detector parameters 

    Args:
        isotope_name (str): Name of isotope used for Monte Carlo scatter simulation
        collimator_type (str): Collimator type used for Monte Carlo scatter simulation. 
        cover_thickness (float): Cover thickness used for simulation. Currently assumes aluminum is used.
        backscatter_thickness (float): Equivalent backscatter thickness used for simulation. Currently assumes pyrex is used.
        energy_resolution_140keV (float): Energy resolution in percent of the detector at 140keV. Currently uses the relationship that resolution is proportional to sqrt(E) for E in keV.
        advanced_collimator_modeling (bool, optional): Whether or not to use advanced collimator modeling that can be used to model septal penetration and scatter. Defaults to False.
        random_collimator_movement (bool, optional): Whether or not to include random collimator movement (e.g. holes are not fixed in place). Defaults to False.

    Returns:
        dict: Dictionary of SIMIND parameters obtainable from isotope and detector parameters
    """
    index_dict = {
        'fi': isotope_name,
        'cc': collimator_type,
        '22': energy_resolution_140keV, # TODO: add energy resolution arbitrary function as argument
        '53': int(advanced_collimator_modeling),
        '59': int(random_collimator_movement),
        '13': cover_thickness, # TODO: add material as argument, default aluminum
        '11': backscatter_thickness, # TODO: add material as argument, default pyrex
        '09': crystal_thickness
    }
    return index_dict

def get_energy_window_params_dicom(
    file_NM: str,
    idxs: Sequence[int]
) -> Sequence[str]:
    """Obtain energy window parameters from a DICOM file: this includes a list of strings which, when written to a file, correspond to a typical "scattwin.win" file used by SIMIND.

    Args:
        file_NM (str): DICOM projection file name
        idxs (Sequence[int]): Indices corresponding to the energy windows to extract. More than one index is provided in cases where multi-photopeak reconstruction is used and scatter needs to be obtained at all windows.

    Returns:
        Sequence[str]: Lines of the "scattwin.win" file corresponding to the energy windows specified by the indices.
    """
    lines = []
    for idx in idxs:
        lower, upper = dicom.get_energy_window_bounds(file_NM, idx)
        lines.append(f'{lower},{upper},0')
    return lines
        
def get_energy_window_params_simind(headerfiles: Sequence[str]|str)-> Sequence[str]:
    """Obtain energy window parameters from a list of SIMIND header files: this includes a list of strings which, when written to a file, correspond to a typical "scattwin.win" file used by SIMIND.

    Args:
        headerfiles (Sequence[str]): SIMIND header files

    Returns:
        Sequence[str]: Lines of the "scattwin.win" file corresponding to the energy windows specified by the header files.
    """
    if type(headerfiles) is str:
        headerfiles = [headerfiles]
    lines = []
    for headerfile in headerfiles:
        lower, upper = simind.get_energy_window_bounds(headerfile)
        lines.append(f'{lower},{upper},0')
    return lines
        
def create_simind_command(index_dict: dict, parallel_idx: int) -> str:
    """Creates the terminal command to run SIMIND with the specified parameters

    Args:
        index_dict (dict): Dictionary of SIMIND parameters
        parallel_idx (int): Random seed used for simulation, used to differentiate between parallel simulations

    Returns:
        str: Terminal command to run SIMIND with the specified parameters
    """
    simind_command = f'temp_output{parallel_idx}/14:-7/15:-7/CA:1/RR:{parallel_idx}/01:-208/in:x2,6x'
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
        # TODO: REMOVE THESE
        # ---------------
        # ---------------
        # ---------------
        subprocess.run(['cp', f'temp_output0.hct', f'/disk1/er165/lu177_SYME_jaszak_lowres/temp_output0.hct'], cwd=temp_path)
        subprocess.run(['cp', f'temp_output0.ict', f'/disk1/er165/lu177_SYME_jaszak_lowres/temp_output0.ict'], cwd=temp_path)
        subprocess.run(['cp', f'temp_output0.res', f'/disk1/er165/lu177_SYME_jaszak_lowres/temp_output0.res'], cwd=temp_path)
                                   
def run_scatter_simulation(
    source_map: torch.Tensor,
    attenuation_map_140keV: torch.Tensor,
    object_meta: ObjectMeta,
    proj_meta: SPECTProjMeta,
    energy_window_params: list,
    primary_window_idxs: Sequence[int],
    simind_index_dict: dict,
    n_events: int,
    n_parallel: int = 1,
    return_total: bool = False
):
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
        _type_: _description_
    """
    temp_dir = tempfile.TemporaryDirectory()
    # Create window file
    with open(os.path.join(temp_dir.name, 'scattwin.win'), 'w') as f:
        f.write('\n'.join(energy_window_params))
    # Radial positions
    np.savetxt(os.path.join(temp_dir.name, f'radii_corfile.cor'), proj_meta.radii)
    # update number of events per parallel simulation
    simind_index_dict.update({'NN':n_events/n_parallel/1e3})
    # Save attenuation map and source map to TEMP directory
    save_attenuation_map(attenuation_map_140keV, object_meta.dr[0], temp_dir.name)
    save_source_map(source_map, temp_dir.name)
    # Move simind.smc and energy_resolution.erf to TEMP directory
    module_path = os.path.dirname(os.path.abspath(__file__))
    smc_filepath = os.path.join(module_path, "../data/simind.smc")
    subprocess.Popen(['cp', smc_filepath, f'{temp_dir.name}/simind.smc']) 
    # Create simind commands and run simind in parallel
    simind_commands = [create_simind_command(simind_index_dict, i) for i in range(n_parallel)]
    procs = [subprocess.Popen([f'simind', 'simind', simind_command, 'radii_corfile.cor'], stdout=subprocess.DEVNULL, cwd=temp_dir.name) for simind_command in simind_commands]
    for p in procs:
        p.wait()
    time.sleep(0.1) # sometimes the last file is not written yet
    # Add together projection data from all seperate processes
    add_together(n_parallel, len(primary_window_idxs), temp_dir.name)
    proj_simind_scatter = simind.get_projections([f'{temp_dir.name}/sca_w{i+1}.h00' for i in range(len(primary_window_idxs))])
    proj_simind_tot = simind.get_projections([f'{temp_dir.name}/tot_w{i+1}.h00' for i in range(len(primary_window_idxs))])
    # Remove data files from temporary directory
    temp_dir.cleanup()
    # Return data
    if return_total:
        return proj_simind_scatter, proj_simind_tot
    else:
        return proj_simind_scatter

class MonteCarloScatterCallback(Callback):
    """Callback used to incorporate Monte Carlo scatter simulation into the reconstruction process

    Args:
        likelihood (Likelihood): Likelihood used in reconstruction
        object_initial (torch.Tensor): Initial object used in reconstruction
        simind_index_dict (dict): SIMIND parameters used for the simulation
        attenuation_map_140keV (torch.Tensor): Attenuation map at 140keV used for the simulation
        calibration_factor (float): Calibration factor (in counts per second per MBq) used for the simulation, must match the calibration factor of the collected data
        energy_window_params (list): List of strings which constitute a typical "scattwin.win" file used by SIMIND
        primary_window_idxs (list): Indices from the energy_window_params list corresponding to indices used as photopeak in reconstruction. For single photopeak reconstruction, this will be a list of length 1, while for multi-photopeak reconstruction, this will be a list of length > 1.
        n_events (_type_, optional): Number of events to use in Monte Carlo Scatter simulation. Defaults to 1e6.
        n_parallel (int, optional): Number of parallel simulation to run. Defaults to 1.
        run_every_iter (int, optional): How often the scatter should be updated in terms of iterations. Defaults to 1.
        run_every_subsets (int, optional): How often the scatter should be updated in terms of subsets. Defaults to 1.
        final_iter (int, optional): Stops updating scatter after this number of iterations. Defaults to np.inf.
        post_smoothing_sigma_r (float, optional): Smooth scatter estimate in r direction after Monte Carlo simulation (specified in cm). Defaults to 0.
        post_smoothing_sigma_z (float, optional): Smooth scatter estimate in z direction after Monte Carlo simulation (specified in cm). Defaults to 0.
    """
    def __init__(
        self,
        likelihood: Likelihood,
        object_initial: torch.Tensor,
        simind_index_dict: dict,
        attenuation_map_140keV: torch.Tensor,
        calibration_factor: float,
        energy_window_params: list,
        primary_window_idxs: list,
        n_events = 1e6,   
        n_parallel = 1,
        run_every_iter = 1,
        run_every_subsets = 1,
        final_iter: int = np.inf, # when to stop updating scatter
        post_smoothing_sigma_r: float = 0,
        post_smoothing_sigma_z: float = 0,
    ):
        self.likelihood = likelihood
        self.object_initial = object_initial
        self.index_dict = simind_index_dict
        self.attenuation_map_140keV = attenuation_map_140keV
        self.calibration_factor = calibration_factor
        self.energy_window_params = energy_window_params
        self.primary_window_idxs = primary_window_idxs
        self.n_events = n_events
        self.n_parallel = n_parallel
        self.run_every_iter = run_every_iter
        self.run_every_subsets = run_every_subsets
        self.final_iter = final_iter
        self.post_smoothing_sigma_r = post_smoothing_sigma_r
        self.post_smoothing_sigma_z = post_smoothing_sigma_z
        self.run_scatter_simulation(object_initial)
        
    def run_scatter_simulation(self, object: torch.Tensor):
        """Runs the Monte Carlo scatter simulation given the reconstruction update ``object``

        Args:
            object (torch.Tensor): Reconstruction updated image estimate
        """
        self.scatter_MC = run_scatter_simulation(
            source_map = object,
            attenuation_map_140keV = self.attenuation_map_140keV,
            object_meta = self.likelihood.system_matrix.object_meta,
            proj_meta = self.likelihood.system_matrix.proj_meta,
            energy_window_params=self.energy_window_params,
            primary_window_idxs = self.primary_window_idxs,
            simind_index_dict = self.index_dict,
            n_events = self.n_events,   
            n_parallel = self.n_parallel,
        ) / self.calibration_factor * object.sum()
        # Smooth scatter if sigmas are given
        self.scatter_MC = get_smoothed_scatter(
            scatter = self.scatter_MC,
            proj_meta = self.likelihood.system_matrix.proj_meta,
            sigma_r = self.post_smoothing_sigma_r,
            sigma_z = self.post_smoothing_sigma_z
        )
        # Update likelihood
        self.likelihood.additive_term = self.scatter_MC
        #print(f'Object sum: {object.sum().item()}')
        #print(f'Scatter sum: {self.scatter_MC.sum().item()}')
        #print('-----------------------------------')
        
    def run(self, object: torch.Tensor, n_iter: int, n_subset: int) -> torch.Tensor:
        """Runs the callback

        Args:
            object (torch.Tensor): Current image estimate
            n_iter (int): Iteration number
            n_subset (int): Subset number

        Returns:
            torch.Tensor: Updated object (not updated in this case)
        """
        if ((n_iter+1) % self.run_every_iter == 0) * ((n_iter+1) < self.final_iter) * ((n_subset+1) % self.run_every_subsets == 0):
            self.run_scatter_simulation(object)
        return object