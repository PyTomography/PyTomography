from __future__ import annotations
from collections.abc import Sequence
import pandas as pd
import os
import torch
import numpy as np
import h5py
from pytomography.metadata.PET import PETTOFMeta

def get_detector_info(scanner_name: str):
    """Obtains the PET geometry information for a given scanner.

    Args:
        scanner_name (str): Name of the scanner

    Returns:
        dict: PET geometry dictionary required for obtaining lookup table
    """
    module_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(module_path, "../../data/pet_scanner_info.txt")
    df = pd.read_csv(filepath, index_col='scanner_name', skipinitialspace=True)
    info = dict(df.loc[scanner_name])
    # Convert floats to ints
    info['crystalTransNr'] = int(info['crystalTransNr'])
    info['submoduleTransNr'] = int(info['submoduleTransNr'])
    info['moduleTransNr'] = int(info['moduleTransNr'])
    info['rsectorTransNr'] = int(info['rsectorTransNr'])
    info['crystalAxialNr'] = int(info['crystalAxialNr'])
    info['submoduleAxialNr'] = int(info['submoduleAxialNr'])
    info['moduleAxialNr'] = int(info['moduleAxialNr'])
    info['rsectorAxialNr'] = int(info['rsectorAxialNr'])
    info['TOF'] = int(info['TOF'])
    # Get extra info
    info['NrCrystalsPerRing'] = info['crystalTransNr'] * info['submoduleTransNr'] * info['moduleTransNr'] * info['rsectorTransNr']
    info['NrRings'] = info['crystalAxialNr'] * info['submoduleAxialNr'] * info['moduleAxialNr'] * info['rsectorAxialNr']
    info['firstCrystalAxis'] = 0 # first crystal along X axis
    return info

def get_tof_meta(scanner_name: str) -> PETTOFMeta:
    """Obtains the PET TOF metadata for a given scanner

    Args:
        scanner_name (str): Name of the scanner

    Returns:
        PETTOFMeta: PET TOF metadata
    """
    module_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(module_path, "../../data/pet_scanner_info.txt")
    df = pd.read_csv(filepath, index_col='scanner_name', skipinitialspace=True)
    info = dict(df.loc[scanner_name])
    info['num_tof_bins'] = int(info['num_tof_bins'])
    tof_meta = PETTOFMeta(
        num_bins = info['num_tof_bins'],
        tof_range = info['tof_range'],
        fwhm = info['tof_fwhm'],
    )
    return tof_meta

def modify_tof_events(TOF_ids: torch.Tensor, scanner_name: str):
    """Modifies TOF indices based on the scanner name

    Args:
        TOF_ids (torch.Tensor): 1D tensor of TOF indices
        scanner_name (str): Name of scanner

    Returns:
        torch.Tensor: Modified TOF indices
    """
    if scanner_name=='discovery_MI':
        TOF_ids = -(TOF_ids// 13) + 14
    return TOF_ids
    
def get_detector_ids_hdf5(
    listmode_file: str,
    scanner_name: str,
    )-> torch.Tensor:
    """Returns the detector indices obtained from an HDF5 listmode file

    Args:
        listmode_file (str): Path to the listmode file
        scanner_name (str): Name of the PET scanner

    Returns:
        torch.Tensor: Listmode form of the detector IDS for each event
    """
    info = get_detector_info(scanner_name)
    # Get detector ids
    data = h5py.File(listmode_file, 'r')
    events = torch.tensor(np.array(data['MiceList/TofCoinc']).astype(np.int32))
    detector_ids0 = info['NrCrystalsPerRing']*events[:,0] + events[:,1]
    detector_ids1 = info['NrCrystalsPerRing']*events[:,2] + events[:,3]
    if info['TOF']:
        tof_ids = events[:,4]
        tof_ids = modify_tof_events(tof_ids, scanner_name)
        detector_ids = torch.stack([detector_ids0, detector_ids1, tof_ids], dim=1)
    else:
        detector_ids = torch.stack([detector_ids0, detector_ids1], dim=1)
    return detector_ids

def get_weights_hdf5(correction_file: str) -> torch.Tensor:
    """Obtain the multiplicative weights from an HDF5 file that correct for attenuation and sensitivty effects for each of the detected listmode events.

    Args:
        correction_file (str): Path to the correction file

    Returns:
        torch.Tensor: 1D tensor that contains the weights for each listmode event.
    """
    data = h5py.File(correction_file, 'r')
    weights = torch.tensor(np.array(data['correction_lists/atten'][:] * data['correction_lists/sens'][:]))
    return weights

def get_additive_term_hdf5(correction_file: str) -> torch.Tensor:
    """Obtain the additive term from an HDF5 file that corrects for random and scatte effects for each of the detected listmode events.

    Args:
        correction_file (str): Path to the correction file

    Returns:
        torch.Tensor: 1D tensor that contains the additive term for each listmode event.
    """
    data = h5py.File(correction_file, 'r')
    additive_term = torch.tensor(np.array(data['correction_lists/contam']))
    return additive_term

def get_sensitivity_ids_hdf5(corrections_file: str, scanner_name: str) -> torch.Tensor:
    """Obtain the detector indices corresponding to all valid detector pairs (nonTOF): this is used to obtain the sensitivity weights for all detector pairs when computing the normalization factor.

    Args:
        corrections_file (str): Path to the correction file
        scanner_name (str): Name of the scanner

    Returns:
        torch.Tensor[2,N_events]: Tensor yielding all valid detector pairs
    """
    data = h5py.File(corrections_file, 'r')
    info = get_detector_info(scanner_name)
    all_ids = data['all_xtals/xtal_ids'][:].astype(np.int32)
    detector_ids_all = torch.stack([
        torch.tensor(info['NrCrystalsPerRing']*all_ids[:,1] + all_ids[:,0]),
        torch.tensor(info['NrCrystalsPerRing']*all_ids[:,3] + all_ids[:,2])
    ], dim=1)
    return detector_ids_all

def get_sensitivity_ids_and_weights_hdf5(corrections_file: str, scanner_name: str)->Sequence[torch.Tensor, torch.Tensor]:
    """Obtain the detector indices and corresponding detector weights for all valid detector pairs (nonTOF).

    Args:
        corrections_file (str): Path to the correction file
        scanner_name (str): Name of the scanner

    Returns:
        torch.Tensor[2,N_events], torch.Tensor[N_events]: Tensor yielding all valid detector pairs and tensor yielding corresponding weights.
    """
    data = h5py.File(corrections_file, 'r')
    info = get_detector_info(scanner_name)
    all_ids = data['all_xtals/xtal_ids'][:].astype(np.int32)
    detector_ids_all = torch.stack([
        torch.tensor(info['NrCrystalsPerRing']*all_ids[:,1] + all_ids[:,0]),
        torch.tensor(info['NrCrystalsPerRing']*all_ids[:,3] + all_ids[:,2])
    ], dim=1)
    weights_sensitivity = torch.tensor(np.array(data['all_xtals/atten'][:] * data['all_xtals/sens'][:]))
    return detector_ids_all, weights_sensitivity
    
        