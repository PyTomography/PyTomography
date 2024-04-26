import pandas as pd
import os
import torch
import numpy as np
import h5py

def get_detector_info(scanner_name):
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
    return info

def modify_tof_events(TOF_ids, scanner_name):
    if scanner_name=='discovery_MI':
        TOF_ids = -(TOF_ids// 13) + 14
    return TOF_ids
    
def get_detector_ids_hdf5(listmode_file, scanner_name, return_multiplicative_corrections=False, return_additive_term = False):
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

def get_weights_hdf5(correction_file):
    data = h5py.File(correction_file, 'r')
    weights = torch.tensor(np.array(data['correction_lists/atten'][:] * data['correction_lists/sens'][:]))
    return weights

def get_additive_term_hdf5(correction_file):
    data = h5py.File(correction_file, 'r')
    additive_term = torch.tensor(np.array(data['correction_lists/contam']))
    return additive_term

def get_sensitivity_ids_hdf5(corrections_file, scanner_name):
    data = h5py.File(corrections_file, 'r')
    info = get_detector_info(scanner_name)
    all_ids = data['all_xtals/xtal_ids'][:].astype(np.int32)
    detector_ids_all = torch.stack([
        torch.tensor(info['NrCrystalsPerRing']*all_ids[:,1] + all_ids[:,0]),
        torch.tensor(info['NrCrystalsPerRing']*all_ids[:,3] + all_ids[:,2])
    ], dim=1)
    return detector_ids_all

def get_sensitivity_ids_and_weights_hdf5(corrections_file, scanner_name):
    data = h5py.File(corrections_file, 'r')
    info = get_detector_info(scanner_name)
    all_ids = data['all_xtals/xtal_ids'][:].astype(np.int32)
    detector_ids_all = torch.stack([
        torch.tensor(info['NrCrystalsPerRing']*all_ids[:,1] + all_ids[:,0]),
        torch.tensor(info['NrCrystalsPerRing']*all_ids[:,3] + all_ids[:,2])
    ], dim=1)
    weights_sensitivity = torch.tensor(np.array(data['all_xtals/atten'][:] * data['all_xtals/sens'][:]))
    return detector_ids_all, weights_sensitivity
    
        