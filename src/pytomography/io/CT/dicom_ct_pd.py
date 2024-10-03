from __future__ import annotations
import pydicom
import struct
import torch
import numpy as np
import pytomography
from pytomography.metadata.CT import CTGen3ProjMeta

def get_geometry_info_from_datasets_gen3(paths):
    projections = []
    source_phis = []
    source_rhos = []
    source_zs = []
    source_phi_offsets = []
    source_rho_offsets = []
    source_z_offsets = []
    detector_centers_phi_idx = []
    detector_centers_z_idx = []
    for path in paths:
        ds = pydicom.dcmread(path)
        source_phis.append(struct.unpack('<f', ds[0x7031,0x1001].value)[0])
        source_zs.append(struct.unpack('<f', ds[0x7031,0x1002].value)[0])
        source_rhos.append(struct.unpack('<f', ds[0x7031,0x1003].value)[0])
        source_phi_offsets.append(struct.unpack('<f', ds[0x7033,0x100B].value)[0])
        source_z_offsets.append(struct.unpack('<f', ds[0x7033,0x100C].value)[0])
        source_rho_offsets.append(struct.unpack('<f', ds[0x7033,0x100D].value)[0])
        center_i, center_j = struct.unpack('<2f', ds[0x7031,0x1033].value)
        detector_centers_phi_idx.append(center_i)
        detector_centers_z_idx.append(center_j)
        data = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        projections.append(torch.tensor(data).to(pytomography.dtype))
    projections = torch.stack(projections)
    source_phis = torch.tensor(source_phis)
    source_rhos = torch.tensor(source_rhos)
    source_zs = torch.tensor(source_zs)
    source_phi_offsets = torch.tensor(source_phi_offsets)
    source_rho_offsets = torch.tensor(source_rho_offsets)
    source_z_offsets = torch.tensor(source_z_offsets)
    detector_centers_phi_idx = torch.tensor(detector_centers_phi_idx)
    detector_centers_z_idx = torch.tensor(detector_centers_z_idx)
    return projections, source_phis, source_rhos, source_zs, source_phi_offsets, source_rho_offsets, source_z_offsets, detector_centers_phi_idx, detector_centers_z_idx

def get_projections_and_metadata_gen3(paths):
    projections, source_phis, source_rhos, source_zs, source_phi_offsets, source_rho_offsets, source_z_offsets, detector_centers_phi_idx, detector_centers_z_idx = get_geometry_info_from_datasets_gen3(paths)
    # assumes all files have same spacing
    ds = pydicom.dcmread(paths[0])
    detector_tranverse_spacing = struct.unpack('<f', ds[0x7029,0x1002].value)[0]
    DSD = struct.unpack('<f', ds[0x7031,0x1031].value)[0]
    phi_det_spacing = np.arcsin(detector_tranverse_spacing/DSD)
    z_det_spacing = struct.unpack('<f', ds[0x7029,0x1006].value)[0]
    proj_meta = CTGen3ProjMeta(source_phis, source_rhos, source_zs, source_phi_offsets, source_rho_offsets, source_z_offsets, detector_centers_phi_idx, detector_centers_z_idx, phi_det_spacing, z_det_spacing, DSD, shape=projections.shape[1:])
    return projections, proj_meta