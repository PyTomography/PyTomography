from __future__ import annotations
import pytomography
import torch
from . import prd
# import petsird as prd
from pytomography.metadata.PET import PETTOFMeta
from typing import Sequence

def get_detector_ids(
    petsird_file: str,
    read_tof: bool | None = None,
    read_energy: bool | None = None,
    time_block_ids: Sequence[int] | None = None,
    return_header : bool = False
) -> tuple[prd.types.Header, torch.Tensor]:
    """Read all time blocks of a PETSIRD listmode file

    Parameters
    ----------
    petsird_file : str
        the PETSIRD listmode file
    read_tof : bool | None, optional
        read the TOF bin information of every event
        default None means that is is auto determined
        based on the scanner information (length of tof bin edges)
    read_energy : bool | None, optional
        read the energy information of every event
        default None means that is is auto determined
        based on the scanner information (length of energy bin edges)

    Returns
    -------
    tuple[prd.types.Header, torch.Tensor]
        PRD listmode file header, 2D array containing all event attributes
    """
    with prd.BinaryPrdExperimentReader(petsird_file) as reader:
        header = reader.read_header()
        if read_tof is None:
            r_tof: bool = len(header.scanner.tof_bin_edges) > 1
        else:
            r_tof = read_tof
        if read_energy is None:
            r_energy: bool = len(header.scanner.energy_bin_edges) > 1
        else:
            r_energy = read_energy
        event_attribute_list = []
        for time_block in reader.read_time_blocks():
            if (time_block_ids is None) or time_block.id in time_block_ids:
                if r_tof and r_energy:
                    event_attribute_list += [
                        [
                            e.detector_1_id,
                            e.detector_2_id,
                            e.tof_idx,
                            e.energy_1_idx,
                            e.energy_2_idx,
                        ]
                        for e in time_block.prompt_events
                    ]
                elif r_tof and (not r_energy):
                    event_attribute_list += [
                        [
                            e.detector_1_id,
                            e.detector_2_id,
                            e.tof_idx,
                        ]
                        for e in time_block.prompt_events
                    ]
                elif (not r_tof) and r_energy:
                    event_attribute_list += [
                        [
                            e.detector_1_id,
                            e.detector_2_id,
                            e.energy_1_idx,
                            e.energy_2_idx,
                        ]
                        for e in time_block.prompt_events
                    ]
                else:
                    event_attribute_list += [
                        [
                            e.detector_1_id,
                            e.detector_2_id,
                        ]
                        for e in time_block.prompt_events
                    ]
                    
    detector_ids = torch.tensor(event_attribute_list).cpu()
    if return_header:
        return detector_ids, header
    else:
        return detector_ids
    
def get_scanner_LUT_from_header(header: prd.Header) -> torch.Tensor:
    """Obtains the scanner lookup table (relating detector IDs to physical coordinates) from a PETSIRD header.

    Args:
        header (prd.Header): PETSIRD header

    Returns:
        torch.Tensor: scanner lookup table.
    """
    x_pos = [det.x for det in header.scanner.detectors]
    y_pos = [det.y for det in header.scanner.detectors]
    z_pos = [det.z for det in header.scanner.detectors]
    scanner_LUT = torch.vstack([
        torch.tensor(x_pos),
        torch.tensor(y_pos),
        torch.tensor(z_pos)
    ]).T.cpu()
    return scanner_LUT

def get_TOF_meta_from_header(header: prd.Header, n_sigmas: float = 3.) -> PETTOFMeta:
    """Obtain time of flight metadata from a PETSIRD header

    Args:
        header (prd.Header): PETSIRD header
        n_sigmas (float, optional): Number of sigmas to consider when performing TOF projection. Defaults to 3..

    Returns:
        PETTOFMeta: Time of flight metadata.
    """
    num_tof_bins = header.scanner.tof_bin_edges.shape[0] - 1
    tof_range = (header.scanner.tof_bin_edges[-1] - header.scanner.tof_bin_edges[0])
    fwhm_tof = header.scanner.tof_resolution 
    return PETTOFMeta(
        num_bins = num_tof_bins,
        tof_range = tof_range,
        fwhm = fwhm_tof,
        n_sigmas = n_sigmas
    )