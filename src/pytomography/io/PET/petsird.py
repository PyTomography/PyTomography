from __future__ import annotations
import pytomography
import torch
from . import prd
from typing import Sequence

def read_petsird(
    petsird_file: str,
    read_tof: bool | None = None,
    read_energy: bool | None = None,
    time_block_ids: Sequence[int] | None = None,
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
        # Read header and build lookup table
        header = reader.read_header()

        # bool that decides whether the scanner has TOF and whether it is
        # meaningful to read TOF
        if read_tof is None:
            r_tof: bool = len(header.scanner.tof_bin_edges) > 1
        else:
            r_tof = read_tof

        # bool that decides whether the scanner has energy and whether it is
        # meaningful to read energy
        if read_energy is None:
            r_energy: bool = len(header.scanner.energy_bin_edges) > 1
        else:
            r_energy = read_energy

        # loop over all time blocks and read all meaningful event attributes
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

    return header, torch.tensor(event_attribute_list).to(pytomography.device)