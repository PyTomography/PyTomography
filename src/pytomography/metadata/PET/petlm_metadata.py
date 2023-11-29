import torch
import pytomography
from pytomography.io.PET import prd

class PETLMProjMeta():
    def __init__(
        self,
        header: prd.Header,
        tof: bool = False,
        n_sigmas_tof: float = 3.,
    ):
        """Computes projection metadata for PET listmode data. Using the header (PETSIRD format), it defines a lookup table between detector IDs and detector coordinates. In addition, if ``tof=True``, then necessary time of flight binning information is additionally stored. 

        Args:
            header (Header): Header obtained from the ``BinaryPrdExperimentReader`` class of the PETSIRD library.
            tof (bool, optional): Whether or not to store time of flight information. Defaults to False.
            n_sigmas_tof (float, optional): Number of sigmas to consider during time of flight projections. Defaults to 3..
        """
        self.scanner_lut = torch.tensor(
            [[det.x, det.y, det.z] for det in header.scanner.detectors],
            dtype=torch.float32,
            device=pytomography.device,
        )
        self.tof = tof
        if tof:
            self.num_tof_bins = header.scanner.tof_bin_edges.shape[0] - 1
            self.tofbin_width = (header.scanner.tof_bin_edges[1] - header.scanner.tof_bin_edges[0])
            self.sigma_tof = torch.tensor([header.scanner.tof_resolution / 2.355], dtype=torch.float32) 
            self.tofcenter_offset = torch.tensor([0], dtype=torch.float32)
            self.nsigmas = n_sigmas_tof