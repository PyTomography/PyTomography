from __future__ import annotations
import pytomography
import torch

class PETTOFMeta():
    """Class for PET time of flight metadata. Contains information such as spatial binning and resolution.

        Args:
            num_bins (int): Number of bins used to discretize time of flight data
            tof_range (float): Total range accross all TOF bins in mm.
            fwhm (float): FWHM corresponding to TOF uncertainty in mm.
            n_sigmas (float): Number of sigmas to consider when using TOF projection. Defaults to 3.
            bin_type (str, optional): How the bins are arranged. Currently, the only option is symmetry, which means the bins are distributed symmetrically (evenly on each side) between the center of all LOR pairs. Defaults to 'symmetric'.
    """
    def __init__(
        self,
        num_bins: int,
        tof_range: float,
        fwhm: float,
        n_sigmas: float = 3,
        bin_type: str = 'symmetric'
    ):  
        # Get bin width from range
        tof_bin_edges = torch.linspace(-1,1,num_bins+1) * tof_range / 2
        bin_width = tof_bin_edges[1] - tof_bin_edges[0]
        # Store data
        self.num_bins = num_bins
        self.bin_width = bin_width
        self.sigma = torch.tensor([fwhm / 2.355], dtype=pytomography.dtype) 
        self.n_sigmas = n_sigmas
        self.bin_edges = tof_bin_edges
        self.bin_positions = (torch.arange(-self.num_bins/2, self.num_bins/2) + 0.5) * self.bin_width
        if bin_type=='symmetric':
            if self.num_bins%2==0:
                self.center_offset = -torch.tensor([self.bin_width/2], dtype=torch.float32)
            else:
                self.center_offset = torch.tensor([0], dtype=pytomography.dtype)
            