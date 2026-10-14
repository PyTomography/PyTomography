"""Time of flight binning of PET data."""
from __future__ import annotations
import pytomography
import torch

class PETTOFMeta():
    r"""Time of flight binning of PET data: the number, width and resolution of the TOF bins, in mm along the LOR.

    Every quantity is a length along the LOR, in mm. Bin :math:`k` of :math:`n` covers
    :math:`[(k - n/2)\,w, (k + 1 - n/2)\,w]`, with :math:`w` = ``tof_range / num_bins``, measured from the LOR's midpoint
    towards the crystal in column 1 of the events (the projector's ``xend``). An event's position is
    :math:`(t_1 - t_2)\,c/2`, with :math:`t_1` the time of column 0's detection (PETSIRD's definition), so its bin is
    the one that holds that position. The bins are centred on the midpoint for any number of bins.

    Args:
        num_bins (int): Number of TOF bins.
        tof_range (float): Total length covered by all the bins, in mm.
        fwhm (float): FWHM of the TOF resolution, in mm.
        n_sigmas (float, optional): How far the TOF kernel reaches in the projector, in sigmas. Defaults to 3.
        bin_type (str, optional): How the bins are arranged. The only option is ``'symmetric'``: the bins are spread
            evenly on each side of the LOR's midpoint. Defaults to ``'symmetric'``.

    Attributes:
        num_bins (int): Number of TOF bins.
        bin_width (torch.Tensor): Width of a bin, in mm.
        sigma (torch.Tensor): Sigma of the TOF resolution (``fwhm / 2.355``), in mm.
        n_sigmas (float): How far the TOF kernel reaches, in sigmas.
        bin_edges (torch.Tensor): The ``num_bins + 1`` bin edges, in mm from the LOR's midpoint.
        bin_positions (torch.Tensor): The bins' centres, in mm from the LOR's midpoint.
        center_offset (torch.Tensor): Offset of the bins from the LOR's midpoint given to the projector (0).

    Raises:
        ValueError: If ``bin_type`` is not ``'symmetric'``.
    """
    def __init__(
        self,
        num_bins: int,
        tof_range: float,
        fwhm: float,
        n_sigmas: float = 3,
        bin_type: str = 'symmetric'
    ):
        if bin_type != 'symmetric':
            raise ValueError(f"the only bin_type is 'symmetric', not {bin_type!r}")
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
        # parallelproj's bin k covers [(k - n/2) w, (k + 1 - n/2) w] + center_offset from the LOR's midpoint, which is
        # bin_edges[k] to bin_edges[k + 1] with no offset, for an odd or an even number of bins. (Before PyTomography 4
        # an even number of bins had center_offset = -w/2, which moved the projector's bins half a bin away from the
        # bins the readers fill.)
        self.center_offset = torch.tensor([0], dtype=pytomography.dtype)
