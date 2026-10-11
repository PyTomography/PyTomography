"""#7: the projector's TOF bins are the readers' bins, for an even number of bins too. A point on one LOR, projected
into two TOF bins by parallelproj 2 (on the GPU): it falls in the bin that holds its position along the LOR. With the
offset of -w/2 that PyTomography gave an even number of bins before version 4, a point at the centre of bin 1 sat on
the edge of the projector's bin, and a point at the LOR's midpoint fell in bin 1 alone. GPU memory: a few MB."""
from __future__ import annotations

import pytest
import torch

import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta, PETTOFMeta

pytestmark = pytest.mark.gpu

WIDTH = 40.0     # mm: two bins, [-40, 0] and [0, 40] from the midpoint towards the second crystal
SIGMA = 8.5      # mm


def _bins(x_mm: float) -> torch.Tensor:
    """The two TOF bins of a point at x on the LOR from (-100, 0, 0) (column 0) to (+100, 0, 0) (column 1)."""
    from pytomography.projectors.PET import PETLMSystemMatrix
    meta = ObjectMeta(dr=(1.0, 1.0, 1.0), shape=(121, 21, 21))
    image = torch.zeros(meta.shape, device=pytomography.device)
    image[60 + int(round(x_mm)), 10, 10] = 1.0
    lut = torch.tensor([[-100.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    events = torch.tensor([[0, 1, 0], [0, 1, 1]], dtype=torch.int32)
    tof = PETTOFMeta(num_bins=2, tof_range=2 * WIDTH, fwhm=SIGMA * 2.355, n_sigmas=5)
    proj_meta = PETLMProjMeta(events, scanner_LUT=lut, tof_meta=tof)
    system_matrix = PETLMSystemMatrix(meta, proj_meta, sort_events=False, FOV_scale_enabled=False)
    return system_matrix.forward(image).cpu()


def test_a_point_falls_in_the_bin_that_holds_it():
    centre_of_bin_1 = _bins(+WIDTH / 2)
    assert centre_of_bin_1[1] > 20 * centre_of_bin_1[0]
    centre_of_bin_0 = _bins(-WIDTH / 2)
    assert centre_of_bin_0[0] > 20 * centre_of_bin_0[1]
    midpoint = _bins(0.0)
    assert abs(float(midpoint[0] - midpoint[1])) < 0.02 * float(midpoint.sum())
