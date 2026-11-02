"""cardiac_spect.shift_object shifts along x by tx and along y by ty: it applied tx along y and ty along x. With the
shifts get_shift_values measures (x first, then y), shift_object moves the heart to the centre of the slice."""
from __future__ import annotations

import torch

from pytomography.utils.cardiac_spect import get_shift_values, shift_object


def test_shift_object_moves_x_by_tx_and_y_by_ty():
    obj = torch.zeros(20, 16, 12)
    obj[5, 9, 3] = 1.0
    shifted = shift_object(obj, 2.0, -3.0, 1.0)                     # samples at (i + 2, j - 3, k + 1)
    assert tuple(torch.nonzero(shifted.cpu() > 0.5)[0].tolist()) == (3, 12, 2)


def test_measured_shifts_centre_an_off_centre_blob():
    i, j, k = torch.meshgrid(torch.arange(32.0), torch.arange(28.0), torch.arange(8.0), indexing='ij')
    blob = (((i - 9) ** 2 + (j - 19) ** 2) < 9).float() * (k >= 0)
    _, sx, sy = get_shift_values(blob[:, :, 4])                      # the shifts that centre it, along x then y
    centred = shift_object(blob, -sx, -sy, 0.0).cpu()
    com = [(centred[:, :, 4] * c[:, :, 4]).sum() / centred[:, :, 4].sum() for c in (i, j)]
    assert abs(com[0] - 16) < 0.6 and abs(com[1] - 14) < 0.6
