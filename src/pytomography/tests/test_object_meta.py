"""ObjectMeta: the grid of an object and where it is in the patient (LPS, mm). The affine is derived from the grid, so
it cannot go stale (#27); with_grid and subsample keep the centre and the frame."""
import numpy as np
import pytest
import torch

from pytomography.metadata import ObjectMeta
from pytomography.metadata.SPECT import SPECTObjectMeta
from pytomography.utils import plot_utils


def test_affine_is_derived_from_the_grid():
    meta = ObjectMeta(dr=(2.0, 3.0, 4.0), shape=(5, 6, 7), centre=(10.0, -20.0, 300.0))
    A = meta.affine
    assert A.dtype == np.float64 and A.shape == (4, 4)
    assert np.allclose(A[:3, :3], np.diag([2, 3, 4])) and np.allclose(A[3], [0, 0, 0, 1])
    assert np.allclose(A @ [2, 2.5, 3, 1], [10, -20, 300, 1])                 # the centre of the grid
    assert np.allclose(A @ [0, 0, 0, 1], [10 - 4, -20 - 7.5, 300 - 12, 1])    # voxel 0 is the most inferior, right, anterior
    meta.dr = (1.0, 1.0, 1.0)                                                 # a changed grid changes the affine
    assert np.allclose(meta.affine @ [2, 2.5, 3, 1], [10, -20, 300, 1]) and meta.affine[0, 0] == 1.0


def test_defaults_and_validation():
    meta = ObjectMeta(dr=[2, 2, 2], shape=torch.Size([4, 4, 4]))
    assert meta.centre == (0.0, 0.0, 0.0) and meta.frame == 'scanner' and meta.units is None and meta.modality is None
    assert meta.dr == (2.0, 2.0, 2.0) and meta.shape == (4, 4, 4) and isinstance(meta.shape[0], int)
    assert (meta.dx, meta.dy, meta.dz) == (2.0, 2.0, 2.0)
    assert ObjectMeta(dr=torch.tensor([1.0, 2.0, 3.0]), shape=(1, 2, 3), modality='pt').modality == 'PT'
    with pytest.raises(ValueError):
        ObjectMeta(dr=(1, 1), shape=(4, 4, 4))
    with pytest.raises(ValueError):
        ObjectMeta(dr=(1, 1, -1), shape=(4, 4, 4))
    with pytest.raises(ValueError):
        ObjectMeta(dr=(1, 1, 1), shape=(4, 4, 4.5))
    with pytest.raises(ValueError, match='frame'):
        ObjectMeta(dr=(1, 1, 1), shape=(4, 4, 4), frame='world')


def test_with_grid_keeps_the_centre_frame_and_class():
    meta = SPECTObjectMeta(dr=(4.8, 4.8, 4.8), shape=(128, 128, 96), centre=(1.0, -150.0, 400.0), frame='patient')
    new = meta.with_grid(shape=(64, 64, 48))
    assert type(new) is SPECTObjectMeta and new is not meta
    assert new.centre == meta.centre and new.frame == 'patient' and new.units == 'counts' and new.modality == 'NM'
    assert new.shape == (64, 64, 48) and new.dr == meta.dr and meta.shape == (128, 128, 96)   # the original is unchanged
    assert new.padded_shape[0] == 64 + 2 * new.pad_size and meta.padded_shape[0] == 128 + 2 * meta.pad_size
    centre = np.array([*((np.array(new.shape) - 1) / 2), 1])
    assert np.allclose(new.affine @ centre, [*meta.centre, 1])


def test_subsample_keeps_the_centre():
    """#27: the coarse grid's voxels are larger and fewer, its centre where the fine grid's was."""
    meta = ObjectMeta(dr=(2.0, 2.0, 3.0), shape=(10, 11, 8), centre=(5, 6, 7), frame='patient', units='Bq/mL', modality='PT')
    coarse = meta.subsample(2)
    assert coarse.dr == (4.0, 4.0, 6.0) and coarse.shape == (5, 5, 4)
    assert coarse.centre == meta.centre and coarse.frame == 'patient' and coarse.units == 'Bq/mL'
    assert np.allclose(coarse.affine @ [2, 2, 1.5, 1], [5, 6, 7, 1])
    assert meta.subsample((1, 1, 2)).shape == (10, 11, 4)
    with pytest.raises(ValueError):
        meta.subsample(0)


def test_spect_grid_is_in_mm():
    meta = SPECTObjectMeta(dr=(4.8, 4.8, 4.8), shape=(128, 128, 96), centre=(0, -158.1, 414.9))
    assert np.isclose(meta.affine[0, 0], 4.8)
    assert np.allclose(meta.affine @ [63.5, 63.5, 47.5, 1], [0, -158.1, 414.9, 1])
    with pytest.warns(UserWarning, match='mm'):
        SPECTObjectMeta(dr=(0.48, 0.48, 0.48), shape=(8, 8, 8))           # cm, as before PyTomography 4.0


def test_dual_imshow_picks_the_slice_whose_centre_is_nearest():
    """#40: with 4.8 mm SPECT slices over 1.25 mm CT slices, the CT slice is the one whose centre is nearest the SPECT
    slice's centre (the old choice from slice edges was 1.4 slices off)."""
    spect = ObjectMeta(dr=(4.8, 4.8, 4.8), shape=(4, 4, 10), centre=(0, 0, 0)).affine
    ct = ObjectMeta(dr=(1.0, 1.0, 1.25), shape=(4, 4, 40), centre=(0, 0, 0.37)).affine
    for k in range(10):
        z = spect[2, 3] + 4.8 * k
        expected = int(np.argmin(np.abs(ct[2, 3] + 1.25 * np.arange(40) - z)))
        assert plot_utils.matching_slice(k, spect, ct, 2, 40) == expected
    assert plot_utils.matching_slice(0, spect, ObjectMeta(dr=1, shape=(4, 4, 4), centre=(0, 0, 100)).affine, 2, 4) is None


def test_dual_imshow_draws_both_images():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a = ObjectMeta(dr=(4.0, 4.0, 4.0), shape=(8, 8, 8)).affine
    b = ObjectMeta(dr=(1.0, 1.0, 1.0), shape=(32, 32, 32)).affine
    for plane in ('axial', 'coronal', 'sagittal'):
        plt.figure()
        plot_utils.dual_imshow(plane, torch.rand(8, 8, 8), np.random.rand(32, 32, 32), 3, a, b, {}, {'alpha': 0.5})
        assert len(plt.gca().images) == 2
        plt.close()
