"""KEMTransform's top_N: each voxel keeps its top_N largest kernel entries. Without equal entries the voxels keep exactly
what ranking the whole kernel at once kept (in the interior: at the edges the replicated neighbours are equal); of equal
entries the nearest neighbours are kept; and the work goes in chunks of voxels, so a small chunk gives the same kernel."""
from __future__ import annotations

import pytest
import torch

import pytomography
from pytomography.metadata import ObjectMeta
from pytomography.transforms.shared import KEMTransform


def _kem(support, top_N, **kwargs):
    kem = KEMTransform([support], support_kernels_params=[[0.5]], distance_kernel_params=[2.0], size=5, top_N=top_N, **kwargs)
    kem.object_meta = ObjectMeta(dr=(2.0, 2.0, 3.0), shape=support.shape)
    return kem


def _old_kernel(kem):
    """The kernel as it was computed before: every entry of every voxel ranked at once with two argsorts."""
    kem.top_N, top_N = None, kem.top_N
    kem.compute_kernel()                                             # without top_N: normalised and distance-scaled
    kem.top_N = top_N
    n = kem.size**3
    xv, yv, zv = torch.meshgrid([kem.object_meta.dx*kem.idxs, kem.object_meta.dy*kem.idxs, kem.object_meta.dz*kem.idxs], indexing='ij')
    d = torch.sqrt(xv**2+yv**2+zv**2)
    support = kem.kernel / kem.distance_kernel(d, *kem.distance_kernel_params).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    shape = kem.kernel.shape[3:]
    rank = torch.argsort(torch.argsort(support.reshape((n, *shape)), dim=0), dim=0)
    kept = kem.kernel * (rank >= n - top_N).reshape(kem.kernel.shape)
    return kept / kept.sum(dim=(0, 1, 2))


@pytest.mark.parametrize("top_N", [1, 10, 60, 124])
def test_top_n_keeps_what_ranking_everything_kept(top_N):
    torch.manual_seed(0)
    support = torch.rand(9, 8, 7, dtype=pytomography.dtype)          # no two entries equal
    kem = _kem(support, top_N)
    expected = _old_kernel(kem)
    kem.compute_kernel()
    assert ((kem.kernel > 0).sum(dim=(0, 1, 2)) == top_N).all()
    interior = (..., slice(2, -2), slice(2, -2), slice(2, -2))        # 2 voxels from every edge: no neighbour replicated
    torch.testing.assert_close(kem.kernel[interior], expected[interior])


def test_equal_entries_keep_the_nearest_neighbours():
    support = torch.ones(7, 7, 7, dtype=pytomography.dtype)          # every entry equal
    kem = _kem(support, 5)
    kem.compute_kernel()
    centre = kem.kernel[..., 3, 3, 3]
    kept = torch.nonzero(centre > 0) - kem.idx_max
    # the centre and its 4 neighbours along x and y (2 mm; the next are 2.8 mm away, then 3 mm along z): the 5 nearest
    distance = ((kept * torch.tensor([2.0, 2.0, 3.0])) ** 2).sum(dim=1).sqrt()
    assert len(kept) == 5 and distance.max() == 2.0


def test_chunks_give_the_same_kernel():
    torch.manual_seed(1)
    support = torch.rand(6, 5, 4, dtype=pytomography.dtype)
    kem = _kem(support, 20)
    kem.compute_kernel()
    whole = kem.kernel.clone()
    original = KEMTransform._keep_top_N
    try:
        KEMTransform._keep_top_N = lambda self, d: original(self, d, chunk_bytes=125 * 22 * 7)   # 7 voxels a chunk
        kem.compute_kernel()
    finally:
        KEMTransform._keep_top_N = original
    torch.testing.assert_close(kem.kernel, whole)


def test_top_n_of_every_neighbour_keeps_them_all():
    support = torch.rand(5, 5, 5, dtype=pytomography.dtype)
    every = _kem(support, 125)
    every.compute_kernel()
    none = _kem(support, None)
    none.compute_kernel()
    torch.testing.assert_close(every.kernel, none.kernel)
