"""The SPECT projection loops pass Python scalars (angle index, rotation angle) to the rotation and the transforms,
so a forward/backward projection never synchronises with the device inside the per-angle loop. These tests check
that contract, that subsets and reassigned angles still give the right answer, and that the transforms see ints."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pytomography
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms import Transform
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform

DEV = pytomography.device
N, NA = 16, 12


class _RecordingTransform(Transform):
    """Identity obj2obj transform that records the type of every angle index it is given."""
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, object_i, ang_idx):
        self.seen.append(ang_idx)
        return object_i

    def backward(self, object_i, ang_idx):
        self.seen.append(ang_idx)
        return object_i


def _setup(cache, angles_offset=0.0, extra=()):
    dx = 0.4
    object_meta = SPECTObjectMeta([dx] * 3, (N, N, N))
    angles = list(np.linspace(0, 360, NA, endpoint=False) + 0.37 + angles_offset)
    proj_meta = SPECTProjMeta((N, N), [dx, dx], angles, radii=[15.0 + (i % 3) for i in range(NA)])
    gen = torch.Generator().manual_seed(1)
    mu = (0.15 * torch.rand(N, N, N, generator=gen)).to(DEV)
    att = SPECTAttenuationTransform(attenuation_map=mu, cache_probabilities=cache)
    psf = SPECTPSFTransform(psf_meta=SPECTPSFMeta((0.03, 0.1)))
    sm = SPECTSystemMatrix(obj2obj_transforms=[att, psf, *extra], proj2proj_transforms=[], object_meta=object_meta, proj_meta=proj_meta)
    f = torch.rand(N, N, N, generator=gen).to(DEV)
    g = torch.rand(NA, N, N, generator=gen).to(DEV)
    return sm, f, g


def test_transforms_receive_python_int_angle_indices():
    rec = _RecordingTransform()
    sm, f, g = _setup(False, extra=(rec,))
    sm.set_n_subsets(3)
    sm.forward(f); sm.backward(g); sm.forward(f, 1); sm.backward(sm.get_projection_subset(g, 2), 2)
    assert all(type(i) is int for i in rec.seen)
    assert rec.seen[:NA] == list(range(NA)) and rec.seen[2 * NA:2 * NA + 4] == [1, 4, 7, 10]


@pytest.mark.skipif(not torch.cuda.is_available() or str(DEV) != "cuda", reason="needs CUDA to count device synchronisations")
@pytest.mark.parametrize("cache", [False, True])
def test_projection_loops_do_not_synchronise_with_device(cache):
    from torch.profiler import ProfilerActivity, profile
    sm, f, g = _setup(cache)
    sm.set_n_subsets(3)
    sm.forward(f); sm.backward(g); sm.forward(f, 0)              # warm-up fills the caches
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
        sm.forward(f)                                            # the first profiler session of a process records no CUDA runtime events
    for fn in (lambda: sm.forward(f), lambda: sm.backward(g), lambda: sm.forward(f, 1), lambda: sm.backward(sm.get_projection_subset(g, 1), 1)):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            fn(); torch.cuda.synchronize()
        counts = {e.key: e.count for e in p.key_averages()}
        assert counts.get("cudaLaunchKernel", 0) > NA          # the profiler did capture the CUDA runtime
        # A subset call copies its index list to the host once (one cudaStreamSynchronize). The per-angle loop must add
        # none: indexing a device tensor with a 0-d device tensor, or int()/float() of one, would give NA of them.
        assert counts.get("cudaStreamSynchronize", 0) <= 1


@pytest.mark.parametrize("cache", [False, True])
def test_subsets_consistent_with_full_projection(cache):
    sm, f, g = _setup(cache)
    sm.set_n_subsets(4)
    full = sm.forward(f)
    back = torch.zeros_like(f)
    for k in range(4):
        idx = sm.subset_indices_array[k]
        assert torch.equal(sm.forward(f, k), full[idx])
        back += sm.backward(sm.get_projection_subset(g, k), k)
    assert torch.allclose(back, sm.backward(g), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("cache", [False, True])
def test_reassigned_angles_are_picked_up(cache):
    sm, f, g = _setup(cache)
    sm.forward(f); sm.backward(g)                                # host copies of the angles now exist
    sm.proj_meta.angles = sm.proj_meta.angles + 5.0
    fresh, _, _ = _setup(cache, angles_offset=5.0)
    assert torch.equal(sm.forward(f), fresh.forward(f))
    assert torch.equal(sm.backward(g), fresh.backward(g))
