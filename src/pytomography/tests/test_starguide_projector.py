"""StarGuideSystemMatrix evaluates the translation and the PSF of each view on a slab of the object (the 16 detector
columns plus the PSF reach) with all views of an angle batched, instead of on the whole object one view at a time.
These tests compare it with the previous whole-object implementation (kept here as the reference), check that the
forward and back projections are adjoint, that subsets are consistent, and that the slab adapts to the PSF width."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn.functional import pad

import pytomography
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTPSFMeta, StarGuideProjMeta
from pytomography.projectors.SPECT import StarGuideSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.utils import rotate_detector_z

DEV = pytomography.device
LX, LY, LZ = 40, 40, 12


def _system(psf_params=(0.02, 0.5), n_subsets=3, seed=0, proj2proj_transforms=()):
    gen = torch.Generator().manual_seed(seed)
    dx = 4.0
    object_meta = SPECTObjectMeta(dr=(dx, dx, dx), shape=(LX, LY, LZ))
    angles = torch.tensor(np.repeat([0.0, 33.7, 90.0, 137.2, 200.5, 300.1], 4), dtype=torch.float32)
    offsets = (torch.rand(24, generator=gen) * 2 - 1) * 6 * dx                       # up to 6 voxels either way
    times = 0.5 + torch.rand(24, generator=gen)
    radii = np.array([100 + 10 * i for i in range(24)], dtype=float)
    proj_meta = StarGuideProjMeta((24, 16, LZ), angles.to(DEV), times.to(DEV), offsets.to(DEV), radii)
    mu = (0.015 * torch.rand(LX, LY, LZ, generator=gen)).to(DEV)
    transforms = [SPECTAttenuationTransform(mu, assume_padded=False)]
    if psf_params is not None:
        transforms.append(SPECTPSFTransform(SPECTPSFMeta(psf_params), assume_padded=False))
    sm = StarGuideSystemMatrix(object_meta, proj_meta, obj2obj_transforms=transforms, proj2proj_transforms=list(proj2proj_transforms))
    sm.set_n_subsets(n_subsets)
    f = torch.rand(LX, LY, LZ, generator=gen).to(DEV)
    g = torch.rand(24, 16, LZ, generator=gen).to(DEV)
    return sm, f, g


# ---- the previous implementation (whole object, one view at a time), used as the reference ----
def _reference_forward(sm, object, subset_idx=None):
    angle_subset = sm.subset_indices_array[subset_idx] if subset_idx is not None else None
    N_angles = sm.proj_meta.num_projections if subset_idx is None else len(angle_subset)
    angle_indices = torch.arange(N_angles).to(DEV) if subset_idx is None else angle_subset
    projections = torch.zeros((N_angles, *sm.proj_meta.shape[1:])).to(DEV)
    for angle in torch.unique(sm.proj_meta.angles[angle_indices]):
        idx = sm.proj_meta.angles[angle_indices] == angle
        offsets_i = sm.proj_meta.offsets[angle_indices][idx]
        obj_rotate = rotate_detector_z(object, angles=angle)
        for transform in sm.obj2obj_transforms:
            if type(transform) == SPECTPSFTransform:
                for i, j in enumerate(angle_indices[idx]):
                    obj_rotate[i] = transform.forward(obj_rotate[i], ang_idx=j)
            else:
                obj_rotate = transform.forward(obj_rotate, ang_idx=angle_indices[idx][0]).unsqueeze(0).repeat(len(offsets_i), 1, 1, 1)
        obj_translate_rot = sm._translate_object(obj_rotate, offsets_i / sm.object_meta.dx)
        center = int(obj_translate_rot.shape[2] / 2)
        projections[idx] = obj_translate_rot[:, :, center - 8:center + 8].sum(axis=1)
    return projections * sm.frame_duration_s[angle_indices]


def _reference_backward(sm, proj, subset_idx=None):
    angle_subset = sm.subset_indices_array[subset_idx] if subset_idx is not None else None
    N_angles = sm.proj_meta.num_projections if subset_idx is None else len(angle_subset)
    angle_indices = torch.arange(N_angles).to(DEV) if subset_idx is None else angle_subset
    boundary_box_bp = torch.ones(*sm.object_meta.shape).to(DEV)
    proj_pad = int((sm.object_meta.shape[1] - sm.proj_meta.shape[1]) / 2)
    object = torch.zeros(*sm.object_meta.shape).to(DEV)
    proj = proj * sm.frame_duration_s[angle_indices]
    for angle in torch.unique(sm.proj_meta.angles[angle_indices]):
        idx = sm.proj_meta.angles[angle_indices] == angle
        offsets_i = sm.proj_meta.offsets[angle_indices][idx]
        object_i = pad(proj[idx].unsqueeze(1), [0, 0, proj_pad, proj_pad]) * boundary_box_bp
        object_i = sm._translate_object(object_i, -offsets_i / sm.object_meta.dx)
        for transform in sm.obj2obj_transforms[::-1]:
            if type(transform) == SPECTPSFTransform:
                for i, j in enumerate(angle_indices[idx]):
                    object_i[i] = transform.forward(object_i[i], ang_idx=j)
            else:
                object_i = transform.forward(object_i, ang_idx=angle_indices[idx][0])
        object += torch.stack([rotate_detector_z(o, angles=angle, negative=True) for o in object_i]).sum(axis=0)
    return object


@pytest.mark.parametrize("psf_params", [None, (0.02, 0.05), (0.08, 0.3)])   # no PSF, narrow PSF, PSF wider than the detector
def test_matches_whole_object_implementation(psf_params):
    sm, f, g = _system(psf_params)
    expected_margin = 0 if psf_params is None else max(int(l.layer_r.kernel_size[0]) // 2 for l in sm.obj2obj_transforms[1].layers.values())
    assert sm._psf_margin() == expected_margin                       # the slab adapts to the widest PSF kernel of the data
    for subset_idx in (None, 1):
        gs = g if subset_idx is None else sm.get_projection_subset(g, subset_idx)
        fwd, fwd_ref = sm.forward(f, subset_idx), _reference_forward(sm, f, subset_idx)
        bwd, bwd_ref = sm.backward(gs, subset_idx), _reference_backward(sm, gs, subset_idx)
        assert (fwd - fwd_ref).abs().max() < 1e-5 * fwd_ref.abs().max()
        assert (bwd - bwd_ref).abs().max() < 1e-5 * bwd_ref.abs().max()


@pytest.mark.parametrize("psf_params", [None, (0.02, 0.05)])
def test_forward_and_backward_are_adjoint(psf_params):
    sm, f, g = _system(psf_params)
    for subset_idx in (None, 2):
        gs = g if subset_idx is None else sm.get_projection_subset(g, subset_idx)
        lhs = (sm.forward(f, subset_idx).double() * gs.double()).sum()
        rhs = (f.double() * sm.backward(gs, subset_idx).double()).sum()
        # the residual (~3e-5) is the bilinear rotation, whose transpose is not exactly the rotation by the opposite angle; the previous implementation has the same residual
        assert abs(lhs - rhs) < 2e-4 * abs(lhs)


def test_subsets_are_consistent_with_full_projection():
    sm, f, g = _system()
    full = sm.forward(f)
    back = torch.zeros_like(f)
    for k in range(3):
        assert torch.equal(sm.forward(f, k), full[sm.subset_indices_array[k]])
        back += sm.backward(sm.get_projection_subset(g, k), k)
    assert torch.allclose(back, sm.backward(g), rtol=1e-5, atol=1e-6 * back.abs().max())


def test_projection_loops_do_not_synchronise_with_device():
    if str(DEV) != "cuda":
        pytest.skip("needs CUDA")
    from torch.profiler import ProfilerActivity, profile
    sm, f, g = _system()
    sm.forward(f, 0); sm.backward(sm.get_projection_subset(g, 0), 0)      # builds the per-subset caches
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
        sm.forward(f, 0)
    for fn in (lambda: sm.forward(f, 0), lambda: sm.backward(sm.get_projection_subset(g, 0), 0)):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            fn(); torch.cuda.synchronize()
        counts = {e.key: e.count for e in p.key_averages()}
        assert counts.get("cudaLaunchKernel", 0) > 10
        assert counts.get("cudaStreamSynchronize", 0) == 0


def test_projection_transforms_are_applied():
    """proj2proj_transforms were configured but never applied. Forward projection now ends with them (a mask of the
    detector pixels, then an additive term, on the counts of the subset's views), and back projection starts with
    their adjoints in reverse order (the mask; adding is not part of the adjoint)."""
    from pytomography.transforms.SPECT import AdditiveTermTransform, CutOffTransform
    sm, f, g = _system()
    mask = torch.ones(16, LZ, device=DEV)
    mask[:5] = 0
    mask[:, -2:] = 0
    masked = _system(proj2proj_transforms=[CutOffTransform(mask), AdditiveTermTransform(2.0)])[0]
    for subset_idx in (None, 1):
        gs = g if subset_idx is None else sm.get_projection_subset(g, subset_idx)
        assert torch.allclose(masked.forward(f, subset_idx), sm.forward(f, subset_idx) * mask + 2.0)
        assert torch.allclose(masked.backward(gs, subset_idx), sm.backward(gs * mask, subset_idx))
    assert not torch.allclose(masked.backward(g), sm.backward(g))


@pytest.mark.parametrize("n_subsets, views", [(3, [8, 8, 8]), (4, [8, 8, 4, 4])])
def test_weighting_subset_is_the_share_of_the_views(n_subsets, views):
    """StarGuide had no get_weighting_subset (nor had the SystemMatrix base): its subsets are groups of angles, and a
    subset weighs the share of the views it holds (6 angles of 4 views: 4 subsets hold 2, 2, 1 and 1 angles)."""
    sm = _system(n_subsets=n_subsets)[0]
    assert sm.get_weighting_subset(None) == 1
    assert [sm.get_weighting_subset(k) for k in range(n_subsets)] == pytest.approx([v / 24 for v in views])


@pytest.mark.parametrize("algorithm", ["BSREM", "OSMAPOSL"])
def test_priors_run_on_starguide(algorithm):
    """BSREM, OSMAPOSL and every algorithm with a prior scale the prior by get_weighting_subset: on StarGuide they
    raised an AttributeError."""
    import pytomography.algorithms as algorithms
    from pytomography.likelihoods import PoissonLogLikelihood
    from pytomography.priors import RelativeDifferencePrior
    sm, f, _ = _system(n_subsets=3)
    projections = torch.poisson(20 * sm.forward(f).cpu(), generator=torch.Generator().manual_seed(4)).to(DEV)
    reconstruction = getattr(algorithms, algorithm)(PoissonLogLikelihood(sm, projections), prior=RelativeDifferencePrior(0.1, gamma=2))
    image = reconstruction(n_iters=1, n_subsets=3)
    assert image.shape == (LX, LY, LZ) and torch.isfinite(image).all() and image.sum() > 0
