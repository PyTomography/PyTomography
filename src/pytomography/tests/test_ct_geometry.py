"""Geometry details of the third generation CT model: the metadata leaves the tensors it is given unchanged and maps
object voxels to patient coordinates, and the system matrix models only the voxels inside the scan field of view and
starts reconstructions from zero wherever no ray reaches."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pytomography

parallelproj_core = pytest.importorskip("parallelproj_core", reason="CT projection requires parallelproj 2")
from pytomography.algorithms import SART
from pytomography.metadata import ObjectMeta
from pytomography.metadata.CT import CTGen3ProjMeta
from pytomography.projectors.CT import CTGen3SystemMatrix

DEV = pytomography.device


def _inputs(n_views=48, n_cols=48, n_rows=8):
    """Two helical rotations whose focal spot path runs from z = 90 to 110 mm, as read from a DICOM-CT-PD scan."""
    zero = torch.zeros(n_views)
    DSD = 270.0
    return dict(source_phis=torch.linspace(0, 4 * np.pi, n_views + 1)[:-1], source_rhos=torch.full((n_views,), 150.0),
                source_zs=torch.linspace(90, 110, n_views), source_phi_offsets=zero.clone(), source_rho_offsets=zero.clone(),
                source_z_offsets=zero.clone(), detector_centers_col_idx=torch.full((n_views,), (n_cols + 1) / 2),
                detector_centers_row_idx=torch.full((n_views,), (n_rows + 1) / 2),
                col_det_spacing=float(np.arcsin(5.0 / DSD)), row_det_spacing=2.0, DSD=DSD, shape=(n_cols, n_rows))


def test_metadata_leaves_its_inputs_unchanged():
    """The metadata centres and rotates the focal spot path; it used to do so in place, on the caller's tensors, so a
    second metadata built from the same tensors came out shifted and rotated twice."""
    inputs = _inputs()
    before = {k: v.clone() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    first, second = CTGen3ProjMeta(**inputs), CTGen3ProjMeta(**inputs)
    for k, v in before.items():
        assert torch.equal(inputs[k], v), f'{k} was modified'
    assert torch.equal(first.source_focal_spots, second.source_focal_spots)


def test_patient_affine_maps_the_focal_spot_path_back_to_its_axial_position():
    """Object voxels map to patient coordinates: the isocentre in-plane, and along the axis the slice through each focal
    spot position lies at that position's z as read from the files (which is DICOM patient z, checked against the
    scanner's own reconstruction of an FFS scan)."""
    inputs = _inputs()
    raw_z = inputs['source_zs'].double()
    meta = CTGen3ProjMeta(**inputs, patient_position='FFS')
    assert meta.z_center == pytest.approx(100.0)
    object_meta = ObjectMeta(dr=(2.0, 2.0, 3.0), shape=(10, 12, 7))
    A = meta.get_patient_affine(object_meta)
    centre = torch.tensor([(10 - 1) / 2, (12 - 1) / 2, (7 - 1) / 2, 1.0], dtype=torch.float64)
    assert torch.allclose(A @ centre, torch.tensor([0.0, 0.0, 100.0, 1.0], dtype=torch.float64))
    # slice index of each focal spot in the object, then its patient z
    k = meta.source_focal_centers[:, 2].double() / 3.0 + (7 - 1) / 2
    patient_z = A[2, 2] * k + A[2, 3]
    assert torch.allclose(patient_z, raw_z, atol=1e-4)
    with pytest.raises(NotImplementedError):
        CTGen3ProjMeta(**_inputs(), patient_position='HFS').get_patient_affine(object_meta)


def _system(shape, fov_mask=True):
    return CTGen3SystemMatrix(ObjectMeta(dr=(2.0, 2.0, 2.0), shape=shape), CTGen3ProjMeta(**_inputs()), fov_mask=fov_mask)


def _radius(shape):
    x = (torch.arange(shape[0]) - (shape[0] - 1) / 2) * 2.0
    y = (torch.arange(shape[1]) - (shape[1] - 1) / 2) * 2.0
    return torch.sqrt(x[:, None] ** 2 + y[None, :] ** 2)[:, :, None].expand(shape).to(DEV)


def test_voxels_outside_the_field_of_view_are_not_modelled():
    """The image (160 mm square) is larger than the field of view the fans cover (radius 63 mm). An object outside the
    field of view projects to nothing with the mask, and back projections are zero there; without it, it is seen."""
    shape = (80, 80, 12)
    sm, unmasked = _system(shape), _system(shape, fov_mask=False)
    assert sm.fov_radius == pytest.approx(150.0 * np.sin(23.5 * np.arcsin(5.0 / 270.0)))
    outside = (_radius(shape) > sm.fov_radius + 2).float()
    assert sm.forward(outside).abs().max() == 0
    assert unmasked.forward(outside).abs().max() > 0
    BP = sm.backward(torch.ones(sm.proj_meta.N_angles, *sm.proj_meta.shape, device=DEV))
    assert BP[_radius(shape) > sm.fov_radius].abs().max() == 0


def test_reconstruction_starts_from_zero_where_no_ray_reaches():
    """Voxels no ray reaches are never updated, so they used to keep the initial value of 1: in the corners outside the
    field of view, and in end slices beyond the axial reach of the scan. They now start, and stay, at zero."""
    shape = (80, 80, 40)                                     # 80 mm in z, against a focal spot path of 20 mm
    sm = _system(shape)
    initial = sm._get_object_initial()
    coverage = sm._coverage()
    assert torch.equal(initial > 0, coverage > 0)
    assert initial[:, :, 0].sum() == 0 and initial[:, :, -1].sum() == 0              # beyond the axial reach
    assert initial[_radius(shape) > sm.fov_radius].sum() == 0                          # outside the field of view
    assert initial[40, 40, 20] == 1
    g = sm.forward(torch.rand(shape, generator=torch.Generator().manual_seed(0)).to(DEV) * (_radius(shape) < 40))
    recon = SART(sm, g)(n_iters=1, n_subsets=4)
    assert torch.isfinite(recon).all()
    assert recon[initial == 0].abs().max() == 0
