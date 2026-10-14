"""Regression test: two-bed Lu-177 SPECT reconstruction, stitching and DICOM export, on the tutorial data
(SPECT/Lu177-PSMA-GEDisc). Skipped unless PYTOMOGRAPHY_DATA points at the tutorial data."""
import warnings

import numpy as np
import pydicom
import pytest
import torch

from pytomography import io
from pytomography.algorithms import OSEM
from pytomography.io import SPECT
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform

pytestmark = pytest.mark.data

PEAK, LOWER, UPPER = 1, 3, 2           # the 208 keV window and the windows beside it


def reconstruct_single_bed(projections, object_meta, proj_meta, files_CT):
    window = proj_meta.energy_windows[PEAK]
    scatter = SPECT.energy_window_scatter(projections, proj_meta, PEAK, LOWER, UPPER)
    attenuation_map = io.attenuation_map(files_CT, object_meta, window.photon_keV)
    psf_meta = SPECT.collimator_psf("GI-MEGP", window.photon_keV)
    system_matrix = SPECTSystemMatrix(
        obj2obj_transforms=[SPECTAttenuationTransform(attenuation_map), SPECTPSFTransform(psf_meta)],
        proj2proj_transforms=[],
        object_meta=object_meta,
        proj_meta=proj_meta)
    likelihood = PoissonLogLikelihood(system_matrix, projections[PEAK], additive_term=scatter)
    return OSEM(likelihood)(n_iters=4, n_subsets=8)


def test_multi_bed_recon(data_dir, tmp_path):
    folder = data_dir / "SPECT" / "Lu177-PSMA-GEDisc"
    if not folder.exists():
        pytest.skip(f"{folder} not found")
    files_NM = [str(folder / "bed1_projections.dcm"), str(folder / "bed2_projections.dcm")]
    files_CT = [str(p) for p in sorted((folder / "CT").glob("*.dcm"))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", io._vendors.UnverifiedVendorRule)        # GE's NM rule is 'check'
        beds = SPECT.read_dicom_beds(files_NM)
    assert [round(b[1].centre[2], 1) for b in beds] == [-177.0, 177.0]           # the inferior bed first
    assert beds[0][2].energy_windows[PEAK].photon_keV == pytest.approx(208.4)
    recons = [reconstruct_single_bed(*bed, files_CT) for bed in beds]
    recon_stitched, object_meta = SPECT.stitch_beds(recons, [bed[1] for bed in beds])
    assert torch.isfinite(recon_stitched).all() and recon_stitched.min() >= 0
    assert recon_stitched.shape == object_meta.shape

    [path] = io.save_dicom(recon_stitched, tmp_path / "pytomo_recon", object_meta, reference=files_NM[0],
                           series_description="OSEM_4it_8ss")
    ds = pydicom.dcmread(str(path))
    assert "RECON TOMO" in ds.ImageType
    # #28: the stitched image starts at the first slice of the lowest bed, and is saved there
    lowest = beds[0][1]
    first = lowest.affine @ np.array([0, 0, 0, 1.0])
    assert np.allclose([float(v) for v in ds.DetectorInformationSequence[0].ImagePositionPatient], first[:3], atol=1e-3)
