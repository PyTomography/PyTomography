"""Regression test: two-bed Lu-177 SPECT reconstruction, stitching and DICOM export, on the tutorial data
(SPECT/Lu177-PSMA-GEDisc). Skipped unless PYTOMOGRAPHY_DATA points at the tutorial data."""
import pydicom
import pytest
import torch

from pytomography.algorithms import OSEM
from pytomography.io.SPECT import dicom
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform

pytestmark = pytest.mark.data


def reconstruct_single_bed(i, projections_all, files_NM, files_CT, index_peak=1, index_lower=3, index_upper=2):
    projections = projections_all[i]
    file_NM = files_NM[i]
    object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=index_peak)
    photopeak = projections[index_peak]
    scatter = dicom.get_energy_window_scatter_estimate_projections(file_NM, projections, index_peak, index_lower, index_upper)
    attenuation_map = dicom.get_attenuation_map_from_CT_slices(files_CT, file_NM, index_peak=index_peak)
    psf_meta = dicom.get_psfmeta_from_scanner_params("GI-MEGP", energy_keV=208)
    system_matrix = SPECTSystemMatrix(
        obj2obj_transforms=[SPECTAttenuationTransform(attenuation_map), SPECTPSFTransform(psf_meta)],
        proj2proj_transforms=[],
        object_meta=object_meta,
        proj_meta=proj_meta)
    likelihood = PoissonLogLikelihood(system_matrix, photopeak, additive_term=scatter)
    return OSEM(likelihood)(n_iters=4, n_subsets=8)


def test_multi_bed_recon(data_dir, tmp_path):
    folder = data_dir / "SPECT" / "Lu177-PSMA-GEDisc"
    if not folder.exists():
        pytest.skip(f"{folder} not found")
    files_NM = [str(folder / "bed1_projections.dcm"), str(folder / "bed2_projections.dcm")]
    files_CT = [str(p) for p in sorted((folder / "CT").glob("*.dcm"))]
    assert pydicom.dcmread(files_NM[0]).EnergyWindowInformationSequence

    projections_all = dicom.load_multibed_projections(files_NM)
    recons = torch.stack([reconstruct_single_bed(i, projections_all, files_NM, files_CT) for i in range(2)])
    recon_stitched = dicom.stitch_multibed(recons=recons, files_NM=files_NM)
    assert torch.isfinite(recon_stitched).all() and recon_stitched.min() >= 0

    save_path = tmp_path / "pytomo_recon"
    dicom.save_dcm(save_path=str(save_path), object=recon_stitched, file_NM=files_NM[0],
                   recon_name="OSEM_4it_8ss", single_dicom_file=True)
    files_recon = list(save_path.glob("**/*.dcm"))
    assert len(files_recon) == 1
    assert "RECON TOMO" in pydicom.dcmread(files_recon[0], force=True).ImageType
