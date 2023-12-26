from src.pytomography.io.SPECT import dicom


def test_dicom():
    dcm_file = "tests/jaszczak_disco_c1_j0_fov1.dcm"
    meta = dicom.get_metadata(dcm_file, index_peak=0)
    assert meta is not None
