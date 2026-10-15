"""The CT readers on files written here (CPU): DICOM-CT-PD with its private tags as VR UN and as explicit VR, its
vendor rules and checks, the patient frame with k foot to head; Nikon XTek cone beam (axis order, angle
sign, flat field, binning); HU."""
from __future__ import annotations

import warnings

import numpy as np
import pydicom
import pytest
import torch

from pytomography import io
from pytomography.io._vendors import UnverifiedVendorRule
from pytomography.metadata import ObjectMeta

N_COLS, N_ROWS = 21, 4

# DICOM-CT-PD private tags: (group, element), explicit VR, value of view i
_VIEW_TAGS = {
    (0x7031, 0x1001): ('FL', lambda i: 0.3 * i),               # focal centre angle (rad)
    (0x7031, 0x1002): ('FL', lambda i: 100.0 + 2.0 * i),       # focal centre z (mm): the table moves the patient
    (0x7031, 0x1003): ('FL', lambda i: 541.0),
    (0x7033, 0x100B): ('FL', lambda i: 0.0),
    (0x7033, 0x100C): ('FL', lambda i: 0.0),
    (0x7033, 0x100D): ('FL', lambda i: 0.0),
}


def _write(path, instance, explicit=True, manufacturer='GE', position='FFS', shape_tag='CYLINDRICAL'):
    ds = pydicom.Dataset()
    ds.InstanceNumber, ds.Manufacturer, ds.PatientPosition = instance, manufacturer, position
    ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.PhotometricInterpretation = N_COLS, N_ROWS, 1, 'MONOCHROME2'
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.RescaleSlope, ds.RescaleIntercept, ds.SpiralPitchFactor = 0.001, 0, 1.0
    ds.PixelData = (np.arange(N_COLS * N_ROWS, dtype=np.uint16).reshape(N_COLS, N_ROWS) + instance).tobytes()
    tags = {k: (vr, f(instance)) for k, (vr, f) in _VIEW_TAGS.items()}
    tags.update({(0x7031, 0x1033): ('FL', [11.25, 2.5]), (0x7031, 0x1031): ('FL', 949.0),
                 (0x7029, 0x1002): ('FL', 1.0), (0x7029, 0x1006): ('FL', 1.1),
                 (0x7029, 0x100B): ('CS', shape_tag), (0x7029, 0x1010): ('US', N_ROWS), (0x7029, 0x1011): ('US', N_COLS),
                 (0x7037, 0x1009): ('CS', 'HELICAL'), (0x7037, 0x100A): ('CS', 'FANBEAM'),
                 (0x7041, 0x1001): ('DS', '0.0186')})
    for tag_, (vr, value) in tags.items():
        ds.add_new(tag_, vr, value)
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = (pydicom.uid.ExplicitVRLittleEndian if explicit else pydicom.uid.ImplicitVRLittleEndian)
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.save_as(str(path), enforce_file_format=True)


def _scan(folder, n=6, **options):
    folder.mkdir()
    for i in range(1, n + 1):
        _write(folder / f'{i}.dcm', i, **options)
    return folder


def test_private_tags_as_un_and_as_explicit_vr_read_the_same(tmp_path):
    explicit = _scan(tmp_path / 'explicit', explicit=True)
    implicit = _scan(tmp_path / 'implicit', explicit=False)
    assert isinstance(pydicom.dcmread(str(implicit / '1.dcm'))[0x7031, 0x1001].value, bytes)      # VR UN
    p1, o1, m1 = io.CT.read_dicom_ct_pd(explicit, filter_low_signal=False)
    p2, o2, m2 = io.CT.read_dicom_ct_pd(implicit, filter_low_signal=False)
    assert torch.equal(p1, p2)
    for name in ('source_phis', 'source_zs', 'source_rhos', 'detector_centers_col_idx', 'zs_det', 'phis_det'):
        assert torch.equal(getattr(m1, name), getattr(m2, name)), name
    assert m1.water_attenuation == m2.water_attenuation == pytest.approx(0.0186)
    assert m1.col_angle_spacing == pytest.approx(1.0 / 949.0)            # radians: the arc over the distance
    assert float(m1.detector_centers_col_idx[0]) == pytest.approx(N_COLS - 11.25)   # GE's central column rule
    assert o1.centre == o2.centre


def test_the_frame_is_the_patients_with_k_foot_to_head(tmp_path):
    """The focal spot z of the files is the patient's: the object frame keeps it (centred on z_center), and the grid's
    centre in the patient is (0, 0, z_center)."""
    folder = _scan(tmp_path / 's')
    projections, object_meta, proj_meta = io.CT.read_dicom_ct_pd(folder, filter_low_signal=False,
                                                                 object_meta=ObjectMeta((1.0, 1.0, 0.5), (16, 16, 8)))
    assert projections.shape == (6, N_COLS, N_ROWS) and projections.device.type == 'cpu'
    z = 100.0 + 2.0 * np.arange(1, 7)
    assert proj_meta.z_center == pytest.approx(z.mean())
    assert np.allclose(proj_meta.source_focal_centers[:, 2].numpy() + proj_meta.z_center, z, atol=1e-5)
    assert (object_meta.frame, object_meta.modality, object_meta.units) == ('patient', 'CT', '1/mm')
    assert object_meta.centre == pytest.approx((0.0, 0.0, z.mean()))
    assert object_meta.affine[2, 2] > 0                                    # k runs foot to head
    assert (object_meta.dr, object_meta.shape) == ((1.0, 1.0, 0.5), (16, 16, 8))
    _, default, _ = io.CT.read_dicom_ct_pd(folder, filter_low_signal=False)
    assert default.shape[:2] == (512, 512)
    assert default.dr[2] == pytest.approx(1.1 * 541.0 / 949.0)             # the rows' spacing at the isocentre


def test_scans_the_reader_cannot_model_raise_or_warn(tmp_path):
    with pytest.raises(NotImplementedError, match='cylindrical'):
        io.CT.read_dicom_ct_pd(_scan(tmp_path / 'flat', shape_tag='FLAT'), filter_low_signal=False)
    with pytest.raises(NotImplementedError, match='feet first supine'):
        io.CT.read_dicom_ct_pd(_scan(tmp_path / 'hfs', position='HFS'), filter_low_signal=False)
    with pytest.warns(UnverifiedVendorRule, match='Siemens'):
        _, _, meta = io.CT.read_dicom_ct_pd(_scan(tmp_path / 'siemens', manufacturer='SIEMENS'), filter_low_signal=False)
    assert float(meta.detector_centers_col_idx[0]) == pytest.approx(11.25)          # as stored
    with pytest.warns(UserWarning, match='verified on GE scans only'):
        io.CT.read_dicom_ct_pd(_scan(tmp_path / 'other', manufacturer='Philips'), filter_low_signal=False)
    with pytest.warns(UserWarning, match='PhotonStatistics'):
        io.CT.read_dicom_ct_pd(_scan(tmp_path / 'ge'))                       # the photons cannot be filtered


def test_hounsfield_units():
    mu = torch.tensor([0.0, 0.0186, 0.0372])
    meta = type('Meta', (), {'water_attenuation': 0.0186})()
    assert torch.allclose(io.CT.to_hu(mu, meta), torch.tensor([-1000.0, 0.0, 1000.0]))
    assert torch.allclose(io.CT.from_hu(io.CT.to_hu(mu, 0.0186), 0.0186), mu)
    with pytest.raises(ValueError, match='water'):
        io.CT.to_hu(mu, type('Meta', (), {'water_attenuation': None})())


# ---------- Nikon XTek ----------

def _nikon(folder, n_views=4, rows=8, columns=12):
    """A Nikon scan: readings I = 60000 exp(-(view + column / 100 + row / 1000)) in 16-bit TIFFs."""
    from PIL import Image as PILImage
    folder.mkdir()
    (folder / 'scan.xtekct').write_text('\n'.join([
        '[XTekCT]', 'Name=scan', 'InputSeparator=_', 'SrcToObject=80.0', 'SrcToDetector=1000.0',
        f'DetectorPixelsX={columns}', f'DetectorPixelsY={rows}', 'DetectorPixelSizeX=0.2', 'DetectorPixelSizeY=0.2',
        'WhiteLevel=60000.0', f'Projections={n_views}', 'InitialAngle=0', 'AngularStep=10']))
    angles = [10.0 - 10.0 * i for i in range(n_views)]
    (folder / 'scan.ang').write_text('Proj\tAngle(deg)\n' + ''.join(f'{i + 1}:\t{a:.3f}\n' for i, a in enumerate(angles)))
    truth = []
    for v in range(n_views):
        line = v * 0.1 + np.arange(columns)[None, :] / 100 + np.arange(rows)[:, None] / 1000      # (rows, columns)
        reading = np.round(60000 * np.exp(-line)).astype(np.uint16)
        PILImage.fromarray(reading).save(folder / f'scan_{v + 1:04d}.tif')
        truth.append(-np.log(reading.astype(np.float64) / 60000))
    (folder / 'scan_9999.tif').write_bytes(b'not listed in the .ang file')
    return np.array(truth), np.array(angles)


def test_nikon_axes_angles_and_flat_field(tmp_path):
    truth, angles = _nikon(tmp_path / 'scan')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        projections, object_meta, proj_meta = io.CT.read_nikon(tmp_path / 'scan', cor=0.3)
    assert projections.shape == (4, 12, 8) and projections.dtype == torch.float32      # (views, u = column, v = row)
    assert np.allclose(projections.numpy(), truth.transpose(0, 2, 1), atol=1e-5)
    assert np.allclose(proj_meta.angles.cpu().numpy(), -np.radians(angles), atol=1e-6)  # the stage turns the sample
    assert proj_meta.DSO == pytest.approx(80.0) and proj_meta.DSD == pytest.approx(1000.0)
    assert float(proj_meta.COR[1]) == pytest.approx(-0.3)
    assert object_meta.dr == pytest.approx((0.2 / 12.5,) * 3) and object_meta.shape == (12, 12, 8)
    assert (object_meta.frame, object_meta.modality, object_meta.units) == ('scanner', 'CT', '1/mm')
    binned, binned_meta, binned_proj = io.CT.read_nikon(tmp_path / 'scan', binning=2)
    readings = 60000 * np.exp(-truth)
    assert binned.shape == (4, 6, 4)
    assert np.allclose(binned.numpy(), -np.log(np.round(readings).reshape(4, 4, 2, 6, 2).mean(axis=(2, 4)) / 60000).transpose(0, 2, 1), atol=1e-5)
    assert binned_proj.dr == (0.4, 0.4) and binned_meta.dr == pytest.approx((0.4 / 12.5,) * 3)


@pytest.mark.data
def test_sophiabeads_fdk_reprojects_to_the_measured_projections(data_dir):
    """SophiaBeads (Nikon XT H 225) read by read_nikon (binning 8, COR 0.24 mm): its FDK image, forward projected along
    the library's own rays, matches the measured line integrals (r >= 0.99 over the central detector rows), and the
    detector's rows or columns flipped match worse (the audit's ct_conebeam_check)."""
    import pytomography
    from pytomography.algorithms import FilteredBackProjection
    from pytomography.projectors.CT import CTConeBeamFlatPanelSystemMatrix
    folder = data_dir / 'CT' / 'SophiaBeads-256' / 'SophiaBeads_256_averaged'
    if not folder.is_dir():
        pytest.skip('CT/SophiaBeads-256 is not in the data folder')
    projections, object_meta, proj_meta = io.CT.read_nikon(folder, binning=8, cor=0.24)
    system_matrix = CTConeBeamFlatPanelSystemMatrix(object_meta, proj_meta)
    image = FilteredBackProjection(projections, system_matrix, filter='ram-lak')()
    system_matrix.set_n_subsets(51)
    rows = slice(int(0.2 * projections.shape[2]), int(0.8 * projections.shape[2]))
    for subset in (0, 25):
        model = system_matrix.forward(image.clamp(min=0), subset).cpu()[:, :, rows].numpy()
        measured = system_matrix.get_projection_subset(projections, subset)[:, :, rows].numpy()
        r = np.corrcoef(model.ravel(), measured.ravel())[0, 1]
        assert r >= 0.99, r
        for flipped in (measured[:, ::-1, :], measured[:, :, ::-1]):
            assert np.corrcoef(model.ravel(), flipped.ravel())[0, 1] < r - 0.02
