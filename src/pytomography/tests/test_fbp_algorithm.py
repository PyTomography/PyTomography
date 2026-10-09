"""Filtered back projection, the parts that need no projector: the algorithm hands the work to the system matrix and
fails clearly for one without filtered back projection; the filter windows and the ramp kernel; and the corrections of
DICOM-CT-PD projections (low-signal filtering, per-column scale, table feed, view order)."""
from __future__ import annotations

import os
import struct

import numpy as np
import pydicom
import pytest
import torch

from pytomography.algorithms import FilteredBackProjection
from pytomography.io.CT import dicom_ct_pd, preprocessing
from pytomography.metadata import ObjectMeta, ProjMeta
from pytomography.metadata.CT import CTGen3ProjMeta
from pytomography.projectors import SystemMatrix
from pytomography.utils import (HannFilter, RamLakFilter, SheppLoganFilter, TabulatedFilter, get_fbp_filter, gpu_budget,
                                ramp_filter)


class _NoFBP(SystemMatrix):
    def __init__(self):
        super().__init__(ObjectMeta(dr=(1, 1, 1), shape=(4, 4, 4)), ProjMeta(torch.zeros(2)))

    def forward(self, object, subset_idx=None):
        return object

    def backward(self, proj, subset_idx=None):
        return proj


class _RecordingFBP(_NoFBP):
    def _fbp(self, projections, filter, **kwargs):
        self.received = (projections, filter, kwargs)
        return torch.zeros(self.object_meta.shape)


def test_a_system_matrix_without_fbp_raises():
    with pytest.raises(NotImplementedError, match='_NoFBP does not support filtered back projection'):
        FilteredBackProjection(torch.zeros(2, 3), _NoFBP())()


def test_the_algorithm_hands_projections_filter_and_options_to_the_system_matrix():
    sm, proj = _RecordingFBP(), torch.ones(2, 3)
    FilteredBackProjection(proj, sm, filter='shepp-logan', slice_thickness=1.25)()
    projections, filter, kwargs = sm.received
    assert projections is proj and isinstance(filter, SheppLoganFilter) and kwargs == {'slice_thickness': 1.25}


def test_filter_windows():
    f = torch.tensor([0.0, 0.25, 0.5, 0.75])
    assert torch.allclose(RamLakFilter()(f, 0.5), torch.ones(4))
    assert torch.allclose(HannFilter()(f, 0.5), torch.tensor([1.0, 0.5, 0.0, 0.0]), atol=1e-6)
    assert float(SheppLoganFilter()(torch.tensor([0.5]), 0.5)) == pytest.approx(2 / np.pi)
    assert torch.allclose(get_fbp_filter('hamming')(torch.tensor([0.0, 0.5]), 0.5), torch.tensor([1.0, 0.08]), atol=1e-6)
    tab = TabulatedFilter([0.0, 0.4], [1.0, 0.5], taper=0.1)
    assert torch.allclose(tab(torch.tensor([0.2, 0.4, 0.45, 0.6]), 1.0), torch.tensor([0.75, 0.5, 0.25, 0.0]), atol=1e-6)


def test_filter_descriptions():
    assert isinstance(get_fbp_filter(None), RamLakFilter)
    assert isinstance(get_fbp_filter('Hann'), HannFilter)
    assert isinstance(get_fbp_filter(HannFilter), HannFilter)
    f = torch.tensor([0.1, 0.3])
    assert torch.allclose(get_fbp_filter(lambda q: 1 - q)(f, 0.5), 1 - f)
    with pytest.raises(ValueError):
        get_fbp_filter('no-such-filter')
    with pytest.raises(TypeError):
        get_fbp_filter(0.5)


def test_the_ramp_filter_is_the_band_limited_kernel():
    # a unit sample comes back as the kernel times the spacing: 1/(4s) at the sample, -1/(pi k)^2 / s at odd k, 0 at even k
    s = 0.5
    p = torch.zeros(3, 9)
    p[:, 4] = 1
    expected = torch.zeros(9)
    expected[4] = 1 / (4 * s)
    for k in (1, 3):
        expected[[4 - k, 4 + k]] = -1 / (np.pi ** 2 * k ** 2 * s)
    q = ramp_filter(p, s)
    assert torch.allclose(q, expected.expand(3, 9), atol=1e-6)
    assert torch.allclose(ramp_filter(p.T, s, dim=0), q.T)
    # a window multiplies the response: Hann halves it at half the Nyquist frequency
    wave = torch.cos(np.pi / 2 * torch.arange(128.0))
    ratio = ramp_filter(wave, 1.0, HannFilter())[32:96].abs().max() / ramp_filter(wave, 1.0)[32:96].abs().max()
    assert float(ratio) == pytest.approx(0.5, abs=0.02)


def test_gpu_budget_is_the_request_off_cuda():
    assert gpu_budget(2e8, device='cpu') == 2e8


def _poisson(n_photons, p_true, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(n_photons * np.exp(-p_true)).astype(np.float64)
    return -np.log(np.maximum(counts, 0.5) / n_photons)          # zero counts clipped, as a scanner might


def test_low_signal_filter_removes_bias_and_noise_and_leaves_good_rays_alone():
    V, C, R = 60, 200, 16
    n0 = np.full((V, C), 20000.0)
    p_true = np.tile(np.linspace(2, 10, C)[None, :, None], (V, 1, R))    # 2700 down to 0.9 expected photons
    p = torch.tensor(_poisson(n0[:, :, None], p_true), dtype=torch.float32)
    out = preprocessing.filter_low_signal(p, torch.tensor(n0), n_target=30)
    c = int(np.argmin(np.abs(p_true[0, :, 0] - 8.5)))                    # about 4 photons
    sl = (slice(5, -5), slice(c - 1, c + 2), slice(2, -2))
    assert abs(float(p[sl].mean()) - 8.5) > 0.08                         # the raw log is biased...
    assert abs(float(out[sl].mean()) - 8.5) < 0.05                       # ...the filtered one is not
    assert float(out[sl].std()) < 0.4 * float(p[sl].std())
    good = 20000 * np.exp(-p_true) >= 120
    assert torch.equal(out[torch.from_numpy(good)], p[torch.from_numpy(good)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_low_signal_filter_on_the_gpu_matches_the_host():
    V, C, R = 40, 120, 12
    n0 = np.full((V, C), 20000.0)
    p_true = np.tile(np.linspace(2, 10, C)[None, :, None], (V, 1, R))
    p = torch.tensor(_poisson(n0[:, :, None], p_true, seed=1), dtype=torch.float32)
    host = preprocessing.filter_low_signal(p, torch.tensor(n0), 30, device='cpu')
    gpu = preprocessing.filter_low_signal(p, torch.tensor(n0), 30, device='cuda', chunk=16)
    assert gpu.device == p.device
    torch.testing.assert_close(gpu, host, rtol=0, atol=1e-4)
    good = torch.from_numpy(20000 * np.exp(-p_true) >= 120)
    assert torch.equal(gpu[good], p[good])


def _meta(n_views=8, n_cols=21):
    zero = torch.zeros(n_views)
    return CTGen3ProjMeta(torch.linspace(0, 2 * np.pi, n_views), torch.full((n_views,), 500.0), torch.linspace(-5, 5, n_views),
                          zero, zero.clone(), zero.clone(), torch.full((n_views,), (n_cols + 1) / 2), torch.full((n_views,), 2.5),
                          0.01, 1.0, 1000.0, shape=(n_cols, 4))


def test_column_scale_follows_the_distance_from_the_isocentre():
    meta = _meta()
    factors = preprocessing.column_scale(meta, g0=-0.015, g2=0.02, t_hold=40.0)
    t = 500.0 * torch.sin(meta.phis_det[:, 0].double()).abs()
    expected = 1 - 0.015 + 0.02 * (torch.clamp(t, max=40.0) / 100) ** 2
    assert torch.allclose(factors, expected)
    proj = torch.ones(8, 21, 4)
    assert torch.allclose(preprocessing.scale_columns(proj, meta, -0.015, 0.02, 40.0)[3, :, 1], expected.float())


def test_table_feed_of_a_helix():
    phis = torch.linspace(0, 6 * np.pi, 300)
    zs = -40.0 / (2 * np.pi) * phis                                       # 40 mm per rotation, table moving down
    assert dicom_ct_pd._table_feed(phis, zs) == pytest.approx(40.0, rel=1e-6)


def _save(ds, path):
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    try:
        ds.save_as(path, enforce_file_format=True)
    except TypeError:                                                     # pydicom < 3
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(path, write_like_original=False)


def _write_dicom(path, instance):
    ds = pydicom.Dataset()
    ds.InstanceNumber = instance
    _save(ds, path)


def _write_ctpd(path, instance, manufacturer, n_cols=21, n_rows=4, column_tag=11.25, value=0):
    """A minimal DICOM-CT-PD projection: the tags the reader needs and a projection of ``value`` everywhere."""
    ds = pydicom.Dataset()
    ds.InstanceNumber, ds.Manufacturer = instance, manufacturer
    ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.PhotometricInterpretation = n_cols, n_rows, 1, 'MONOCHROME2'
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.RescaleSlope, ds.RescaleIntercept = 1, 0
    ds.PixelData = np.full((n_cols, n_rows), value, np.uint16).tobytes()
    f = lambda *v: struct.pack(f'<{len(v)}f', *v)
    for tag, value in (((0x7031, 0x1001), f(0.1 * instance)), ((0x7031, 0x1002), f(-1.0 * instance)), ((0x7031, 0x1003), f(541.0)),
                       ((0x7033, 0x100B), f(0.0)), ((0x7033, 0x100C), f(0.0)), ((0x7033, 0x100D), f(0.0)),
                       ((0x7031, 0x1033), f(column_tag, 2.5)), ((0x7031, 0x1031), f(949.0)), ((0x7029, 0x1002), f(1.0)),
                       ((0x7029, 0x1006), f(1.0))):
        ds.add_new(tag, 'OB', value)
    _save(ds, path)


def test_views_are_read_in_acquisition_order(tmp_path):
    for name, instance in (('a.dcm', 3), ('b.dcm', 1), ('c.dcm', 2)):
        _write_dicom(os.path.join(tmp_path, name), instance)
    assert [os.path.basename(p) for p in dicom_ct_pd.sorted_paths(str(tmp_path))] == ['b.dcm', 'c.dcm', 'a.dcm']


def test_the_central_column_follows_the_manufacturer(tmp_path):
    # GE counts DetectorCentralElement from the other end of the detector, from zero
    tag = torch.full((3,), 11.25)
    assert torch.equal(dicom_ct_pd._central_column(tag, 21, 'GE MEDICAL SYSTEMS'), 21 - tag)
    assert torch.equal(dicom_ct_pd._central_column(tag, 21, 'SIEMENS'), tag)
    assert torch.equal(dicom_ct_pd._central_column(tag, 21, 'GE', 'tag'), tag)
    assert torch.equal(dicom_ct_pd._central_column(tag, 21, 'GE', 10.5), torch.full((3,), 10.5))
    with pytest.raises(ValueError):
        dicom_ct_pd._central_column(tag, 21, 'GE', 'other')
    for manufacturer, expected in (('GE', 21 - 11.25), ('SIEMENS', 11.25)):
        folder = tmp_path / manufacturer
        folder.mkdir()
        for i in (1, 2, 3):
            _write_ctpd(str(folder / f'{i}.dcm'), i, manufacturer)
        _, meta = dicom_ct_pd.get_projections_and_metadata_gen3(str(folder), low_signal_filter=False)
        assert float(meta.detector_centers_col_idx[0]) == pytest.approx(expected)
    _, meta = dicom_ct_pd.get_projections_and_metadata_gen3(str(tmp_path / 'GE'), low_signal_filter=False, central_column='tag',
                                                           central_column_offset=0.5)
    assert float(meta.detector_centers_col_idx[0]) == pytest.approx(11.75)


def test_the_reader_applies_the_column_scale(tmp_path):
    for i in (1, 2, 3):
        _write_ctpd(str(tmp_path / f'{i}.dcm'), i, 'GE', value=1000)
    plain, meta = dicom_ct_pd.get_projections_and_metadata_gen3(str(tmp_path), low_signal_filter=False)
    scale = dict(g0=-0.01, g2=0.5)
    scaled, _ = dicom_ct_pd.get_projections_and_metadata_gen3(str(tmp_path), low_signal_filter=False, column_scale=scale)
    expected = preprocessing.scale_columns(plain, meta, **scale)
    assert not torch.equal(expected, plain) and torch.equal(scaled, expected)
