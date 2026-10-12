"""``listmode_to_sinogram`` bins list mode events with ``torch.bincount`` on the events' own device, instead of
``torch.histogramdd`` on the CPU, which allocated one sinogram per CPU thread. These tests check that the sinograms are
those ``torch.histogramdd`` gave: identical counts, weighted sums to float32 rounding, with and without TOF, and the same
on either device. No projector is needed."""
from __future__ import annotations

import pytest
import torch

from pytomography.io.PET import shared
from pytomography.metadata.PET import PETTOFMeta

INFO = dict(min_rsector_difference=0, crystal_length=20.0, radius=120.0, firstCrystalAxis=0,
            rsectorTransNr=16, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=2, moduleTransSpacing=0.0,
            moduleAxialSpacing=17.0, submoduleTransNr=1, submoduleAxialNr=1, submoduleTransSpacing=0.0,
            submoduleAxialSpacing=0.0, crystalTransNr=4, crystalAxialNr=4, crystalTransSpacing=4.0,
            crystalAxialSpacing=4.0, NrCrystalsPerRing=64, NrRings=8)
N_DETECTORS = INFO['NrCrystalsPerRing'] * INFO['NrRings']
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def _events(n, tof_meta=None, seed=0):
    gen = torch.Generator().manual_seed(seed)
    d0 = torch.randint(0, N_DETECTORS, (n,), generator=gen)
    d1 = (d0 + torch.randint(1, N_DETECTORS, (n,), generator=gen)) % N_DETECTORS
    columns = [d0, d1] + ([torch.randint(0, tof_meta.num_bins, (n,), generator=gen)] if tof_meta else [])
    return torch.stack(columns, dim=1)


def _histogramdd_reference(detector_ids, info, weights=None, normalization=False, tof_meta=None):
    """The implementation before the change (torch.histogramdd on the CPU)."""
    lor_coordinates, sinogram_index = shared.sinogram_coordinates(info)
    tof_bins = detector_ids[:, 2].clone() if tof_meta is not None else None
    ids = detector_ids[:, :2]
    within_ring_id = (ids % info['NrCrystalsPerRing']).to(torch.long)
    ring_ids = (ids // info['NrCrystalsPerRing']).to(torch.long)
    within_ring_id, idx = within_ring_id.sort(axis=1, descending=True)
    ring_ids = ring_ids.gather(index=idx, dim=1)
    bin_edges = [torch.arange(int(info['NrCrystalsPerRing'] / 2) + 1).to(torch.float32) - 0.5,
                 torch.arange(int(info['NrCrystalsPerRing']) + 2).to(torch.float32) - 0.5,
                 torch.arange(int((info['moduleAxialNr'] * info['crystalAxialNr'])**2) + 1).to(torch.float32) - 0.5]
    def coordinates(a, b):
        return torch.concatenate([lor_coordinates[within_ring_id[:, a], within_ring_id[:, b]], sinogram_index[ring_ids[:, a], ring_ids[:, b]].unsqueeze(1)], dim=-1).to(torch.float32)
    if tof_meta is not None:
        tof_bins[idx[:, 0] == 1] = tof_meta.num_bins - 1 - tof_bins[idx[:, 0] == 1]
        data = coordinates(0, 1)
        return torch.stack([torch.histogramdd(data[tof_bins == b], bin_edges, weight=None if weights is None else weights[tof_bins == b])[0]
                            for b in range(tof_meta.num_bins)], dim=-1)
    sinogram = torch.histogramdd(coordinates(0, 1), bin_edges, weight=weights)[0]
    if normalization:
        sinogram += torch.histogramdd(coordinates(1, 0), bin_edges, weight=weights)[0]
        sinogram /= 2
    return sinogram


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tof", [False, True])
def test_counts_are_identical(tof, device):
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3) if tof else None
    events = _events(50000, tof_meta)
    sinogram = shared.listmode_to_sinogram(events.to(device), INFO, tof_meta=tof_meta)
    assert sinogram.device.type == 'cpu' and sinogram.dtype == torch.float32
    assert torch.equal(sinogram, _histogramdd_reference(events, INFO, tof_meta=tof_meta))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tof, normalization", [(False, False), (False, True), (True, False)])
def test_weighted_sums_agree(tof, normalization, device):
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3) if tof else None
    events = _events(50000, tof_meta, seed=1)
    weights = torch.rand(events.shape[0], generator=torch.Generator().manual_seed(2))
    sinogram = shared.listmode_to_sinogram(events.to(device), INFO, weights=weights, normalization=normalization, tof_meta=tof_meta)
    reference = _histogramdd_reference(events, INFO, weights=weights, normalization=normalization, tof_meta=tof_meta)
    assert sinogram.shape == reference.shape
    assert torch.allclose(sinogram, reference, rtol=1e-5, atol=1e-5)
