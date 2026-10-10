"""``sinogram_to_listmode`` reads a sinogram at the events of a list: it must read each event from the bin that
``listmode_to_sinogram`` puts that event in, whichever order the event gives its two crystals in. It did not for events
whose first crystal has the smaller within-ring index: the ring pair was reordered twice, which undid it, and the TOF bin
was not mirrored. ``listmode_to_sinogram`` is the reference because its bins match the geometry of the sinogram system
matrix, which the last test checks. No projector is needed."""
from __future__ import annotations

import pytest
import torch

from pytomography.io.PET import shared
from pytomography.metadata.PET import PETSinogramPolygonProjMeta, PETTOFMeta

INFO = dict(min_rsector_difference=0, crystal_length=20.0, radius=120.0, firstCrystalAxis=0,
            rsectorTransNr=16, rsectorAxialNr=1, moduleTransNr=1, moduleAxialNr=2, moduleTransSpacing=0.0,
            moduleAxialSpacing=17.0, submoduleTransNr=1, submoduleAxialNr=1, submoduleTransSpacing=0.0,
            submoduleAxialSpacing=0.0, crystalTransNr=4, crystalAxialNr=4, crystalTransSpacing=4.0,
            crystalAxialSpacing=4.0, NrCrystalsPerRing=64, NrRings=8)
N_DETECTORS = INFO['NrCrystalsPerRing'] * INFO['NrRings']


def _events(n: int, tof_meta: PETTOFMeta | None, seed: int) -> torch.Tensor:
    """Random coincidences between distinct crystals, in both crystal orders (and TOF bins if ``tof_meta``)."""
    gen = torch.Generator().manual_seed(seed)
    d0 = torch.randint(0, N_DETECTORS, (n,), generator=gen)
    d1 = (d0 + torch.randint(1, N_DETECTORS, (n,), generator=gen)) % N_DETECTORS
    columns = [d0, d1] + ([torch.randint(0, tof_meta.num_bins, (n,), generator=gen)] if tof_meta else [])
    return torch.stack(columns, dim=1)


def _bin(event: torch.Tensor, tof_meta: PETTOFMeta | None) -> tuple:
    """The sinogram bin ``listmode_to_sinogram`` puts a single event in."""
    sinogram = shared.listmode_to_sinogram(event.unsqueeze(0), INFO, tof_meta=tof_meta)
    nonzero = (sinogram > 0).nonzero()
    assert nonzero.shape[0] == 1
    return tuple(nonzero[0].tolist())


@pytest.mark.parametrize("tof", [False, True])
def test_events_are_read_from_the_bin_they_are_binned_in(tof):
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3) if tof else None
    events = _events(400, tof_meta, seed=0)
    shape = (INFO['NrCrystalsPerRing'] // 2, INFO['NrCrystalsPerRing'] + 1, (INFO['moduleAxialNr'] * INFO['crystalAxialNr'])**2)
    sinogram = torch.rand(shape + ((tof_meta.num_bins,) if tof else ()), generator=torch.Generator().manual_seed(1))
    values = shared.sinogram_to_listmode(events, sinogram, INFO)
    expected = torch.stack([sinogram[_bin(event, tof_meta)] for event in events])
    swapped = (events[:, 0] % INFO['NrCrystalsPerRing']) < (events[:, 1] % INFO['NrCrystalsPerRing'])
    assert swapped.any() and (~swapped).any()          # both crystal orders are covered
    assert torch.equal(values, expected)


@pytest.mark.parametrize("tof", [False, True])
def test_device_of_the_events_does_not_matter(tof):
    """Events and sinogram on either device give the same values. That includes the pairs whose two crystals have the
    same within-ring index (in different rings), which an unstable sort orders differently on the GPU."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3) if tof else None
    events = _events(2000, tof_meta, seed=2)
    assert ((events[:, 0] - events[:, 1]) % INFO['NrCrystalsPerRing'] == 0).any()      # such pairs are included
    sinogram = torch.rand(shared.listmode_to_sinogram(events, INFO, tof_meta=tof_meta).shape, generator=torch.Generator().manual_seed(3))
    expected = shared.sinogram_to_listmode(events, sinogram, INFO)
    assert torch.equal(shared.sinogram_to_listmode(events.cuda(), sinogram, INFO), expected)
    assert torch.equal(shared.sinogram_to_listmode(events.cuda(), sinogram.cuda(), INFO).cpu(), expected)


@pytest.mark.parametrize("tof", [False, True])
def test_events_looked_up_in_chunks_equal_all_at_once(tof):
    """The events are looked up a chunk at a time (all 107 million events of the GATE mMR brain scan at once took 8 GB
    of temporaries). Any chunk size, a last partial chunk included, gives the values of one chunk of all the events."""
    tof_meta = PETTOFMeta(5, 300.0, 60.0, n_sigmas=3) if tof else None
    events = _events(1000, tof_meta, seed=4)
    sinogram = torch.rand(shared.listmode_to_sinogram(events, INFO, tof_meta=tof_meta).shape, generator=torch.Generator().manual_seed(5))
    expected = shared._dense_sinogram_to_listmode(events, sinogram, INFO, events_per_chunk=events.shape[0])
    for events_per_chunk in (1, 7, 999):
        assert torch.equal(shared._dense_sinogram_to_listmode(events, sinogram, INFO, events_per_chunk=events_per_chunk), expected)
    assert torch.equal(shared.sinogram_to_listmode(events, sinogram, INFO), expected)


def test_listmode_to_sinogram_follows_the_sinogram_geometry():
    """The line of response of the bin an event is binned in joins that event's two crystals, and its TOF bin is
    mirrored exactly when the bin's line of response runs from the event's second crystal to its first."""
    proj_meta = PETSinogramPolygonProjMeta(INFO)
    lut = shared.get_scanner_LUT(INFO).cpu().double()
    detector_coordinates = proj_meta.detector_coordinates.cpu().double()   # [angle, r, crystal, x/y]
    ring_coordinates = proj_meta.ring_coordinates.cpu().double()           # [plane, crystal] -> z
    for event in _events(300, None, seed=3):
        angle, r, plane = _bin(event, None)
        start = torch.cat([detector_coordinates[angle, r, 0], ring_coordinates[plane, 0:1]])
        end = torch.cat([detector_coordinates[angle, r, 1], ring_coordinates[plane, 1:2]])
        first, second = lut[event[0]], lut[event[1]]
        in_order = (start - first).norm() < 1e-3 and (end - second).norm() < 1e-3
        reversed_ = (start - second).norm() < 1e-3 and (end - first).norm() < 1e-3
        assert in_order or reversed_
        mirrored = bool(event[0] % INFO['NrCrystalsPerRing'] < event[1] % INFO['NrCrystalsPerRing'])
        assert bool(reversed_) == mirrored
