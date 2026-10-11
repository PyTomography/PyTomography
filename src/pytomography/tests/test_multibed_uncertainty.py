"""Stitching weights of dicom.stitch_multibed, which PGAAMultiBedSPECT.compute_uncertainty uses to split a mask
of the stitched image between two overlapping SPECT bed positions.

The beds are cut from one acquisition of a small phantom, the way dicom.load_multibed_projections fills each bed's
projections: the two beds hold the same counts in their overlap. The mask spans the seam of the stitched image.
References:
  * without PSF modelling every slice is reconstructed on its own, so the stitched image, and the uncertainty of its
    counts, are those of one long bed covering both
  * with PSF modelling, the exact linearised variance sum_j y_j (dS/dy_j)^2 of the stitched counts S, with the
    derivative taken by autograd through OSEM written with dense matrices of the same system matrix, through both beds
    and the stitch
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import pytomography
from pytomography.algorithms import OSEM, PGAAMultiBedSPECT
from pytomography.callbacks import DataStorageCallback
from pytomography.io.SPECT import dicom
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTPSFTransform

N, NA, LZ, SHIFT, DX = 8, 4, 12, 7, 0.4   # voxels across, angles, slices per bed, bed shift (slices), voxel size (cm)
LZ_LONG = SHIFT + LZ                      # slices of the stitched image
OFFSETS = [SHIFT, 0]                      # first slice of each bed in the stitched image: the upper bed is given first
N_ITERS, N_SUBSETS = 2, 2
SEAM = 10                                 # first stitched slice taken from the upper bed (midslice of the overlap)


def _system_matrix(Lz, psf):
    object_meta = SPECTObjectMeta([DX] * 3, (N, N, Lz))
    proj_meta = SPECTProjMeta((N, Lz), [DX, DX], list(np.linspace(0, 360, NA, endpoint=False)), radii=[15.0] * NA)
    transforms = [SPECTPSFTransform(psf_meta=SPECTPSFMeta((0.03, 0.1)))] if psf else []
    return SPECTSystemMatrix(obj2obj_transforms=transforms, proj2proj_transforms=[], object_meta=object_meta, proj_meta=proj_meta)


def _bed_files(folder):
    """Stand-ins for the beds' projection files: what stitch_multibed reads (z of each bed, slice spacing, size)."""
    files = []
    for name, offset in zip(['upper', 'lower'], OFFSETS):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.20'
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        detector = Dataset()
        detector.ImagePositionPatient = [0.0, 0.0, 100.0 + offset * DX * 10]
        ds.DetectorInformationSequence = [detector]
        ds.PixelSpacing = [DX * 10, DX * 10]
        ds.Rows = N
        path = str(folder / f'{name}.dcm')
        try:
            ds.save_as(path, enforce_file_format=True)
        except TypeError:  # pydicom < 3
            ds.save_as(path, write_like_original=False)
        files.append(path)
    return files


def _phantom_and_mask():
    """A cylinder with a hot ellipsoid across the seam; the mask is the ellipsoid."""
    x = torch.arange(N) - (N - 1) / 2
    X, Y, Z = torch.meshgrid(x, x, torch.arange(LZ_LONG).float(), indexing='ij')
    mask = (X - 0.5) ** 2 / 6 + (Y + 0.5) ** 2 / 6 + (Z - SEAM + 0.5) ** 2 / 12 <= 1
    phantom = 1.0 * (X ** 2 + Y ** 2 < (0.45 * N) ** 2) + 4.0 * mask
    return phantom.to(pytomography.device), mask.to(pytomography.device)


def _data(psf, shared=True, seed=0):
    """Photopeak and scatter window counts of each bed (as in the files' order), on the long grid if shared."""
    gen = torch.Generator().manual_seed(seed)
    phantom, mask = _phantom_and_mask()
    expected = _system_matrix(LZ_LONG, psf).forward(30 * phantom).cpu()
    scatter_expected = torch.full_like(expected, 0.4 * expected.mean().item())
    if shared:
        y, c = torch.poisson(expected, generator=gen), torch.poisson(scatter_expected, generator=gen)
        beds = [(y[..., o:o + LZ], c[..., o:o + LZ]) for o in OFFSETS]
    else:
        beds = [(torch.poisson(expected[..., o:o + LZ], generator=gen), torch.poisson(scatter_expected[..., o:o + LZ], generator=gen)) for o in OFFSETS]
    return beds, mask


def _likelihood(H, y, c, with_scatter_variance):
    """TEW-like scatter: half the scatter window, with variance 1/4 of its counts."""
    y, c = y.to(pytomography.device), c.to(pytomography.device)
    variance = (lambda x: x * (0.25 * c) * x) if with_scatter_variance else None
    return PoissonLogLikelihood(H, y, additive_term=0.5 * c, additive_term_variance_estimate=variance)


def _multibed(files, beds, psf, with_scatter_variance=False):
    algorithms = [OSEM(_likelihood(_system_matrix(LZ, psf), y, c, with_scatter_variance)) for y, c in beds]
    multibed = PGAAMultiBedSPECT(files, algorithms)
    callbacks = [DataStorageCallback(a.likelihood, torch.clone(a.object_prediction)) for a in algorithms]
    image = multibed(N_ITERS, N_SUBSETS, callback=callbacks)
    return multibed, callbacks, image


def _stitch_weights_z():
    """Stitched slices each bed gives the image: its own half of the overlap, without its first and last slice."""
    w = torch.zeros(2, LZ_LONG)
    w[1, 1:SEAM] = 1
    w[0, SEAM:LZ_LONG - 1] = 1
    return w


def test_stitching_weights_are_each_beds_own(tmp_path):
    files = _bed_files(tmp_path)
    recons = torch.rand(2, N, N, LZ, generator=torch.Generator().manual_seed(1)).to(pytomography.device)
    stitched = dicom.stitch_multibed(recons, files)
    weights, zs = dicom.stitch_multibed(recons, files, return_stitching_weights=True)
    assert tuple(weights.shape) == (2, N, N, LZ)
    assert list(zs) == OFFSETS
    rebuilt = torch.zeros_like(stitched)
    for k in range(2):
        rebuilt[..., zs[k]:zs[k] + LZ] += weights[k] * recons[k]
    assert torch.allclose(rebuilt, stitched)
    expected_z = _stitch_weights_z()
    for k in range(2):
        assert torch.equal(weights[k][0, 0].cpu(), expected_z[k, OFFSETS[k]:OFFSETS[k] + LZ])
