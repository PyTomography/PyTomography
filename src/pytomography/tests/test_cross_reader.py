"""The cross-reader consistency test (design section 7): one analytic phantom, three spheres at known LPS positions and
an L-shaped marker that breaks every symmetry, through every image format and through a scan reader.

* Images: written by save_dicom (CT, PT, NM, each in its own orientation), save_nifti, and Interfile and NRRD files
  written here; each is read by load_image and put .on() one grid. They all agree within interpolation error, and every
  sphere's centroid is where the phantom put it, within 0.1 voxel.
* Scans: the phantom's DICOM-CT-PD views, from line integrals along rays placed by the DICOM-CT-PD convention in the
  patient's frame (the one verified on C145), read by read_dicom_ct_pd, reconstructed (FBP), saved with save_nifti
  and read back by load_image: every sphere within one voxel of where it is in the patient.

CPU only, a few hundred MB."""
from __future__ import annotations

import numpy as np
import pydicom
import pytest
import torch

from pytomography import io
from pytomography.io import Image
from pytomography.metadata import ObjectMeta

SPHERES = [((12.0, -6.0, 4.0), 6.0, 1.0), ((-10.0, 9.0, -6.0), 5.0, 2.0), ((2.0, 4.0, 14.0), 4.0, 3.0)]
CENTRE = np.array([20.0, -30.0, 40.0])               # where the phantom (and its grids) sit in the patient (LPS mm)
CT_CENTRE = np.array([5.0, -8.0, 40.0])              # the phantom of the CT scan, near the isocentre


def _phantom(meta: ObjectMeta, centre=CENTRE) -> np.ndarray:
    """The phantom on a grid: the spheres (values 1, 2, 3; centres relative to ``centre``) and an L of value 0.5."""
    x, y, z = [meta.centre[a] + (np.arange(n) - (n - 1) / 2) * d - centre[a]
               for a, (n, d) in enumerate(zip(meta.shape, meta.dr))]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    image = np.zeros(meta.shape, np.float32)
    leg = (np.abs(Z + 14) < 2.5) & (np.abs(Y + 14) < 2.5) & (X > -16) & (X < 6)       # the L: one leg along x...
    leg |= (np.abs(Z + 14) < 2.5) & (np.abs(X + 14) < 2.5) & (Y > -16) & (Y < 2)      # ...and one along y
    image[leg] = 0.5
    for (cx, cy, cz), radius, value in SPHERES:
        image[(X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= radius ** 2] = value
    return image


def _centroids(image: np.ndarray, meta: ObjectMeta, centre=CENTRE) -> list:
    """The centroid (LPS mm, relative to ``centre``) of each sphere: of the voxels within its radius + 3 mm, weighted by
    their value above the background."""
    x, y, z = [meta.centre[a] + (np.arange(n) - (n - 1) / 2) * d - centre[a]
               for a, (n, d) in enumerate(zip(meta.shape, meta.dr))]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    found = []
    for (cx, cy, cz), radius, value in SPHERES:
        near = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= (radius + 3) ** 2
        w = np.clip(image, 0, None) * near
        found.append(np.array([(X * w).sum(), (Y * w).sum(), (Z * w).sum()]) / w.sum())
    return found


def _permuted(image: np.ndarray, affine: np.ndarray, order, flips) -> Image:
    """The same image stored with its axes permuted and flipped (and the affine that says so)."""
    data, A = image, affine.copy()
    for axis in flips:
        A[:3, 3] += A[:3, axis] * (data.shape[axis] - 1)
        A[:3, axis] *= -1
        data = np.flip(data, axis)
    return Image(np.ascontiguousarray(np.transpose(data, order)), np.column_stack([A[:, list(order)], A[:, 3]]))


def _interfile(folder, image: np.ndarray, dr) -> str:
    """A Siemens-style Interfile image (x fastest), centred on the scanner: no position in the file."""
    (folder / "phantom.v").write_bytes(np.ascontiguousarray(image.transpose(2, 1, 0)).astype("<f4").tobytes())
    lines = ["!INTERFILE:=", "%SMS-MI header name space:=image subheader", "!name of data file:=phantom.v",
             "image data byte order:=LITTLEENDIAN", "number format:=float", "!number of bytes per pixel:=4",
             *[f"matrix size[{k + 1}]:={n}" for k, n in enumerate(image.shape)],
             *[f"scale factor (mm/pixel) [{k + 1}]:={d}" for k, d in enumerate(dr)], "!END OF INTERFILE:="]
    (folder / "phantom.v.hdr").write_text("\n".join(lines))
    return folder / "phantom.v.hdr"


def _nrrd(folder, image: Image) -> str:
    """A NRRD in RAS space, as 3D Slicer may write one."""
    A = np.diag([-1.0, -1.0, 1.0, 1.0]) @ image.affine                                 # LPS to RAS
    directions = " ".join("(" + ",".join(f"{v:.6f}" for v in A[:3, k]) + ")" for k in range(3))
    header = ["NRRD0004", "type: float", "dimension: 3", "space: right-anterior-superior",
              f"sizes: {' '.join(str(n) for n in image.shape)}", f"space directions: {directions}",
              "kinds: domain domain domain", "endian: little", "encoding: raw",
              "space origin: (" + ",".join(f"{v:.6f}" for v in A[:3, 3]) + ")"]
    data = np.ascontiguousarray(image.data.transpose(2, 1, 0)).astype("<f4").tobytes()
    (folder / "phantom.nrrd").write_bytes(("\n".join(header) + "\n\n").encode() + data)
    return folder / "phantom.nrrd"


def test_every_image_format_puts_the_phantom_in_the_same_place(tmp_path):
    written = ObjectMeta(dr=(2.0, 2.0, 2.5), shape=(28, 26, 20), centre=tuple(CENTRE), frame="patient")
    phantom = _phantom(written)
    image = Image(phantom * 100, written.affine, frame="patient")
    target = ObjectMeta(dr=(2.4, 2.4, 3.0), shape=(22, 22, 16), centre=tuple(CENTRE + [1.0, -1.5, 0.5]), frame="patient")
    files = {
        "CT, x flipped (IOP -1, 0, 0)": io.save_dicom(_permuted(image.data, image.affine, (0, 1, 2), [0]), tmp_path / "ct",
                                                      modality="CT", units="HU"),
        "PT, coronal": io.save_dicom(_permuted(image.data, image.affine, (0, 2, 1), [2]), tmp_path / "pt",
                                     modality="PT", units="Bq/mL"),
        "NM, slices head first": io.save_dicom(_permuted(image.data, image.affine, (0, 1, 2), [2]), tmp_path / "nm",
                                               modality="NM", units="counts"),
        "NIfTI": io.save_nifti(_permuted(image.data, image.affine, (1, 0, 2), [1]), tmp_path / "phantom.nii.gz"),
        "NRRD (RAS)": _nrrd(tmp_path, _permuted(image.data, image.affine, (2, 1, 0), [0])),
    }
    on_target = {}
    for name, written_files in files.items():
        source = written_files if isinstance(written_files, list) and len(written_files) > 1 else (
            written_files[0] if isinstance(written_files, list) else written_files)
        loaded = io.load_image(source)
        on_target[name] = loaded.on(target).cpu().numpy() / 100
    # Interfile has no position: it is centred on the scanner, so it goes on a grid in the scanner frame
    interfile = io.load_image(_interfile(tmp_path, image.data, written.dr))
    centred_target = ObjectMeta(target.dr, target.shape, centre=tuple(np.array(target.centre) - CENTRE), frame="scanner")
    on_target["Interfile (centred)"] = interfile.on(centred_target).cpu().numpy() / 100
    truth = np.array(_centroids(phantom, written))    # the spheres as the written voxels hold them
    assert np.abs(truth - np.array([c for c, _, _ in SPHERES])).max() < 0.7
    reference = on_target["NIfTI"]
    for name, values in on_target.items():
        assert np.abs(values - reference).max() < 0.02, name                  # all the same, to the stored precision
        found = np.array(_centroids(values, target))    # the centred Interfile grid holds the target's voxels
        assert np.abs(found - truth).max() < 0.1 * min(target.dr), (name, found - truth)


def _ct_pd_views(folder, n_views=640, n_cols=96, n_rows=16, feed=14.0, z0=40.0):
    """DICOM-CT-PD views of the phantom (GE, FFS) over four turns, their line integrals computed along rays that the
    DICOM-CT-PD convention puts in the patient's frame: the focal spot centre of a view of angle phi at
    (rho sin(phi), -rho cos(phi), z), detector element (c, r) at the focal centre - DSD (cos(phi' + gamma_c),
    sin(phi' + gamma_c), 0) - (0, 0, (r - r0) dz), phi' = phi - pi/2, gamma_c = (c - c0) transverse spacing / DSD."""
    folder.mkdir()
    rho, dsd, spacing, dz_det = 300.0, 540.0, 2.0, 2.0
    c0_tag, r0 = (n_cols + 1) / 2, (n_rows + 1) / 2
    c0 = n_cols - c0_tag                                                   # the GE rule (n - tag)
    phis = np.arange(n_views) * (2 * np.pi / 160)
    zs = z0 + feed / (2 * np.pi) * (phis - phis.mean())
    gamma = (np.arange(1, n_cols + 1) - c0) * spacing / dsd
    rows = -(np.arange(1, n_rows + 1) - r0) * dz_det
    for v, (phi, z) in enumerate(zip(phis, zs)):
        p = phi - np.pi / 2
        focal = np.array([rho * np.cos(p), rho * np.sin(p), z])
        det = np.stack(np.broadcast_arrays(focal[0] - dsd * np.cos(p + gamma)[:, None],
                                           focal[1] - dsd * np.sin(p + gamma)[:, None],
                                           focal[2] + rows[None, :]), axis=-1)      # (columns, rows, 3)
        d = det - focal
        length = np.linalg.norm(d, axis=-1)
        u = d / length[..., None]
        line = np.zeros(length.shape)
        for (cx, cy, cz), radius, value in SPHERES:
            oc = focal - (np.array([cx, cy, cz]) + CT_CENTRE)
            b = (u * oc).sum(-1)
            disc = b ** 2 - ((oc ** 2).sum() - radius ** 2)
            s = np.sqrt(np.clip(disc, 0, None))
            line += 0.01 * value * np.clip(np.minimum(-b + s, length) - np.clip(-b - s, 0, None), 0, None) * (disc > 0)
        stored = np.round(line / 1e-4).astype(np.uint16)                   # RescaleSlope 1e-4
        ds = pydicom.Dataset()
        ds.InstanceNumber, ds.Manufacturer, ds.PatientPosition = v + 1, "GE MEDICAL SYSTEMS", "FFS"
        ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.PhotometricInterpretation = n_cols, n_rows, 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.RescaleSlope, ds.RescaleIntercept = 1e-4, 0
        ds.PixelData = stored.tobytes()
        for tag_, vr, value in (((0x7031, 0x1001), "FL", phi), ((0x7031, 0x1002), "FL", z), ((0x7031, 0x1003), "FL", rho),
                                ((0x7031, 0x1031), "FL", dsd), ((0x7031, 0x1033), "FL", [c0_tag, r0]),
                                ((0x7029, 0x1002), "FL", spacing), ((0x7029, 0x1006), "FL", dz_det),
                                ((0x7029, 0x100B), "CS", "CYLINDRICAL"), ((0x7037, 0x100A), "CS", "FANBEAM"),
                                ((0x7041, 0x1001), "DS", "0.0186")):
            ds.add_new(tag_, vr, value)
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian      # private tags read back as UN
        ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        ds.save_as(str(folder / f"{v:04d}.dcm"), enforce_file_format=True)
    return folder


def test_a_ct_pd_scan_reconstructs_the_phantom_where_it_is_in_the_patient(tmp_path):
    """DICOM-CT-PD to read_dicom_ct_pd to FBP to save_nifti to load_image: every sphere lands within one voxel of where
    it is in the patient (k runs foot to head: the sphere at +14 mm is above the others)."""
    pytest.importorskip("parallelproj_core")
    from pytomography.algorithms import FilteredBackProjection
    from pytomography.projectors.CT import CTGen3SystemMatrix
    folder = _ct_pd_views(tmp_path / "ctpd")
    grid = ObjectMeta(dr=(1.5, 1.5, 1.5), shape=(48, 48, 28))
    projections, object_meta, proj_meta = io.CT.read_dicom_ct_pd(folder, object_meta=grid, filter_low_signal=False)
    assert object_meta.centre == pytest.approx((0.0, 0.0, 40.0), abs=1e-3)
    image = FilteredBackProjection(projections, CTGen3SystemMatrix(object_meta, proj_meta), filter="hann")()
    saved = io.save_nifti(image, tmp_path / "recon.nii.gz", object_meta)
    back = io.load_image(saved)
    on_phantom_grid = ObjectMeta(dr=(1.5, 1.5, 1.5), shape=(40, 40, 28), centre=tuple(CT_CENTRE), frame="patient")
    values = back.on(on_phantom_grid).cpu().numpy() / 0.01
    found = np.array(_centroids(values, on_phantom_grid, CT_CENTRE))
    truth = np.array([centre for centre, _, _ in SPHERES])
    assert np.abs(found - truth).max() < 1.5, found - truth
