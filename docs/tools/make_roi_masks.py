"""Make the regions of interest that the SPECT tutorials measure, once, as labelled segmentations (``.seg.nrrd``, LPS
mm). The tutorials load them with ``io.load_mask(file).on(object_meta, label=..., threshold=0.5)`` instead of
finding the regions in code. Each file is on a fine grid of its own, so ``.on()`` gives the fraction of any voxel a
region fills, on any grid. Each file's header says how it was made.

    python docs/tools/make_roi_masks.py OUT_FOLDER

Run it with PyTomography 4 on PYTHONPATH (the version whose read_dicom applies Siemens' registration to the CT) and
PYTOMOGRAPHY_DATA set. It takes under a minute and about 4 GB of RAM, and needs no GPU (CUDA_VISIBLE_DEVICES=-1 keeps
it on the CPU). The files then go with their datasets' records (SPECT tutorial data, version 2: staged in
``D:/pytomography_records/spect-v2``), with their sizes and sha256 in the dataset registry as pending parts until the
record is published.

``lu177_nema_symt2_rois.seg.nrrd`` (SPECT/Lu177-NEMA-SymT2: the NEMA IQ phantom on the Siemens Symbia T2), frame
'patient' (the CT's), 0.8 mm voxels:

* ``sphere_37mm`` ... ``sphere_10mm``: balls of the spheres' inside diameters. t_algorithms_measured used to find
  their centres in the notebook; this script finds them the same way. The NEMA geometry puts the centres on a
  circle of 57.2 mm radius around the lung insert's axis, in one transverse plane. The fit starts from the insert's
  axis in the CT and from the hottest voxel, and places the circle where it holds the most counts of Siemens' own
  reconstruction, smoothed by a Gaussian of 2 voxels and put on read_dicom's grid by its own header. Each sphere but
  the 10 mm one is then moved on its own, by at most 10 mm, to where it holds the most counts.
* ``lung_insert``: the CT's foam (below -400 HU) inside the phantom.
* ``background``: the phantom's water at least 15 mm from anything else (the phantom's walls, lid and base, the lung
  insert, the spheres' stems) and at least 20 mm from every sphere. The phantom is the CT above -200 HU: its largest
  piece, with its holes filled slice by slice (the bed, below -200 HU, is not part of it). Its water is the CT between
  -100 and 40 HU, smoothed by 1 mm.

``jaszak_rois.seg.nrrd`` (SPECT/SIMIND-Jaszak: the SIMIND Jaszczak phantom), frame 'scanner', centred on the rotation
axis like read_simind's grid, on SIMIND's own grid of 1.2 mm voxels:

* ``sphere_1`` ... ``sphere_6``: the six spheres, SIMIND's sources 1 (the largest) to 6, from the 3D Slicer
  segmentation that the tutorial data ship (``jaszak_spheres.npz``). Their volumes match the simulation's sources
  within 0.6% (3.4% for the 0.45 mL sphere).
* ``background``: the water at least 15 mm from the phantom's outside and from every sphere. The water is the
  simulation's attenuation map (4.8 mm voxels) above half its maximum.
"""
from __future__ import annotations

import gzip
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage, optimize

DATA = Path(os.environ.get('PYTOMOGRAPHY_DATA', Path.home() / 'pytomography_data')) / 'SPECT'
MADE_BY = 'docs/tools/make_roi_masks.py'


def write_seg_nrrd(path: Path, labels: np.ndarray, affine: np.ndarray, names: dict, frame: str, notes: dict) -> None:
    """A label image (axes x, y, z; uint8) as a gzip NRRD in LPS with 3D Slicer's segment fields and
    ``pytomography_frame``. ``affine`` maps voxel indices to LPS mm and must be axis-aligned."""
    d = np.diag(affine)[:3]
    assert np.allclose(affine[:3, :3], np.diag(d)) and np.all(d > 0)
    lines = ['NRRD0004', f'# PyTomography regions of interest, made by {MADE_BY}', 'type: unsigned char',
             'dimension: 3', 'space: left-posterior-superior', 'sizes: ' + ' '.join(str(n) for n in labels.shape),
             'space directions: ' + ' '.join('(' + ','.join(f'{d[a]:.6g}' if a == b else '0' for b in range(3)) + ')'
                                             for a in range(3)),
             'kinds: domain domain domain', 'encoding: gzip',
             'space origin: (' + ','.join(f'{v:.6f}' for v in affine[:3, 3]) + ')', f'pytomography_frame:={frame}']
    for n, (name, value) in enumerate(names.items()):
        lines += [f'Segment{n}_ID:={name}', f'Segment{n}_Name:={name}', f'Segment{n}_LabelValue:={value}',
                  f'Segment{n}_Layer:=0']
    lines += [f'{key}:={value}' for key, value in notes.items()]
    data = gzip.compress(np.ascontiguousarray(labels.astype(np.uint8).transpose(2, 1, 0)).tobytes(), 9, mtime=0)
    path.write_bytes(('\n'.join(lines) + '\n\n').encode('latin-1') + data)


def crop_to_labels(labels: np.ndarray, affine: np.ndarray, margin: int = 2):
    """The label image cut to the box around its labels (plus ``margin`` voxels), and that box's affine."""
    ijk = np.argwhere(labels > 0)
    lo = np.maximum(ijk.min(0) - margin, 0)
    hi = np.minimum(ijk.max(0) + 1 + margin, labels.shape)
    A = affine.copy()
    A[:3, 3] = affine[:3, :3] @ lo + affine[:3, 3]
    return labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], A


def largest_piece(mask: np.ndarray) -> np.ndarray:
    pieces, _ = ndimage.label(mask)
    return pieces == np.argmax(np.bincount(pieces.ravel())[1:]) + 1


def fill_slices(mask: np.ndarray) -> np.ndarray:
    return np.stack([ndimage.binary_fill_holes(mask[:, :, k]) for k in range(mask.shape[2])], axis=-1)


def summary(name: str, labels: np.ndarray, names: dict, voxel_mm: float) -> str:
    return (f'{name}: {labels.shape} voxels of {voxel_mm} mm; '
            + ', '.join(f'{k} {int((labels == v).sum()) * voxel_mm ** 3 / 1000:.2f} mL' for k, v in names.items()))


# ---------------------------------------------------------------------------- the NEMA IQ phantom, Siemens Symbia T2

DIAMETERS = [37, 28, 22, 17, 13, 10]          # mm, in order around the ring
RING_RADIUS = 57.2                             # mm


def nema_sphere_centres(io, path: Path, files_CT: list) -> tuple[np.ndarray, str]:
    """The spheres' centres in LPS mm, found as t_algorithms_measured found them (see the docstring), and the fit."""
    projections, object_meta, proj_meta = io.SPECT.read_dicom(str(path / 'projection_data.dcm'))
    mu = io.attenuation_map(files_CT, object_meta, energy=proj_meta.energy_windows[0].photon_energy).cpu().numpy()
    image = (io.load_image(path / 'scanner_recon.dcm').on(object_meta) / proj_meta.num_projections).cpu().numpy()
    dx = object_meta.dr[0]                     # positions below are in mm on the voxel grid (index x dx)
    mu_water = float(np.median(mu[mu > 0.005]))
    phantom = fill_slices(largest_piece(ndimage.binary_opening(mu > 0.8 * mu_water, structure=np.ones((5, 5, 1)))))
    lung = ndimage.binary_erosion(phantom, structure=np.ones((7, 7, 1))) & (mu < 0.6 * mu_water)
    lung_axis = np.array(ndimage.center_of_mass(lung))[:2] * dx

    def sphere_box(centre, radius, n=5):
        # the fraction of each voxel inside a sphere (n x n x n points per voxel), in the box of voxels around it
        lo = np.maximum(np.floor((centre - radius) / dx - 1).astype(int), 0)
        hi = np.minimum(np.ceil((centre + radius) / dx + 2).astype(int), object_meta.shape)
        points = (np.arange(n) + 0.5) / n - 0.5
        axes = [((np.arange(l, h)[:, None] + points) * dx).ravel() - c for l, h, c in zip(lo, hi, centre)]
        X, Y, Z = np.meshgrid(*axes, indexing='ij')
        inside = X ** 2 + Y ** 2 + Z ** 2 <= radius ** 2
        fraction = inside.reshape(hi[0] - lo[0], n, hi[1] - lo[1], n, hi[2] - lo[2], n).mean(axis=(1, 3, 5))
        return tuple(slice(l, h) for l, h in zip(lo, hi)), fraction

    def ring_centres(cx, cy, cz, angle, order):
        a = np.deg2rad(angle) + order * np.arange(6) * np.pi / 3
        return np.stack([cx + RING_RADIUS * np.cos(a), cy + RING_RADIUS * np.sin(a), np.full(6, cz)], axis=1)

    def counts_in_sphere(centre, d, img):
        box, fraction = sphere_box(np.asarray(centre), d / 2)
        return (fraction * img[box]).sum()

    smooth = ndimage.gaussian_filter(image, 2)
    hottest = np.array(np.unravel_index(np.argmax(smooth), image.shape)) * dx       # the 37 mm sphere
    start = [*lung_axis, hottest[2], np.degrees(np.arctan2(hottest[1] - lung_axis[1], hottest[0] - lung_axis[0]))]
    in_ring = lambda p, order: sum(counts_in_sphere(c, d, smooth) for c, d in zip(ring_centres(*p, order), DIAMETERS))
    fits = {order: optimize.minimize(lambda p: -in_ring(p, order), start, method='Nelder-Mead',
                                     options={'xatol': 0.05, 'fatol': 1e-3}) for order in (1, -1)}
    order = min(fits, key=lambda o: fits[o].fun)                                   # both directions round the ring
    ring = fits[order].x
    centres = ring_centres(*ring, order)
    moved = {}
    for i, d in enumerate(DIAMETERS[:-1]):     # each sphere on its own, but the 10 mm one
        on_ring = centres[i].copy()
        fit = optimize.minimize(lambda c: -counts_in_sphere(c, d, smooth), on_ring, method='Nelder-Mead',
                                bounds=[(c - 10, c + 10) for c in on_ring], options={'xatol': 0.05, 'fatol': 1e-4})
        centres[i] = fit.x
        moved[d] = float(np.linalg.norm(fit.x - on_ring))
    lps = (object_meta.affine[:3, :3] @ (centres / dx).T).T + object_meta.affine[:3, 3]
    fit = (f'ring centre {ring[0] - lung_axis[0]:+.1f} mm, {ring[1] - lung_axis[1]:+.1f} mm from the lung insert axis '
           f'in the CT; sphere plane at slice {ring[2] / dx:.2f} of the read_dicom grid; spheres moved from the ring by '
           + ', '.join(f'{d} mm {m:.1f}' for d, m in moved.items()) + ' mm')
    return lps, fit


def nema(io, out: Path, step: float = 0.8) -> Path:
    from pytomography.metadata import ObjectMeta
    path = DATA / 'Lu177-NEMA-SymT2'
    files_CT = sorted(str(f) for f in (path / 'CT').glob('*'))
    centres, fit = nema_sphere_centres(io, path, files_CT)
    print(fit)
    ct = io.load_image(files_CT)
    # a fine grid around the phantom (the CT's largest piece above -200 HU), with a margin of 6 mm
    ijk = np.argwhere(largest_piece(ct.data > -200))
    corners = (ct.affine[:3, :3] @ np.stack([ijk.min(0), ijk.max(0)]).T).T + ct.affine[:3, 3]
    lo, hi = corners.min(0) - 6, corners.max(0) + 6
    shape = tuple(int(np.ceil((h - l) / step)) + 1 for l, h in zip(lo, hi))
    grid = ObjectMeta((step,) * 3, shape, centre=tuple((lo + hi) / 2), frame='patient')
    hu = ct.on(grid, fill=-1000).cpu().numpy()
    phantom = fill_slices(largest_piece(hu > -200))
    lung = largest_piece(phantom & (hu < -400))
    labels = np.zeros(shape, np.uint8)
    names = {}
    for value, (d, c) in enumerate(zip(DIAMETERS, centres), start=1):
        # the ball, in the box of fine voxels around it
        b_lo = np.maximum(np.floor((c - d / 2 - grid.affine[:3, 3]) / step).astype(int) - 1, 0)
        b_hi = np.minimum(np.ceil((c + d / 2 - grid.affine[:3, 3]) / step).astype(int) + 2, shape)
        axes = [grid.affine[a, 3] + step * np.arange(b_lo[a], b_hi[a]) - c[a] for a in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing='ij')
        box = tuple(slice(l, h) for l, h in zip(b_lo, b_hi))
        labels[box][X ** 2 + Y ** 2 + Z ** 2 <= (d / 2) ** 2] = value
        names[f'sphere_{d}mm'] = value
    spheres = labels > 0
    labels[lung & ~spheres] = 7
    names['lung_insert'] = 7
    # the water: the CT inside the phantom between -100 and 40 HU, smoothed by 1 mm to quiet its noise. The rest is the
    # phantom's plastic (walls, lid and base), the lung insert and the spheres' stems; noise specks are left out.
    smooth = ndimage.gaussian_filter(hu, 1.0 / step)
    pieces, _ = ndimage.label(phantom & ~((smooth > -100) & (smooth < 40)))
    sizes = np.bincount(pieces.ravel())
    water = phantom & ~np.isin(pieces, [k for k in np.flatnonzero(sizes >= 50) if k > 0])
    background = (water & (ndimage.distance_transform_edt(water, sampling=step) >= 15)
                  & (ndimage.distance_transform_edt(~spheres, sampling=step) >= 20))
    labels[background] = 8
    names['background'] = 8
    labels, A = crop_to_labels(labels, grid.affine)
    notes = {'pytomography_made_by': MADE_BY, 'pytomography_sphere_fit': fit,
             'pytomography_sphere_centres_lps_mm': '; '.join(f'{d} mm ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})'
                                                             for d, c in zip(DIAMETERS, centres))}
    target = out / 'lu177_nema_symt2_rois.seg.nrrd'
    write_seg_nrrd(target, labels, A, names, 'patient', notes)
    print(summary(target.name, labels, names, step))
    return target


# ---------------------------------------------------------------------------- the SIMIND Jaszczak phantom

def jaszak(io, out: Path) -> Path:
    path = DATA / 'SIMIND-Jaszak'
    fine = np.load(path / 'jaszak_spheres.npz')['labels'].astype(np.uint8)     # 512^3 of 1.2 mm, axes x, y, z
    _, object_meta, proj_meta = io.SPECT.read_simind(str(path / 'lu177_SYME_jaszak' / 'tot_w1.h00'))
    n = fine.shape[0] // object_meta.shape[0]                                   # 4 fine voxels per grid voxel
    assert fine.shape == tuple(n * s for s in object_meta.shape)
    # fine voxel i is at grid index (i - (n - 1) / 2) / n
    S = np.diag([1 / n, 1 / n, 1 / n, 1.0])
    S[:3, 3] = -(n - 1) / 2 / n
    A = object_meta.affine @ S
    step = float(A[0, 0])
    mu = io.attenuation_map(path / 'lu177_SYME_jaszak' / 'amap.hct', object_meta,
                            energy=proj_meta.energy_windows[0].photon_energy, layout='mu').cpu().numpy()
    water = np.repeat(np.repeat(np.repeat(mu > 0.5 * mu.max(), n, 0), n, 1), n, 2)
    # the distances on the box around the water only
    box = tuple(slice(max(l - 2, 0), h + 3) for l, h in zip(np.argwhere(water).min(0), np.argwhere(water).max(0)))
    w, s = water[box], fine[box] > 0
    background = np.zeros_like(water)
    background[box] = (w & (ndimage.distance_transform_edt(w, sampling=step) >= 15)
                       & (ndimage.distance_transform_edt(~s, sampling=step) >= 15))
    labels = fine.copy()
    labels[background & (fine == 0)] = 7
    names = {f'sphere_{k}': k for k in range(1, 7)}
    names['background'] = 7
    labels, A = crop_to_labels(labels, A)
    notes = {'pytomography_made_by': MADE_BY,
             'pytomography_sources': "the spheres: jaszak_spheres.npz (3D Slicer, 2024); the water: the simulation's "
                                     'attenuation map amap.hct'}
    target = out / 'jaszak_rois.seg.nrrd'
    write_seg_nrrd(target, labels, A, names, 'scanner', notes)
    print(summary(target.name, labels, names, step))
    return target


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    from pytomography import io
    nema(io, out)
    jaszak(io, out)


if __name__ == '__main__':
    main()
