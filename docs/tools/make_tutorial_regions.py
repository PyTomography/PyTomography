"""The regions of interest that the PET and CT tutorials analyse, found once from the tutorial data and saved as 3D Slicer
segmentations (``.seg.nrrd``) that come with PyTomography (``src/pytomography/datasets/files``; ``datasets.fetch`` copies
each into its dataset's folder). A tutorial then takes a region in one line,
``io.load_mask(file).on(object_meta, label=name)``, instead of finding it in code.

Each file is in LPS mm, says its frame (``pytomography_frame:=``) and how it was made, and names its segments as the
SPECT tutorials' region files do (``docs/tools/make_roi_masks.py``: ``sphere_37mm``, ``background``). Its voxels are those of the
grid its tutorial reconstructs on, where the regions were found (cropped to the regions), so the tutorial gets back
exactly these voxels, and ``.on()`` puts the regions on any other grid by the fraction of each voxel inside them.
Segments that overlap (the spheres and the background inside the phantom) are saved in layers, as 3D Slicer does.

``gate_braininwater_rois.seg.nrrd`` (``PET/GATE-mMR-BrainInWater``, for t_PETGATE_DIP)
    ``grey_matter`` and ``white_matter`` of the brain phantom: its MR's intensity classes (180 to 195 and 135 to 150)
    on the tutorial's grid of 1.25 mm voxels, one voxel in from their edges.
``petsird_mmr_nema_rois.seg.nrrd`` (``PET/PETSIRD-mMR-NEMA``, for t_PETSIRD)
    ``phantom``: the phantom in the attenuation map (above 0.005 / mm, the largest connected region).
    ``sphere_37mm`` to ``sphere_10mm``: the six spheres, by their inner diameters, at the centres fitted to the
    tutorial's reconstruction without scatter correction: the NEMA arrangement (centres 60 degrees apart on a circle of
    about 57 mm radius, in one plane, the diameters in order round it), fitted in position, angle, radius and height. The header records the fit.
    ``background``: the phantom at least 30 mm from the plane of the spheres, 4 voxels in from its walls and 5 slices
    from its ends (where the image's edges are blurred). ``ring_0``, ``ring_1``, ...: the background in rings 10 mm
    wide around the phantom's axis (through the centre of the background's cross-section), ring n from 10 n to
    10 (n + 1) mm; the rings of at least 500 voxels.
``ldct_c145_rois.seg.nrrd`` (``CT/ldct-c145``, for t_CT_GEN3 and t_fbp_ct)
    ``body``: the scanner's image above -700 HU after a 2 mm Gaussian blur, inside its field of view. ``soft_tissue``:
    10 to 70 HU after a 7 x 7 x 3 voxel mean, 3 voxels (1 slice) in from its edges. ``liver``: a block of 26 x 26 x 16
    mm in the liver, where the tissue is uniform.

The PETSIRD spheres are fitted to the reconstruction that t_PETSIRD saves (its ``osem_without_scatter_correction``
DICOM series), so run that tutorial first. Usage::

    python docs/tools/make_tutorial_regions.py [--only gate petsird ct] [--out FOLDER] [--petsird-recon FOLDER]

It writes into ``src/pytomography/datasets/files`` by default and prints each file's size and sha256, which go into
``src/pytomography/datasets/registry.py``. Made on 13 Oct 2026 from the data the registry pins.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

import pytomography
from pytomography import datasets, io
from pytomography.metadata import ObjectMeta

REPO = Path(__file__).resolve().parents[2]
MADE_BY = 'docs/tools/make_tutorial_regions.py'
COLOURS = [(0.95, 0.85, 0.55), (0.85, 0.35, 0.3), (0.3, 0.55, 0.85), (0.4, 0.75, 0.45), (0.8, 0.5, 0.8),
           (0.95, 0.6, 0.2), (0.5, 0.8, 0.85), (0.7, 0.7, 0.3), (0.6, 0.45, 0.35), (0.55, 0.55, 0.55)]


# ---------- writing a 3D Slicer segmentation ----------

def _layers(segments):
    """Each segment's layer: the first in which it overlaps no other segment (3D Slicer's layers)."""
    layers, out = [], []
    for _, mask in segments:
        for n, used in enumerate(layers):
            if not (used & mask).any():
                used |= mask
                out.append(n)
                break
        else:
            layers.append(mask.copy())
            out.append(len(layers) - 1)
    return out, len(layers)


def write_segmentation(path: Path, segments, affine: np.ndarray, frame: str, notes: dict) -> None:
    """Boolean masks ``(name, mask)`` on the grid of ``affine`` (voxel index to LPS mm) as a 3D Slicer segmentation,
    cropped to the voxels they cover; overlapping segments go into layers (a 4D file, the layer first). ``notes`` become
    key-value fields of the header (``pytomography_...:=``)."""
    union = np.any([m for _, m in segments], axis=0)
    lo = np.array([np.nonzero(union.any(axis=tuple(a for a in range(3) if a != t)))[0].min() for t in range(3)])
    hi = np.array([np.nonzero(union.any(axis=tuple(a for a in range(3) if a != t)))[0].max() + 1 for t in range(3)])
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    segments = [(name, np.ascontiguousarray(mask[box])) for name, mask in segments]
    origin = affine[:3, :3] @ lo + affine[:3, 3]
    layer_of, n_layers = _layers(segments)
    shape = segments[0][1].shape
    data = np.zeros((n_layers, *shape), np.uint8)
    values = [0] * n_layers
    fields = []
    for n, ((name, mask), layer) in enumerate(zip(segments, layer_of)):
        values[layer] += 1
        data[layer][mask] = values[layer]
        idx = np.nonzero(mask)
        extent = ' '.join(f'{int(a.min())} {int(a.max())}' for a in idx) if mask.any() else '0 -1 0 -1 0 -1'
        r, g, b = COLOURS[n % len(COLOURS)]
        fields += [f'Segment{n}_Color:={r:g} {g:g} {b:g}', f'Segment{n}_ColorAutoGenerated:=0',
                   f'Segment{n}_Extent:={extent}', f'Segment{n}_ID:={name}',
                   f'Segment{n}_LabelValue:={values[layer]}', f'Segment{n}_Layer:={layer}', f'Segment{n}_Name:={name}',
                   f'Segment{n}_NameAutoGenerated:=0', f'Segment{n}_Tags:=|']
    vec = lambda v: '(' + ','.join(f'{c:.10g}' for c in v) + ')'
    directions = ' '.join(vec(affine[:3, a]) for a in range(3))
    if n_layers == 1:
        raw = np.ascontiguousarray(data[0].transpose(2, 1, 0)).tobytes()
        geometry = ['dimension: 3', f'sizes: {shape[0]} {shape[1]} {shape[2]}', f'space directions: {directions}',
                    'kinds: domain domain domain']
    else:
        raw = np.ascontiguousarray(data.transpose(3, 2, 1, 0)).tobytes()
        geometry = ['dimension: 4', f'sizes: {n_layers} {shape[0]} {shape[1]} {shape[2]}',
                    f'space directions: none {directions}', 'kinds: list domain domain domain']
    header = ['NRRD0004', f'# PyTomography regions of interest, made by {MADE_BY}', 'type: unsigned char', *geometry,
              'space: left-posterior-superior', 'encoding: gzip', f'space origin: {vec(origin)}', *fields,
              'Segmentation_ContainedRepresentationNames:=Binary labelmap|',
              'Segmentation_MasterRepresentation:=Binary labelmap',
              'Segmentation_ReferenceImageExtentOffset:=0 0 0',
              f'pytomography_frame:={frame}', f'pytomography_made_by:={MADE_BY}',
              *[f'{key}:={value}' for key, value in notes.items()]]
    path.write_bytes(('\n'.join(header) + '\n\n').encode('latin-1') + gzip.compress(raw, compresslevel=9, mtime=0))


def check(path: Path, segments, object_meta: ObjectMeta) -> None:
    """The file read back with io.load_mask gives every segment's voxels exactly on the tutorial's grid."""
    mask = io.load_mask(path)
    for name, truth in segments:
        back = mask.on(object_meta, label=name, threshold=0.5).cpu().numpy()
        assert np.array_equal(back, truth), f'{path.name}: {name} differs when read back'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f'{path.name}: {path.stat().st_size} bytes, sha256 {digest}')
    for name, truth in segments:
        print(f'    {name}: {int(truth.sum())} voxels')


# ---------- PET/GATE-mMR-BrainInWater: grey and white matter ----------

def gate(out: Path) -> None:
    folder = datasets.data_dir() / 'PET' / 'GATE-mMR-BrainInWater'
    # the tutorial's grid: read_gate places it in the patient (the registry's record of the phantom in GATE's world);
    # the events of one run are enough for that
    _, object_meta, _ = io.PET.read_gate(folder / 'simple_phantom' / 'mMR_voxBrain_withSimplePhantom_1.root',
                                         folder / 'mMR_Geometry.mac', events='delayeds', doi_mm=9)
    object_meta = object_meta.with_grid(dr=(1.25, 1.25, 1.25), shape=(204, 204, 154))
    mr = io.load_image(folder / 'fdg_pet_phantom_mri.nii.gz').on(object_meta).cpu().numpy()
    cube = np.ones((3, 3, 3))
    segments = [('grey_matter', ndimage.binary_erosion((mr >= 180) & (mr < 195), cube)),
                ('white_matter', ndimage.binary_erosion((mr >= 135) & (mr < 150), cube))]
    path = out / 'gate_braininwater_rois.seg.nrrd'
    write_segmentation(path, segments, object_meta.affine, object_meta.frame,
                       {'pytomography_sources': "the brain phantom's MR, fdg_pet_phantom_mri.nii.gz (Belzunce's record), "
                                                "averaged over the Deep Image Prior tutorial's 1.25 mm voxels"})
    check(path, segments, object_meta)


# ---------- PET/PETSIRD-mMR-NEMA: the NEMA IQ phantom ----------

DIAMETERS = (37, 28, 22, 17, 13, 10)      # mm, inner: the spheres in order round the ring
RING_RADIUS = 57.2                         # mm, the circle of the spheres' centres (NEMA NU 2)


def _fit_spheres(smooth: np.ndarray, dr: np.ndarray, hot: np.ndarray):
    """The six spheres' centres (voxel indices) fitted to an image: the NEMA arrangement, started from every way the
    hottest voxel ``hot`` can be in one of the spheres, and refined in centre, angle, radius and height."""
    g = np.linspace(-0.5, 0.5, 9)
    ball = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    ball = ball[np.linalg.norm(ball, axis=1) <= 0.5]
    coefficients = ndimage.spline_filter(smooth, order=3)   # cubic interpolation: linear would pull centres to voxels

    def centres(p, direction):
        cx, cy, cz, angle, radius = p
        return [np.array([cx + radius * np.cos(angle + direction * n * np.pi / 3) / dr[0],
                          cy + radius * np.sin(angle + direction * n * np.pi / 3) / dr[1], cz]) for n in range(6)]

    def score(p, direction):  # the mean over the inner 80% of each sphere, weighted by its diameter
        return -sum(d * ndimage.map_coordinates(coefficients, (c[None] + ball * 0.8 * d / dr).T, order=3,
                                                prefilter=False).mean()
                    for c, d in zip(centres(p, direction), DIAMETERS))

    starts = []
    for direction in (1, -1):
        for n in range(6):              # the hottest voxel in sphere n
            for angle_hot in np.radians(np.arange(0, 360, 10)):
                cx = hot[0] - RING_RADIUS * np.cos(angle_hot) / dr[0]
                cy = hot[1] - RING_RADIUS * np.sin(angle_hot) / dr[1]
                p = (cx, cy, hot[2], angle_hot - direction * n * np.pi / 3, RING_RADIUS)
                starts.append((score(p, direction), p, direction))
    starts.sort(key=lambda s: s[0])
    best = None
    for _, p, direction in starts[:6]:
        fit = minimize(score, p, args=(direction,), method='Nelder-Mead',
                       options={'xatol': 1e-4, 'fatol': 1e-12, 'maxiter': 8000})
        if best is None or fit.fun < best[0]:
            best = (fit.fun, fit.x, direction)
    _, p, direction = best
    return centres(p, direction), p


def petsird(out: Path, recon_folder: Path | None) -> None:
    folder = datasets.data_dir() / 'PET' / 'PETSIRD-mMR-NEMA'
    # the tutorial's grid (read_petsird centres it on the rings, frame 'scanner'), its attenuation map and its
    # reconstruction without scatter correction
    object_meta = ObjectMeta(dr=(2.08626, 2.08626, 2.03125), shape=(200, 200, 127), frame='scanner', modality='PT',
                             units='counts')
    dr = np.array(object_meta.dr)
    mu = io.attenuation_map(folder / '20170809_NEMA_MUMAP_UCL.v.hdr', object_meta).cpu().numpy()
    if recon_folder is None:
        recon_folder = datasets.output_dir('PET/PETSIRD-mMR-NEMA') / 't_PETSIRD' / 'osem_without_scatter_correction'
    recon = io.load_image(recon_folder, frame='scanner').on(object_meta).cpu().numpy()

    regions, _ = ndimage.label(mu > 0.005)       # 1/mm: the phantom's water (0.0093/mm) and walls, not the air
    phantom = regions == np.argmax(np.bincount(regions.ravel())[1:]) + 1

    # the spheres, started from the hottest voxel inside the phantom (4 voxels in from its edges), smoothed
    smooth = ndimage.gaussian_filter(recon, 1.0)
    inner = ndimage.binary_erosion(phantom, structure=np.ones((9, 9, 1)))
    hot = np.array(np.unravel_index(np.argmax(np.where(inner, smooth, 0)), smooth.shape), dtype=float)
    centres, p = _fit_spheres(smooth, dr, hot)
    print(f'spheres: circle of radius {p[4]:.1f} mm, plane at slice {p[2]:.2f}; centres (voxels): '
          + ', '.join(f'{d} mm {np.round(c, 1).tolist()}' for d, c in zip(DIAMETERS, centres)))
    # each sphere: the voxels at least half inside it (5 x 5 x 5 points in each voxel)
    offsets = (np.arange(5) + 0.5) / 5 - 0.5
    sub = np.stack(np.meshgrid(offsets, offsets, offsets, indexing='ij'), -1).reshape(-1, 3)
    spheres = []
    for d, c in zip(DIAMETERS, centres):          # the largest first
        box = [np.arange(max(int(c[t] - d / dr[t]), 0), min(int(c[t] + d / dr[t]) + 2, object_meta.shape[t]))
               for t in range(3)]
        ijk = np.stack(np.meshgrid(*box, indexing='ij'), -1)
        points = (ijk[..., None, :] + sub - c) * dr
        coverage = (np.linalg.norm(points, axis=-1) <= d / 2).mean(axis=-1)
        mask = np.zeros(object_meta.shape, bool)
        mask[np.ix_(*box)] = coverage >= 0.5
        spheres.append((f'sphere_{d}mm', mask))

    # the background: at least 30 mm from the plane of the spheres, 4 voxels in from the walls, 5 slices from the ends
    far = np.abs(np.arange(object_meta.shape[2]) - p[2]) * dr[2] >= 30
    background = ndimage.binary_erosion(phantom, structure=np.ones((9, 9, 11))) & far[None, None, :]
    # in rings 10 mm wide around the phantom's axis, through the centre of the background's cross-section
    cx, cy = np.argwhere(background)[:, :2].mean(axis=0)
    r = np.hypot((np.arange(object_meta.shape[0])[:, None] - cx) * dr[0],
                 (np.arange(object_meta.shape[1])[None, :] - cy) * dr[1])[:, :, None]
    rings = []
    for n in range(int(np.broadcast_to(r, background.shape)[background].max() // 10) + 1):
        ring = background & (r >= 10 * n) & (r < 10 * (n + 1))
        if ring.sum() >= 500:
            rings.append((f'ring_{n}', ring))
    segments = [('phantom', phantom), *spheres, ('background', background), *rings]
    path = out / 'petsird_mmr_nema_rois.seg.nrrd'
    mm = lambda c: object_meta.affine[:3, :3] @ c + object_meta.affine[:3, 3]      # voxel index to mm
    ring_centre = mm(np.array(p[:3]))
    notes = {'pytomography_sources': "the phantom: the scan's attenuation map 20170809_NEMA_MUMAP_UCL.v; the spheres: "
                                     "t_PETSIRD's reconstruction without scatter correction, smoothed by 1 voxel",
             'pytomography_sphere_fit': f'the centres on a circle of radius {p[4]:.2f} mm centred at '
                                        f'({ring_centre[0]:.2f}, {ring_centre[1]:.2f}) mm, in the plane '
                                        f'z = {ring_centre[2]:.2f} mm',
             'pytomography_sphere_centres_mm': '; '.join(f'{d} mm ({x:.2f}, {y:.2f}, {z:.2f})'
                                                         for d, (x, y, z) in zip(DIAMETERS, map(mm, centres)))}
    write_segmentation(path, segments, object_meta.affine, object_meta.frame, notes)
    check(path, segments, object_meta)


# ---------- CT/ldct-c145: body, soft tissue, liver ----------

def ct(out: Path) -> None:
    folder = datasets.data_dir() / 'CT' / 'ldct-c145'
    # the tutorials' grid: the scanner's pixel size, placed in the patient by the reader
    proj, object_meta, _ = io.CT.read_dicom_ct_pd(folder / 'full_dose_projections',
                                                  object_meta=ObjectMeta(dr=(0.662109, 0.662109, 1.0),
                                                                         shape=(512, 512, 316)))
    del proj
    scanner = io.load_image(sorted((folder / 'full_dose_images').glob('*.dcm')))
    hu = scanner.on(object_meta, fill=-3000).cpu().numpy()          # -3000: no scanner image here
    dx, _, dz = object_meta.dr
    valid = hu > -1500           # inside the scanner's field of view (it pads outside with a lower value)
    blurred = ndimage.gaussian_filter(np.where(valid, hu, -1000.0), (2 / dx, 2 / dx, 2 / dz))   # 2 mm
    body = valid & (blurred > -700)
    smooth = ndimage.uniform_filter(np.where(valid, hu, -1000.0), (7, 7, 3))
    soft_tissue = ndimage.binary_erosion((smooth > 10) & (smooth < 70) & valid, np.ones((7, 7, 3)))
    liver = np.zeros(object_meta.shape, bool)
    liver[100:140, 200:240, 20:36] = True                           # 26 x 26 x 16 mm, k running from the feet up
    segments = [('body', body), ('soft_tissue', soft_tissue), ('liver', liver)]
    path = out / 'ldct_c145_rois.seg.nrrd'
    write_segmentation(path, segments, object_meta.affine, object_meta.frame,
                       {'pytomography_sources': "the scanner's images (full_dose_images) on the CT tutorials' grid"})
    check(path, segments, object_meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--only', nargs='+', choices=['gate', 'petsird', 'ct'], default=['gate', 'petsird', 'ct'])
    parser.add_argument('--out', type=Path, default=REPO / 'src' / 'pytomography' / 'datasets' / 'files')
    parser.add_argument('--petsird-recon', type=Path, default=None,
                        help="t_PETSIRD's reconstruction without scatter correction (its DICOM folder); default: "
                             "in the tutorial's output folder")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter('ignore', UserWarning)
    print(f'PyTomography {pytomography.__version__} from {Path(pytomography.__file__).parent}')
    if 'gate' in args.only:
        gate(args.out)
    if 'petsird' in args.only:
        petsird(args.out, args.petsird_recon)
    if 'ct' in args.only:
        ct(args.out)


if __name__ == '__main__':
    main()
