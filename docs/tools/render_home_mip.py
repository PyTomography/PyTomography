"""Render the home page's rotating image: the stitched Lu-177-PSMA SPECT of the multi-bed tutorial over its CT.

Each frame is a maximum intensity projection (MIP) of the SPECT, in PyTomography's 'pet' colour scale, laid over a MIP of
the patient's CT in grey, seen from one angle round the patient's long axis. The colour limits are the same in every
frame. The patient lies horizontally, head on the left, as on the scanner's bed. The site uses an animated WebP: a GIF
of the same frames is seven times larger. It runs on the CPU in under a minute and needs about 3 GB of RAM.

Its inputs are what t_dicommultibed saves and reads, found from PYTOMOGRAPHY_OUTPUT and PYTOMOGRAPHY_DATA by default:

    --recon  the stitched reconstruction the tutorial saves: t_dicommultibed/osem_stitched/NM_0001.dcm in the output
             folder of SPECT/Lu177-PSMA-GEDisc
    --ct     the CT folder of the SPECT/Lu177-PSMA-GEDisc dataset

    python docs/tools/render_home_mip.py --out docs/source/_static/landing/mip.webp
    python docs/tools/render_home_mip.py --recon RUN/outputs/.../NM_0001.dcm --ct DATA/SPECT/Lu177-PSMA-GEDisc/CT \\
        --out docs/source/_static/landing/mip.webp --frames-dir preview/    # also every 6th frame as PNG

It needs PyTomography 4 (for io.load_image), SciPy and Pillow. The colour settings are render()'s defaults.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # the CPU is enough, and the GPU is shared

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from scipy import ndimage

import pytomography
from pytomography import datasets, io
from pytomography.metadata import ObjectMeta

pytomography.device = torch.device("cpu")

DATASET = "SPECT/Lu177-PSMA-GEDisc"


def pet_colours() -> np.ndarray:
    """PyTomography's 'pet' colour scale (src/pytomography/data/pet_colors.txt) as an (N, 3) array in [0, 1]."""
    path = Path(pytomography.__file__).parent / "data" / "pet_colors.txt"
    return np.loadtxt(path).reshape(-1, 3) / 255.0


def lookup(table: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The colour of each value v in [0, 1], interpolated linearly in the table."""
    x = np.clip(v, 0, 1) * (len(table) - 1)
    i = np.minimum(x.astype(int), len(table) - 2)
    f = (x - i)[..., None]
    return table[i] * (1 - f) + table[i + 1] * f


def centre_of(affine: np.ndarray, shape) -> np.ndarray:
    """The LPS position (mm) of the centre of a grid with this affine and shape."""
    return affine[:3, :3] @ ((np.asarray(shape) - 1) / 2) + affine[:3, 3]


def body_mask(hu: np.ndarray, voxel_mm: float) -> np.ndarray:
    """The patient without the bed: tissue above -400 HU, opened by about 6 mm so that the bed's thin shell and its
    pad come away from the patient, the largest connected part, grown back out to the skin, holes filled."""
    tissue = hu > -400
    cross = ndimage.generate_binary_structure(3, 1)
    r = max(1, int(round(6 / voxel_mm)))
    labels, _ = ndimage.label(ndimage.binary_opening(tissue, structure=cross, iterations=r))
    body = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1
    body = ndimage.binary_dilation(body, structure=cross, iterations=r + 1) & tissue
    for k in range(body.shape[2]):
        body[:, :, k] = ndimage.binary_fill_holes(body[:, :, k])
    return body


def rotated_mips(volume: torch.Tensor, angles: np.ndarray, background: float) -> np.ndarray:
    """MIPs along y of the volume (x, y, z; square in x and y) turned by each angle about the grid's z axis:
    an array (angle, z, x)."""
    slices = (volume - background).permute(2, 1, 0).unsqueeze(1).contiguous()  # (z, 1, y, x), zero outside
    out = []
    for a in angles:
        c, s = np.cos(a), np.sin(a)
        theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=torch.float32).expand(slices.shape[0], 2, 3)
        grid = F.affine_grid(theta, slices.shape, align_corners=False)
        turned = F.grid_sample(slices, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        out.append(turned[:, 0].amax(dim=1).numpy() + background)  # max over y: (z, x)
    return np.stack(out)


def projections(recon: Path, ct_folder: Path, n_frames: int, voxel_mm: float, spect_fwhm_mm: float):
    """The CT's and the SPECT's MIPs, (angle, z, x) each, on a grid of voxel_mm round the patient."""
    t0 = time.time()
    spect = io.load_image(recon)
    ct = io.load_image(sorted(str(f) for f in ct_folder.iterdir() if f.is_file()))
    print(f"SPECT {spect.data.shape} ({spect.units}), CT {ct.data.shape} ({ct.units}), read in {time.time() - t0:.0f} s")

    # 1. The CT on a grid of voxel_mm over the CT's own extent, and the patient in it
    d = voxel_mm
    extent = np.abs(np.diag(ct.affine)[:3]) * np.asarray(ct.data.shape)
    shape = np.ceil(extent / d).astype(int)
    grid = ObjectMeta(dr=(d, d, d), shape=tuple(shape), centre=tuple(centre_of(ct.affine, ct.data.shape)), frame="patient")
    body = body_mask(ct.on(grid, fill=-1000.0).numpy(), d)

    # 2. A box round the patient: square in x and y and centred on the patient's axis, so that every angle fits; along
    #    z, where both the CT and the SPECT have the patient
    i, j, _ = np.nonzero(body)
    ci, cj = (i.min() + i.max()) / 2, (j.min() + j.max()) / 2
    radius = np.sqrt((i - ci) ** 2 + (j - cj) ** 2).max() + 3
    half = spect.data.shape[2] * spect.affine[2, 2] / 2
    spect_z = centre_of(spect.affine, spect.data.shape)[2] + np.array([-half, half])
    z_mm = grid.affine[2, 2] * np.arange(shape[2]) + grid.affine[2, 3]
    ks = np.nonzero(body.any(axis=(0, 1)) & (z_mm > spect_z[0]) & (z_mm < spect_z[1]))[0]
    n = int(np.ceil(2 * radius))
    centre = grid.affine @ np.array([ci, cj, (ks.min() + ks.max()) / 2, 1.0])
    box = ObjectMeta(dr=(d, d, d), shape=(n, n, int(ks.max() - ks.min() + 1)), centre=tuple(centre[:3]), frame="patient")
    hu = ct.on(box, fill=-3024.0)
    # the CT's field of view, less 10 mm: the arms reach its edge, where the CT's values are not to be trusted
    in_plane = np.zeros((3, 3, 3), bool)
    in_plane[:, :, 1] = ndimage.generate_binary_structure(2, 1)
    fov = ndimage.binary_erosion(hu.numpy() > -1500, structure=in_plane, iterations=int(round(10 / d)),
                                 border_value=1)
    body = torch.from_numpy(body_mask(hu.numpy(), d) & fov)
    hu = torch.where(body, hu, torch.tensor(-1000.0))
    activity = spect.on(box).numpy().clip(min=0)
    activity = torch.from_numpy(ndimage.gaussian_filter(activity, spect_fwhm_mm / 2.355 / d)) * body
    print(f"grid {tuple(box.shape)} of {d} mm, {time.time() - t0:.0f} s")

    # 3. The MIPs at every angle
    angles = 2 * np.pi * np.arange(n_frames) / n_frames
    return rotated_mips(hu, angles, -1000.0), rotated_mips(activity, angles, 0.0)


def render(ct_mips, spect_mips, *, spect_max=0.75, gamma=0.7, opacity=(0.16, 0.5), ct_hu=(-300, 2000), ct_white=0.68,
           fade=0.06, width=1000) -> list[PILImage.Image]:
    """The frames: the CT in grey, and over it the SPECT in the 'pet' colour scale.

    The SPECT's scale runs from 0 to spect_max times its brightest MIP pixel over every frame, raised to gamma (below
    1 it lifts the low end, so that the salivary glands show without the kidneys saturating). It is clear below
    opacity[0] of its scale and opaque above opacity[1]. The CT's MIP runs from black at ct_hu[0] to ct_white at
    ct_hu[1]. Both ends of the patient fade out over a fraction fade of the length, since the CT stops part way
    through the head and the thighs."""
    lo, hi = ct_hu
    grey = np.clip((ct_mips - lo) / (hi - lo), 0, 1) * ct_white
    s = np.clip(spect_mips / (spect_max * spect_mips.max()), 0, 1) ** gamma
    alpha = np.clip((s - opacity[0]) / (opacity[1] - opacity[0]), 0, 1)[..., None]
    rgb = (1 - alpha) * grey[..., None] + alpha * lookup(pet_colours(), s)
    n = rgb.shape[1]
    rgb *= np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / max(1, round(fade * n)), 0, 1)[None, :, None, None]
    # (angle, x, z): the patient's left at the top, as seen from the front, and the head on the left
    rgb = rgb[:, ::-1, ::-1, :].transpose(0, 2, 1, 3)
    h, w = rgb.shape[1:3]
    size = (width, int(round(width * h / w)))
    return [PILImage.fromarray((255 * np.clip(f, 0, 1)).astype(np.uint8)).resize(size, PILImage.LANCZOS) for f in rgb]


def write(frames: list[PILImage.Image], out: Path, ms: int, quality: int = 80) -> None:
    """An animated WebP (lossy, a fifth of the GIF's size for the same look), or a GIF with one palette for every
    frame, so that colours don't flicker, and no dithering, so that the black stays black."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".webp":
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=ms, loop=0, quality=quality, method=6)
        return
    sample = PILImage.fromarray(np.concatenate([np.asarray(f) for f in frames[::5]], axis=0))
    palette = sample.quantize(colors=255, method=PILImage.Quantize.MEDIANCUT, dither=PILImage.Dither.NONE)
    indexed = [f.quantize(palette=palette, dither=PILImage.Dither.NONE) for f in frames]
    indexed[0].save(out, save_all=True, append_images=indexed[1:], duration=ms, loop=0, optimize=True, disposal=1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    outputs = Path(os.environ.get("PYTOMOGRAPHY_OUTPUT") or "pytomography_outputs")  # as datasets.output_dir
    p.add_argument("--recon", type=Path, default=outputs / DATASET / "t_dicommultibed" / "osem_stitched" / "NM_0001.dcm")
    p.add_argument("--ct", type=Path, default=datasets.data_dir() / DATASET / "CT")
    p.add_argument("--out", type=Path, required=True, help="the image to write: .webp (the site's) or .gif")
    p.add_argument("--frames", type=int, default=72, help="views round the patient (default 72, every 5 degrees)")
    p.add_argument("--ms", type=int, default=70, help="milliseconds per frame")
    p.add_argument("--width", type=int, default=1000, help="width of the image in pixels")
    p.add_argument("--quality", type=int, default=80, help="WebP quality, 0 to 100")
    p.add_argument("--voxel-mm", type=float, default=2.0, help="the grid the projections are made on")
    p.add_argument("--spect-fwhm-mm", type=float, default=8.0, help="Gaussian filter on the SPECT before projecting")
    p.add_argument("--frames-dir", type=Path, help="also save every 6th frame here as PNG, to look at")
    args = p.parse_args()
    t0 = time.time()
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) - 4)))  # leave cores for other work

    ct_mips, spect_mips = projections(args.recon, args.ct, args.frames, args.voxel_mm, args.spect_fwhm_mm)
    frames = render(ct_mips, spect_mips, width=args.width)
    write(frames, args.out, args.ms, args.quality)
    if args.frames_dir:
        args.frames_dir.mkdir(parents=True, exist_ok=True)
        for k in range(0, len(frames), 6):
            frames[k].save(args.frames_dir / f"frame_{k:03d}.png")
    print(f"{args.out}: {len(frames)} frames of {frames[0].size[0]} x {frames[0].size[1]}, "
          f"{args.out.stat().st_size / 1e6:.2f} MB, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
