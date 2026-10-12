"""Correct the geometry of the GE NEMA sphere mask (``fdg_spheres.seg.nrrd``) for PyTomography 4's upright GE frame.

The mask of the six spheres, the lung insert and a background region of the GE Discovery MI NEMA scan (dataset
``PET/GE-DMI-NEMA``) was drawn in 3D Slicer, by voxel index, on a PyTomography 3 reconstruction: the 192 x 192 x 71
grid of 2.78 mm voxels, which version 3 put upside down (turned 180 degrees about the scanner's axis). Slicer saved it
with LPS directions (-2.78, -2.78, 2.78) mm and the origin (0, 0, 0), so the tutorials read it by index and never used
its geometry. It was also drawn on the image with its first two axes swapped: the mask's voxel (a, b, k) is version 3's
voxel (b, a, k). Read by index, its "spheres" lay on the background (their mean was the background's, 0.04 against
about 0.33 in the spheres), so the uncertainty that the PET uncertainty tutorial printed before version 4 was a
background region's, not the largest sphere's.

From version 4, ``io.PET.read_ge_hdf5`` puts the grid upright and centred on the scanner's axis, and the tutorials put
the mask on the grid by its geometry (``io.load_mask(file).on(object_meta, label=k)``). Version 3's voxel (i, j, k) is
version 4's (191 - i, 191 - j, k), whose centre is at ((191 - i - 95.5) * 2.78, (191 - j - 95.5) * 2.78,
(k - 35) * 2.78) mm, so the mask's voxel (a, b, k) belongs at ((95.5 - b) * 2.78, (95.5 - a) * 2.78, (k - 35) * 2.78):
space directions (0, -2.78, 0), (-2.78, 0, 0), (0, 0, 2.78) and the origin (95.5 * 2.78, 95.5 * 2.78, -35 * 2.78) =
(265.49, 265.49, -97.3) mm. The voxels stay as they are, and every NRRD reader (3D Slicer included) puts the six
spheres on the hot spheres of the version 4 images, the lung insert on the cold insert.

The grid has no patient frame (the list mode places nothing in the patient), so the script also adds the key-value
field ``pytomography_frame:=scanner``, which ``io.load_image`` reads: the mask is then in the same frame as
``read_ge_hdf5``'s grid, and ``.on()`` puts it there without a warning. Other readers ignore the field.

Run once (12 Oct 2026) on the file as drawn (shipped in ``pytomography/datasets/files`` until 11 Oct); the dataset
registry pins the result::

    python docs/tools/fix_ge_sphere_mask.py path/to/fdg_spheres.seg.nrrd corrected.seg.nrrd
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np

#: the GE reconstruction grid the mask was drawn on (version 3) and that version 4's tutorials use
GRID_SHAPE = (192, 192, 71)
GRID_DR_MM = 2.78


def corrected_geometry() -> tuple[np.ndarray, np.ndarray]:
    """The mask's space directions (one column per array axis) and origin (LPS mm) on version 4's upright, centred
    grid: its voxel (a, b, k) is version 3's (b, a, k), which is version 4's (191 - b, 191 - a, k)."""
    n = np.array(GRID_SHAPE, dtype=float)
    d = GRID_DR_MM
    directions = np.array([[0.0, -d, 0.0], [-d, 0.0, 0.0], [0.0, 0.0, d]]).T   # columns: the axes a, b, k
    origin = np.array([(n[1] - 1) / 2 * d, (n[0] - 1) / 2 * d, -(n[2] - 1) / 2 * d])
    return directions, origin


def _vector(v) -> str:
    return "(" + ",".join(repr(float(x)) for x in np.round(v, 6)) + ")"


def fix(source: Path, target: Path) -> tuple[str, str]:
    raw = source.read_bytes()
    end = raw.find(b"\n\n")
    header, data = raw[:end].decode("latin-1"), raw[end:]
    sizes = tuple(int(v) for v in re.search(r"^sizes: (.*)$", header, re.M).group(1).split())
    if sizes != GRID_SHAPE:
        raise ValueError(f"{source.name} has sizes {sizes}, not the GE grid's {GRID_SHAPE}")
    found_directions = re.search(r"^space directions: (.*)$", header, re.M)
    expected = np.diag([-GRID_DR_MM, -GRID_DR_MM, GRID_DR_MM])
    found = [np.array([float(v) for v in d.split(",")]) for d in re.findall(r"\(([^)]*)\)", found_directions.group(1))]
    if not all(np.allclose(f, e) for f, e in zip(found, expected.T)):
        raise ValueError(f"{source.name}: unexpected space directions {found_directions.group(1)}: already corrected?")
    origin = re.search(r"^space origin: \(([^)]*)\)$", header, re.M)
    if not np.allclose([float(v) for v in origin.group(1).split(",")], 0.0):
        raise ValueError(f"{source.name}: the origin is ({origin.group(1)}), not (0,0,0): already corrected?")
    if "pytomography_frame:=" in header:
        raise ValueError(f"{source.name} already has a pytomography_frame field: already corrected?")
    directions, new_origin = corrected_geometry()
    new_directions = " ".join(_vector(directions[:, axis]) for axis in range(3))
    # the new directions and origin, and the frame as a key-value field on the line after the origin
    header = (header[:found_directions.start(1)] + new_directions + header[found_directions.end(1):origin.start()]
              + f"space origin: {_vector(new_origin)}\npytomography_frame:=scanner" + header[origin.end():])
    target.write_bytes(header.encode("latin-1") + data)
    return new_directions, _vector(new_origin)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="the mask as drawn (origin (0,0,0))")
    parser.add_argument("target", type=Path, help="where to write the corrected mask")
    args = parser.parse_args()
    directions, origin = fix(args.source, args.target)
    content = args.target.read_bytes()
    print(f"space directions: {directions}")
    print(f"space origin: {origin}")
    print(f"size {len(content)} bytes, sha256 {hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
