"""Correct the geometry of the GE NEMA sphere mask (``fdg_spheres.seg.nrrd``) for PyTomography 4's upright GE frame.

The mask of the six spheres (and two background regions) of the GE Discovery MI NEMA scan (dataset
``PET/GE-DMI-NEMA``) was drawn in 3D Slicer, by voxel index, on a PyTomography 3 reconstruction: the 192 x 192 x 71
grid of 2.78 mm voxels, which version 3 put upside down (turned 180 degrees about the scanner's axis). Slicer saved it
with LPS directions (-2.78, -2.78, 2.78) mm and the origin (0, 0, 0), so the tutorials read it by index and never used
its geometry.

From version 4, ``io.PET.read_ge_hdf5`` puts the grid upright and centred on the scanner's axis, and the tutorials put
the mask on the grid by its geometry (``io.load_mask(file).on(object_meta, label=k)``). The voxel that was at index
(i, j, k) of the version 3 grid is at index (191 - i, 191 - j, k) of the version 4 grid, whose voxel centres are at
((n - 95.5) * 2.78, (n - 95.5) * 2.78, (k - 35) * 2.78) mm. The mask's directions already flip i and j, so only its
origin changes: to (95.5 * 2.78, 95.5 * 2.78, -35 * 2.78) = (265.49, 265.49, -97.3) mm. The voxels and the rest of the
header stay as they are, and every NRRD reader (3D Slicer included) puts the spheres on the version 4 images.

The grid has no patient frame (the list mode places nothing in the patient), so the script also adds the key-value
field ``pytomography_frame:=scanner``, which ``io.load_image`` reads: the mask is then in the same frame as
``read_ge_hdf5``'s grid, and ``.on()`` puts it there without a warning. Other readers ignore the field.

Run once (11 Oct 2026) on the file as drawn (shipped in ``pytomography/datasets/files`` until then); the dataset
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


def corrected_origin() -> np.ndarray:
    """The LPS position (mm) of the mask's voxel (0, 0, 0) on version 4's upright, centred grid: version 3's voxel
    (0, 0, 0) is version 4's (191, 191, 0)."""
    shape = np.array(GRID_SHAPE, dtype=float)
    index_v4 = np.array([shape[0] - 1, shape[1] - 1, 0.0])
    return (index_v4 - (shape - 1) / 2) * GRID_DR_MM


def fix(source: Path, target: Path) -> str:
    raw = source.read_bytes()
    end = raw.find(b"\n\n")
    header, data = raw[:end].decode("latin-1"), raw[end:]
    sizes = tuple(int(v) for v in re.search(r"^sizes: (.*)$", header, re.M).group(1).split())
    if sizes != GRID_SHAPE:
        raise ValueError(f"{source.name} has sizes {sizes}, not the GE grid's {GRID_SHAPE}")
    directions = re.search(r"^space directions: (.*)$", header, re.M).group(1)
    expected = [np.diag([-GRID_DR_MM, -GRID_DR_MM, GRID_DR_MM])[i] for i in range(3)]
    found = [np.array([float(v) for v in d.split(",")]) for d in re.findall(r"\(([^)]*)\)", directions)]
    if not all(np.allclose(f, e) for f, e in zip(found, expected)):
        raise ValueError(f"{source.name}: unexpected space directions {directions}")
    origin = re.search(r"^space origin: \(([^)]*)\)$", header, re.M)
    if not np.allclose([float(v) for v in origin.group(1).split(",")], 0.0):
        raise ValueError(f"{source.name}: the origin is ({origin.group(1)}), not (0,0,0): already corrected?")
    if "pytomography_frame:=" in header:
        raise ValueError(f"{source.name} already has a pytomography_frame field: already corrected?")
    new = ",".join(repr(float(v)) for v in np.round(corrected_origin(), 6))
    # the new origin, and the frame as a key-value field on the line after it
    header = header[:origin.start(1)] + new + header[origin.end(1):origin.end()] + "\npytomography_frame:=scanner" + \
        header[origin.end():]
    target.write_bytes(header.encode("latin-1") + data)
    return new


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="the mask as drawn (origin (0,0,0))")
    parser.add_argument("target", type=Path, help="where to write the corrected mask")
    args = parser.parse_args()
    origin = fix(args.source, args.target)
    content = args.target.read_bytes()
    print(f"space origin: ({origin})")
    print(f"size {len(content)} bytes, sha256 {hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
