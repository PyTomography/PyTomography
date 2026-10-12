# Small files of the tutorial data

`pytomography.datasets.fetch()` copies these files into the dataset folders that need them. Each is pinned by size and
sha256 in `../registry.py`; change both together.

| File | Copied into | What it is |
|---|---|---|
| `ac225_psf_model.json` | `SPECT/SIMIND-Jaszak`, `SPECT/Ac225-NEMA-SymT2` | The Ac-225 PSF model fitted with SPECTPSFToolbox, which replaced the pickled PSF operator in the 2025 SPECT record |
| `fdg_spheres.seg.nrrd` | `PET/GE-DMI-NEMA` | 3D Slicer mask of the six spheres (labels 1 to 6, the largest first), the lung insert (7) and a background region (8) of the NEMA IQ phantom in the GE Discovery MI scan, for the PET uncertainty tutorial. Its geometry was corrected for version 4's upright GE frame: it was drawn by index on version 3's upside-down grid, with its first two axes swapped. It says it is in the scanner's frame (`pytomography_frame:=scanner`), like the GE grid (`docs/tools/fix_ge_sphere_mask.py`, 12 Oct 2026) |
| `jaszak_spheres.npz` | `SPECT/SIMIND-Jaszak` | The six spheres of the SIMIND Jaszczak phantom on SIMIND's 512³ grid of 1.2 mm voxels, for the accuracy tutorial: `labels` (uint8, PyTomography's x, y, z order; label k is SIMIND's source k, 1 the largest) and `voxel_size_cm`. Made from a 3D Slicer segmentation (`sphere_segmentations_high_res.seg.nrrd`, 2024) whose sphere volumes match the simulation's sources within 0.6% (3.4% for the 0.45 mL sphere) |

Both are PyTomography tutorial data, under CC BY 4.0, like the rest of the PyTomography tutorial records.
