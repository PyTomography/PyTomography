# Small files of the tutorial data

`pytomography.datasets.fetch()` copies these files into the dataset folders that need them, also into the folders of
datasets that cannot be downloaded yet, once they hold the data. Each is pinned by size and sha256 in `../registry.py`;
change both together. The regions the tutorials measure (`*_rois.seg.nrrd`) are not here: they are hosted with their
data (Luke, 13 Oct 2026), and the registry pins them as parts of their datasets' records.

| File | Copied into | What it is |
|---|---|---|
| `ac225_psf_model.json` | `SPECT/SIMIND-Jaszak`, `SPECT/Ac225-NEMA-SymT2` | The Ac-225 PSF model fitted with SPECTPSFToolbox, which replaced the pickled PSF operator in the 2025 SPECT record |
| `fdg_spheres.seg.nrrd` | `PET/GE-DMI-NEMA` | 3D Slicer mask of the six spheres (labels 1 to 6, the largest first), the lung insert (7) and a background region (8) of the NEMA IQ phantom in the GE Discovery MI scan, for the PET uncertainty tutorial. Its geometry was corrected for version 4's upright GE frame: it was drawn by index on version 3's upside-down grid, with its first two axes swapped. It says it is in the scanner's frame (`pytomography_frame:=scanner`), like the GE grid (`docs/tools/fix_ge_sphere_mask.py`, 12 Oct 2026) |
| `jaszak_spheres.npz` | `SPECT/SIMIND-Jaszak` | The six spheres of the SIMIND Jaszczak phantom on SIMIND's 512³ grid of 1.2 mm voxels, from which the accuracy tutorial's regions (`jaszak_rois.seg.nrrd`, with the SPECT data) were made: `labels` (uint8, PyTomography's x, y, z order; label k is SIMIND's source k, 1 the largest) and `voxel_size_cm`. Made from a 3D Slicer segmentation (`sphere_segmentations_high_res.seg.nrrd`, 2024) whose sphere volumes match the simulation's sources within 0.6% (3.4% for the 0.45 mL sphere) |

They are PyTomography tutorial data, under CC BY 4.0, like the rest of the PyTomography tutorial records.
