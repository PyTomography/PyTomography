# Small files of the tutorial data

`pytomography.datasets.fetch()` copies these files into the dataset folders that need them. Each is pinned by size and
sha256 in `../registry.py`; change both together.

| File | Copied into | What it is |
|---|---|---|
| `ac225_psf_model.json` | `SPECT/SIMIND-Jaszak`, `SPECT/Ac225-NEMA-SymT2` | The Ac-225 PSF model fitted with SPECTPSFToolbox, which replaced the pickled PSF operator in the 2025 SPECT record |
| `fdg_spheres.seg.nrrd` | `PET/GE-DMI-NEMA` | 3D Slicer mask of the six spheres (labels 1 to 6, the largest first), the lung insert (7) and a background region (8) of the NEMA IQ phantom in the GE Discovery MI scan, for the PET uncertainty tutorial. Its geometry was corrected for version 4's upright GE frame: it was drawn by index on version 3's upside-down grid, with its first two axes swapped. It says it is in the scanner's frame (`pytomography_frame:=scanner`), like the GE grid (`docs/tools/fix_ge_sphere_mask.py`, 12 Oct 2026) |
| `jaszak_rois.seg.nrrd` | `SPECT/SIMIND-Jaszak` | The regions the accuracy tutorial measures in the SIMIND Jaszczak phantom, on SIMIND's grid of 1.2 mm voxels (`pytomography_frame:=scanner`, centred on the rotation axis like `read_simind`'s grid): `sphere_1` to `sphere_6` (SIMIND's sources, 1 the largest, from `jaszak_spheres.npz`) and `background` (the water at least 15 mm from the phantom's outside and from every sphere). Made by `docs/tools/make_roi_masks.py` |
| `jaszak_spheres.npz` | `SPECT/SIMIND-Jaszak` | The six spheres of the SIMIND Jaszczak phantom on SIMIND's 512³ grid of 1.2 mm voxels, from which `jaszak_rois.seg.nrrd` was made: `labels` (uint8, PyTomography's x, y, z order; label k is SIMIND's source k, 1 the largest) and `voxel_size_cm`. Made from a 3D Slicer segmentation (`sphere_segmentations_high_res.seg.nrrd`, 2024) whose sphere volumes match the simulation's sources within 0.6% (3.4% for the 0.45 mL sphere) |
| `lu177_nema_symt2_rois.seg.nrrd` | `SPECT/Lu177-NEMA-SymT2` | The regions the tutorials measure in the NEMA IQ phantom of the Siemens Symbia T2 scan, on a grid of 0.8 mm voxels in the CT's frame: `sphere_37mm` to `sphere_10mm` (balls of the inside diameters, at the centres fitted to the scanner's reconstruction), `lung_insert` (the CT's foam) and `background` (the water at least 15 mm from anything else in the CT, such as the phantom's walls, lid and base, the lung insert and the spheres' stems, and 20 mm from every sphere). Made by `docs/tools/make_roi_masks.py`, which records the fit in the header |

All are PyTomography tutorial data, under CC BY 4.0, like the rest of the PyTomography tutorial records.
