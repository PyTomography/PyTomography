# Small files of the tutorial data

`pytomography.datasets.fetch()` copies these files into the dataset folders that need them. Each is pinned by size and
sha256 in `../registry.py`; change both together.

| File | Copied into | What it is |
|---|---|---|
| `ac225_psf_model.json` | `SPECT/SIMIND-Jaszak`, `SPECT/Ac225-NEMA-SymT2` | The Ac-225 PSF model fitted with SPECTPSFToolbox, which replaced the pickled PSF operator in the 2025 SPECT record |
| `fdg_spheres.seg.nrrd` | `PET/GE-DMI-NEMA` | 3D Slicer mask of the six spheres of the NEMA IQ phantom in the GE Discovery MI scan, for the PET uncertainty tutorial |

Both are PyTomography tutorial data, under CC BY 4.0, like the rest of the PyTomography tutorial records.
