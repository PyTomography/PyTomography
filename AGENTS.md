# PyTomography: read this first (AI agents)

These are PyTomography's conventions, for an agent that uses PyTomography or changes it. The docs' Conventions page
(https://pytomography.readthedocs.io/en/latest/conventions.html) has them in full, and `llms.txt` lists every page.

## Units: one convention, no unit suffixes

Every argument, attribute and table value is in these units, and names never carry a unit suffix (`fwhm_r`, not
`fwhm_r_mm`; `photon_energy`, not `photon_keV`). Every docstring states the unit of each argument;
`src/pytomography/tests/test_docstring_units.py` checks that.

| Quantity | Unit |
|---|---|
| Lengths: voxel and pixel sizes, positions, radii, distances, thicknesses, depths, spatial FWHMs | mm |
| Angles: SPECT views, rotations, PET ring angles, smoothing along views | degrees |
| Photon energies, energy windows, energy resolution FWHMs | keV |
| Times and durations | s |
| Attenuation coefficients | 1/mm |
| Projections | counts |

* **Gaussians are given by their FWHM** (`fwhm`, `fwhm_r`, `fwhm_theta`, ...), never by a standard deviation. `n_sigmas`
  and `N_sigmas` only set how far a kernel reaches.
* **A reconstruction is in the units the system matrix makes it.** With SPECT kernels in counts per decay (spectarf's,
  weighted by the photons per decay), an image is in decays during one view; divide by the view's duration
  (`proj_meta.frame_duration`, s) for Bq.
* **Exceptions**, each stated in its docstring:
  * Qualifiers, not units: `intrinsic_fwhm_140keV` (the FWHM at 140 keV, in mm), `energy_resolution_140keV`,
    `attenuation_map_140keV`.
  * Memory budgets (`ram_gb`, `gpu_gb`): GB.
  * Other programs' units are converted where PyTomography reads or calls them: DICOM-CT-PD's radians, SIMIND's cm,
    PETSIRD's ms.
* **Angles are degrees everywhere**, CT geometry included (`CTGen3ProjMeta`, `CTConeBeamFlatPanelProjMeta`); code
  converts to radians only for its own trigonometry.
* **Smoothing follows the data's real spacing.** A FWHM is in mm or degrees even where the data are in steps: PET
  randoms smoothing (`io.PET.smooth_randoms_sinogram`) turns it into radial bins, angles and rings with
  `io.PET.sinogram_spacing(info)`.
* **PyTomography 3 used cm.** Some names are the same in 4 with mm (e.g. the SIMIND detector thicknesses, SPECT voxel
  sizes). Where a value looks like cm, PyTomography warns (voxel sizes below 1 mm) or raises (a crystal thinner than
  3 mm) and says so.

## Frames and axes

* **Readers return one standard form**, `(projections, object_meta, proj_meta)`, whatever the source. Sources are
  converted in the readers; algorithms never special-case a source.
* **Objects** are `[x, y, z]` arrays indexed `(i, j, k)`. `object_meta.affine` (4 x 4) maps a voxel index to its centre
  in LPS mm: i towards the patient's left, j posterior, k superior (k = 0 is the most inferior slice).
* **SPECT projections** are `[W, θ, r, z]` (every energy window) or `[θ, r, z]`, with z rows from foot to head and the
  radius of each view in mm. `proj_meta.energy_windows[i]` has `ranges`, `lower`, `upper`, `width` and
  `photon_energy`, all in keV.
* **PET events** are int32 `[N, 2]` or `[N, 3]`: the two crystals and the TOF bin. A detector id is
  `ring * crystals_per_ring + crystal`. The scanner's `info` holds lengths in mm, `rsector0_angle` in degrees and the
  energy window (`energy_window_low`, `energy_window_high`) in keV.

## Before running anything heavy

* Check that PyTomography uses the GPU on a machine that has one: the docs' "Using these docs with AI assistants"
  page, "Check that PyTomography uses the GPU".
* Plan memory with `estimate_memory()` and `plan_memory()`: the docs' "Memory and speed" page.

## Changing PyTomography

* New arguments: no unit suffix, the unit in the docstring ("in mm"), Gaussians by FWHM. Renaming a released argument:
  keep the old name working with `pytomography.utils._renamed.renamed_arguments`, as CONTRIBUTING.md's deprecation
  policy asks.
* Code goes into `development` by pull request, each closing an issue (`Closes #N`). Before pushing, run the tests as
  GitHub does (`PYTOMOGRAPHY_TEST_DEVICE=cpu pytest -m "not gpu and not data"`) and, on a GPU machine, the whole suite.
