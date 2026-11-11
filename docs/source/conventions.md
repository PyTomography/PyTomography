# Conventions

PyTomography uses one set of units and one set of frames everywhere: in every function's arguments, every attribute, every table and every reader's output. Names carry no unit suffixes (`fwhm_r`, not `fwhm_r_mm`; `photon_energy`, not `photon_keV`). The unit comes from the convention below, and every function's docstring states the unit of each of its arguments. A test (`tests/test_docstring_units.py`) fails when one does not.

## Units

| Quantity | Unit | For example |
|---|---|---|
| Lengths: voxel and pixel sizes, positions, radii, distances, thicknesses, depths | mm | `object_meta.dr`, `proj_meta.radii`, `crystal_thickness`, `doi` |
| Spatial blurs (FWHM) | mm | `GaussianFilter(fwhm)`, `energy_window_scatter(fwhm_r=, fwhm_z=)`, `intrinsic_fwhm_140keV` |
| Angles: views, rotations, ring angles, CT geometry | degrees | `proj_meta.angles`, `rsector0_angle`, `CTGen3ProjMeta.source_phis`, `fwhm_theta` |
| Photon energies, energy windows, energy resolution (FWHM) | keV | `window.photon_energy`, `window.lower`, `energy=` of `io.attenuation_map` |
| Times and durations | s | `proj_meta.frame_duration`, `read_petsird(time_interval=)` |
| Attenuation coefficients | 1/mm | `io.attenuation_map(...)` |
| Projections | counts | every reader's projections |

* **Gaussians are given by their FWHM, never their standard deviation.** FWHM = 2.355 standard deviations. `n_sigmas` and `N_sigmas` only set how far a kernel reaches, in standard deviations.
* **A FWHM is in mm or degrees even where the data are in steps.** `io.PET.smooth_randoms_sinogram(fwhm_r=20, fwhm_theta=7, fwhm_z=40)` turns mm and degrees into radial bins, angles and rings with `io.PET.sinogram_spacing(info)`, measured from the scanner's crystals.
* **A reconstruction is in the units its system matrix makes it.** A system matrix of counts per decay (spectarf's absolute kernels weighted by the photons per decay, as in the Ac-225 tutorials) gives decays per voxel during one view. Divide by `proj_meta.frame_duration` (s) for Bq.
* **Other programs' units are converted where PyTomography reads or calls them:** DICOM-CT-PD's radians, SIMIND's cm, PETSIRD's ms. Inside PyTomography they never appear.

### Exceptions

* **Qualifiers that are not units:** `intrinsic_fwhm_140keV` is the FWHM at 140 keV, in mm. Likewise `energy_resolution_140keV` and `attenuation_map_140keV`.
* **Memory budgets** (`ram_gb`, `gpu_gb`) are in GB.
* **spectarf's own arguments** keep spectarf's names, with their suffixes (`psf_at(arfs, distances_mm, pixel_mm=)`, `DetectorResponse(intrinsic_fwhm_mm_140=)`). `spectarf.pytomography.camera_model(collimator, detector, camera)` turns PyTomography's camera tables into spectarf's camera, so a tutorial needs none of them.

### Before PyTomography 4

PyTomography 3 used cm, and gave smoothing as a standard deviation. Names that kept their meaning but changed unit, such as the SIMIND detector thicknesses (`crystal_thickness`, mm since 4.0), raise an error when a value can only be in cm, for example a crystal thinner than 3 mm. SPECT voxel sizes below 1 mm warn. PyTomography 3's smoothing arguments (`sigma_r`, `sigma_theta`, `sigma_z`, `GaussianFilter(FWHM=)`) still work, with a `DeprecationWarning` that gives the converted FWHM, until 5.0. [Migrate to v4](migration.md) lists every change.

## Frames and axes

* **Every reader returns one standard form**, `(projections, object_meta, proj_meta)`, whatever the source. Sources are converted in the readers; algorithms never special-case a source. The Architecture page's <a href="architecture.html#io">Data and conventions</a> tab gives each source's own conventions and what its reader does.
* **Objects** are `[x, y, z]` arrays indexed `(i, j, k)`. `object_meta.affine` (4 x 4) maps a voxel index to its centre in LPS mm: i towards the patient's left, j posterior, k superior. k = 0 is the most inferior slice.
* **SPECT projections** are `[W, θ, r, z]` (every energy window) or `[θ, r, z]`, with z rows from foot to head and the radius of each view in mm. A view's angle is β = (DICOM angle + 180) mod 360 degrees: β = 0 has the detector anterior, and β grows counter-clockwise seen from the feet. `proj_meta.energy_windows[i]` has `ranges`, `lower`, `upper`, `width` and `photon_energy`, in keV.
* **PET events** are int32 `[N, 2]` or `[N, 3]`: the two crystals, and the TOF bin. A detector id is `ring * crystals_per_ring + crystal`. The scanner's `info` holds its lengths in mm, `rsector0_angle` in degrees, and `energy_window_low` and `energy_window_high` in keV.
* **CT projections** are `−ln(I/I₀)`, `[views, columns, rows]` (helical) or `[views, u, v]` (cone beam). The focal spots and detector elements are in the object frame, in mm and degrees.

## For AI agents

The repository's `AGENTS.md` (which `CLAUDE.md` imports) states these conventions for an agent working on PyTomography, and <a href="llms.txt"><code>llms.txt</code></a> lists this page first. [For AI agents](ai.md) has the rest, such as checking that PyTomography uses the GPU.
