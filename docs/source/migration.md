% DRAFT for the v4 docs site. It replaces docs/source/migration.md on the docs/v4-site branch (same path, so the
% index's "Migrate to v4 <migration>" entry and every link to it keep working). Written from the diff of v3.4.0 and
% development at 0af8495 (10 October 2026). Before publishing:
%   - pyproject.toml still says requires-python >=3.9 and torch>=1.10.2; this page says 3.10 and 2.4, as install.md does.
%   - #294 (memory estimates and plans) is not merged; its section is marked "(coming in 4.0)". Drop it if it misses rc1.
%   - The ARF PSF section of the previous page is left out: spectarf and arf_to_psf are not in development yet.
%   - "Reading and saving data" follows the I/O branches feature/io-pet-ct 052c9c4 and feature/io-core 421da85 (12 October).

# Moving from v3 to v4

PyTomography 4 is a major release. Reading and saving data changed for every modality: one call per scan, lengths in
mm. Past the readers, most code written for v3.4 runs unchanged, and much faster. You need to change your code or your
environment if you use:

| If you use | What changes | Section |
|---|---|---|
| Any reader, attenuation map, mask or writer | One call per scan returns `(projections, object_meta, proj_meta)`. Lengths are mm, k runs foot to head, GE PET is upright. The v3 names are gone. | [Reading and saving data](#reading-and-saving-data) |
| PET or CT | Install parallelproj 2 from conda-forge. parallelproj 1.x no longer works. | [parallelproj 2](#install-parallelproj-2-for-pet-and-ct) |
| `FilteredBackProjection`, `RampFilter`, `HammingFilter`, `projection_type='FBP'` | A new call, a new default filter, and the old filter classes are gone. | [Filtered back projection](#filtered-back-projection) |
| PETSIRD files | Only PETSIRD 0.9 is read, with an optional extra. The reading functions are new. | [PETSIRD](#petsird-version-09-only) |
| Large PET sinograms (TOF, clinical scanners) | They can come back as a `LazySinogram`. A memory budget sizes the big steps. | [PET memory](#pet-memory-one-budget-and-lazy-sinograms) |
| `compute_sss_sparse_sinogram` | It returns a `SparseSinogram`. | [Scatter](#single-scatter-simulation) |
| The DICOM-CT-PD reader or `CTGen3SystemMatrix` | Better defaults change the images. | [CT](#ct-new-defaults-change-the-images) |
| Your own SPECT transform | The angle index is an `int`. | [SPECT](#spect-much-faster) |
| List mode PET with randoms or scatter, GATE normalization | Bug fixes change the numbers. | [Numbers that change](#numbers-that-change) |
| Tutorial data | `pytomography.datasets.fetch()` downloads it. | [Tutorial data](#tutorial-data) |

## What you get

| Workload | v3.4 | v4.0 |
|---|---|---|
| SPECT projector, 128³, 96 angles, attenuation and PSF, one forward projection | 538 ms | 46 ms (29 ms with `cache_probabilities=True`) |
| GE StarGuide NEMA IQ phantom, OSEM 10×10 | 38.0 s | 6.2 s |
| Ac-225 SIMIND phantom with a Monte Carlo PSF operator, OSEM 50×4 | 299 s | 18.5 s |
| GATE list mode TOF PET, 50.8 M events, OSEM 2×14 | 3.6 s | 1.3 s |
| Helical CT (TCIA LDCT case C145), OS-SART 3×40 | 55 s | 32 s |
| RAM for the GATE TOF PET tutorials | 80-105 GB | under 25 GB |

Timings are from an RTX 5090. Unless a section below says otherwise, reconstructions agree with v3.4 to float32
rounding.

## Requirements

| | v3.4 | v4.0 |
|---|---|---|
| Python | 3.9 or newer | 3.10 or newer |
| PyTorch | 1.10 or newer | 2.4 or newer |
| parallelproj (PET, CT) | 1.x | 2.x, from conda-forge |
| numba | not used | required (pip installs it) |
| petsird (PETSIRD files) | a bundled 2023 prototype | ETSI's `petsird` 0.9: `pip install "pytomography[petsird]"` |
| CuPy | required by parallelproj 1.x on the GPU | optional: fast helical CT filtered back projection and TOF scatter |
| idc-index | not used | optional: only to fetch the TCIA CT case `CT/ldct-c145` |

## Reading and saving data

One call reads a scan: every reader returns `(projections, object_meta, proj_meta)` in one standard form for every
modality, and `object_meta.affine` says where every voxel is in the patient. The
<a href="architecture.html#io">Data and conventions</a> tab of the Architecture page shows each data source, its own
conventions and what its reader does; its <a href="architecture.html#io-standard">standard form</a> table is the whole
convention on one screen. The v3 names are gone, not deprecated: they raise `ImportError` or `AttributeError`.

```python
from pytomography import io

# v3
object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=0)
photopeak = dicom.get_projections(file_NM, index_peak=0)
scatter = dicom.get_energy_window_scatter_estimate(file_NM, index_peak=0, index_lower=1, index_upper=2)
attenuation_map = dicom.get_attenuation_map_from_CT_slices(files_CT, file_NM, index_peak=0)
psf_meta = dicom.get_psfmeta_from_scanner_params('SY-ME', energy_keV=208, intrinsic_resolution=0.38)

# v4
projections, object_meta, proj_meta = io.SPECT.read_dicom(file_NM)   # [W, θ, r, z], mm
w = proj_meta.energy_windows[0]                                       # .photon_keV, .ranges_keV
photopeak = projections[0]
scatter = io.SPECT.energy_window_scatter(projections, proj_meta, peak=0, lower=1, upper=2)
attenuation_map = io.attenuation_map(files_CT, object_meta, energy_keV=w.photon_keV)   # 1/mm
psf_meta = io.SPECT.collimator_psf('SY-ME', w.photon_keV, intrinsic_fwhm_mm=3.8)
io.save_dicom(recon, folder, object_meta, reference=file_NM)          # placed; NM; counts
```

**Three changes move arrays or numbers:**

- **mm everywhere.** SPECT lengths were in cm; every length is now in mm and μ in 1/mm: `GaussianFilter(2)` (2 cm)
  becomes `GaussianFilter(20)`, `intrinsic_resolution=0.38` becomes `intrinsic_fwhm_mm=3.8`, and KEM's
  `distance_kernel_params=[0.4]` becomes `[4]` (the default σ is 10 mm). `SPECTObjectMeta` warns when `dr` looks like cm.
- **k runs foot → head** for every reader: `k = 0` is the most inferior slice. CT-PD and StarGuide arrays are flipped
  along axis 2 compared with v3; their saved files land where they did. PETSIRD rings are numbered foot → head.
- **GE PET is upright.** `read_ge_hdf5` turns the scanner 180° about z, so the bed lies posterior; v3's GE images were
  upside down. The hosted sphere mask `fdg_spheres.seg.nrrd` has a corrected origin: `io.load_mask(file).on(object_meta,
  label=k)` puts it on the v4 grid.

### Frame and writers

| v3 | v4 |
|---|---|
| `object_meta.affine_matrix`, `io.patient_affine` | `object_meta.affine`: always set, derived from `dr`, `shape` and `centre` |
| `ObjectMeta(dr, shape)` | `ObjectMeta(dr, shape, centre=(0, 0, 0), frame='scanner', units=None, modality=None)`; `.with_grid(dr=, shape=)`, `.subsample(f)` |
| `save_dicom(image, folder, affine=A, modality=...)` | `io.save_dicom(image, folder, object_meta, *, reference=, units=, modality=, slice_thickness=, series_description=, overwrite=)`: position, units and modality from `object_meta`; an unknown modality raises |
| `save_nifti(image, path, affine=A)` | `io.save_nifti(image, path, object_meta)` |
| an array with an affine you know | `io.save_dicom(io.Image(x, A, units=..., modality=...), folder)` |
| `dicom.save_dcm`, `io.shared.create_ds`, `io.shared.get_header_value` | `io.save_dicom` |

### Images

| v3 | v4 |
|---|---|
| `open_multifile`, `open_singlefile`, `_get_affine_*` | `io.load_image(source)`: `.data` (numpy x, y, z), `.affine`, `.units`, `.modality`, `.meta`, `.labels` |
| `align_images_affine(...)` | `io.load_image(source).on(object_meta, fill=...)`: averages when downsampling; `mode='nearest'`, `label=`, `threshold=` |

### SPECT

| v3 | v4 |
|---|---|
| `dicom.get_metadata` + `get_projections` | `io.SPECT.read_dicom(file, energy_window=None, *, time_slot=None)`: [W, θ, r, z], or [θ, r, z] for one window, never squeezed; multi-rotation files raise; gated files need `time_slot=` |
| `index_peak=` | `energy_window=` |
| `load_multibed_projections` | `io.SPECT.read_dicom_beds(files)`: a list of triples, inferior bed first |
| `stitch_multibed(recons, files_NM)` | `io.SPECT.stitch_beds(recons, object_metas)` → `(image, object_meta)` |
| `PGAAMultiBedSPECT(files_NM, algorithms)` | `PGAAMultiBedSPECT(algorithms)`: each bed's grid from its system matrix; masks go on the stitched `object_meta` |
| `get_starguide_metadata`, `get_starguide_projections`, `get_starguide_patient_affine` | `io.SPECT.read_starguide(files, energy_window=None, *, files_CT=None, object_meta=None, nearest_theta=0.1)`; `StarGuideProjMeta.times` (ms) → `frame_duration_s` (s) |
| `simind.get_metadata` + `get_projections` + `combine_projection_data` | `io.SPECT.read_simind(headers, energy_window=None, *, weights=None, radii_file=None)` |
| `get_psfmeta_from_scanner_params(name, energy_keV, intrinsic_resolution=cm)` | `io.SPECT.collimator_psf(name, photon_keV, intrinsic_fwhm_mm=0, *, intrinsic_fwhm_140keV_mm=None, material=, shape='gaussian' \| 'square')` |
| `simind.get_psfmeta_from_header` | `io.SPECT.simind_psf(header, photon_keV=None)` |
| `get_energy_window_scatter_estimate(_projections)` | `io.SPECT.energy_window_scatter(projections, proj_meta, peak, lower, upper=None, *, sigma_r_mm=, sigma_z_mm=)`: the windows' widths from `proj_meta` |
| `get_energy_window_bounds`, `get_window_width` | `proj_meta.energy_windows[i]`: `.ranges_keV`, `.lower_keV`, `.upper_keV`, `.width_keV`, `.photon_keV`; also `proj_meta.frame_duration_s`, `.acquisition_start`, `.nuclide`, `.units` |
| `get_FOV_mask_from_projections` | `io.SPECT.fov_mask(projections)` |
| `subsample_metadata`, `subsample_projections_and_modify_metadata` | `object_meta.subsample(f)`, `proj_meta.subsample(f, angle_factor, angle_start)`, `io.SPECT.shared.subsample_projections` |
| `PatientPosition` assumed feet first supine | placement from the orientation tags; `io.SPECT.dicom.patient_position(ds)` is only a check, and positions other than FFS warn |

### Attenuation maps and masks

| v3 | v4 |
|---|---|
| `dicom.get_attenuation_map_from_CT_slices`, `CT_to_mumap`, `get_attenuation_map_from_file`, `simind.get_attenuation_map`, `get_starguide_attenuation_map_from_CT_slices` | `io.attenuation_map(source, object_meta, energy_keV=None, *, technique='table', layout=None, units=None)` → μ in 1/mm on the grid. SPECT passes `energy_keV` (`proj_meta.energy_windows[i].photon_keV`); PET defaults to 511. Water is checked against NIST |
| `technique='from_table'`, `'from_cortical_bone_fit'` | `'table'`, `'cortical_bone_fit'`, `'nist'`, or a function of the HU |
| `smi_index_22=3`, `smi_index_22=5` | `layout='mu'`, `layout='mu-castor'`: required for SIMIND maps |
| `SPECTAttenuationTransform(filepath=files_CT)` | `SPECTAttenuationTransform(attenuation_map=io.attenuation_map(files_CT, object_meta, energy_keV=...))` |
| `get_aligned_rtstruct(rt, nm, folder, name)` | `io.load_mask(rt, reference=folder, rois=[name]).on(object_meta, label=name, threshold=0.5)` |
| `get_aligned_nifti_mask(nii, ..., idx)` | `io.load_mask(nii).on(object_meta, label=idx, threshold=0.5)` |

### PET

| v3 | v4 |
|---|---|
| `gate.get_detector_info` + `get_detector_ids_from_root` | `io.PET.read_gate(root_files, macro, *, object_meta=None, events='prompts', tof_meta=None, world_axes=None, world_origin=None, doi_mm=0, ...)` → `(events, object_meta, proj_meta)`; `info` is `proj_meta.info`; hosted datasets take their placement from the registry; out-of-range TOF bins are dropped |
| `mean_interaction_depth` | `doi_mm` |
| `gate.get_patient_affine_from_nifti` | nothing: `object_meta.affine` (save with `io.save_nifti(image, path, object_meta)`) |
| `gate.get_attenuation_map_nifti(path, meta)` | `io.attenuation_map(path, object_meta, units='1/cm')` |
| `gate.get_aligned_attenuation_map(hv, meta)` | `io.attenuation_map(io.PET.gate_image(hv, proj_meta, units='1/cm'), object_meta)` |
| `gate.get_normalization_weights_cylinder_calibration` | `io.PET.gate_normalization(files, info, cylinder_radius_mm)` (NaN → 0) |
| `gate.get_norm_sinogram_from_listmode_data` | `io.PET.normalization_sinogram` |
| `gate.get_sinogram_from_root_data` | `io.PET.listmode_to_sinogram(io.PET.read_gate(...)[0], info)` |
| `gate.listmode_to_sinogram`, `sinogram_to_listmode`, `smooth_randoms_sinogram`, `randoms_sinogram_to_sinogramTOF` | `io.PET.<same name>` (the last returns a new tensor: use its return value) |
| `gate.remove_events_out_of_bounds` | `io.PET.remove_events_outside(events, lut_or_info, object_meta)` |
| `gate.get_radius` | `gate.lor_radius` |
| `shared.get_scanner_LUT`, `shared.crystal_pair_index` | `io.PET.scanner_lut`, `io.PET.crystal_pair_index` |
| the trans and axial id helpers | `io.PET.scanner.detector_ids_from_parts`, `crystal_parts` |
| `info['firstCrystalAxis']` | `rsector0_angle_deg`, `transaxial_direction`, `axial_direction` |
| `info['energy_window_low']`, `['energy_window_high']`, `['TOF']` | `energy_window_low_keV`, `energy_window_high_keV`; `TOF` removed |
| `clinical.get_detector_info('discovery_MI')` | `io.PET.scanner_info('discovery_MI')` |
| `clinical.get_tof_meta` | `proj_meta.tof_meta` |
| `clinical.get_detector_ids_hdf5` | `io.PET.read_ge_hdf5(listmode_file)` |
| `get_weights_hdf5`, `get_additive_term_hdf5`, `get_sensitivity_ids[_and_weights]_hdf5` | `io.PET.read_ge_corrections(file)` → `(event_weights, additive_counts, pair_ids, pair_weights)` |
| `petsird.get_detector_info`, `get_detector_ids`, `get_sensitivity_weights`, `get_TOF_meta[_from_header]`, `get_frame_transform` | `io.PET.read_petsird(path, *, up=, towards_bed=, patient_position=, time_interval_s=, header=, events=)` (times in s) |
| `petsird.get_scanner_LUT[_from_header]` | `io.PET.petsird.element_positions` (PETSIRD's own numbering) |
| `patient_orientation=`, `info['petsird_alignment_deg']` | `patient_position=`, `rsector0_angle_deg` |

### CT

| v3 | v4 |
|---|---|
| `dicom_ct_pd.get_projections_and_metadata_gen3(paths, low_signal_filter=)` | `io.CT.read_dicom_ct_pd(folder, *, filter_low_signal=True, object_meta=None, ...)` → `(projections, object_meta, proj_meta)`; k foot → head |
| `proj_meta.get_patient_affine(object_meta)` | `object_meta.affine` |
| `col_det_spacing` | `col_angle_spacing` (radians) |
| `sorted_paths`, `get_geometry_info_from_datasets_gen3`, `get_photon_statistics`, `get_water_attenuation`, `get_correction_flags` | `proj_meta.photon_counts`, `.water_attenuation`, `.correction_flags` |
| HU by hand | `io.CT.to_hu(mu, proj_meta)`, `io.CT.from_hu(hu, proj_meta)` |
| the micro-CT tutorial's own reading code | `io.CT.read_nikon(folder, binning=4, cor_mm=0.24)` |

## Install parallelproj 2 (for PET and CT)

PET and CT need parallelproj 2. It is on conda-forge, not PyPI. SPECT does not need it, and importing PyTomography
never does: only `pytomography.projectors.PET` and `pytomography.projectors.CT` do, and they say how to install it
when it is missing.

**v3:**

```bash
conda install -c conda-forge libparallelproj parallelproj cupy
```

**v4:** start from a new environment.

```bash
conda create -n pytomography -c conda-forge python=3.12 "parallelproj>=2" numba
conda activate pytomography
pip install pytomography
```

To use the GPU, choose the `libparallelproj` build that matches your CUDA version, and install the PyTorch build for
that CUDA version before PyTomography (see the [PyTorch instructions](https://pytorch.org/get-started/locally/)):

```bash
conda install -c conda-forge "libparallelproj=*=cuda130*" "parallelproj>=2"
```

```{note}
In a conda environment, install numba from conda-forge, as above, so that it matches conda's numpy. pip would
install its own.
```

parallelproj 2 provides the module `parallelproj_core`. Check it with `python -c "import parallelproj_core"`. If your own
code called parallelproj 1.x functions such as `parallelproj.joseph3d_fwd`, those are gone in parallelproj 2.

**TOF PET results change.** Non-TOF reconstructions match v3 to 0.002%. TOF reconstructions change by about 4% RMS,
because parallelproj 2 weights TOF bins differently: each weight is 1.2% higher than before, and bins more than 3
standard deviations from the event get no weight. In our GATE mMR test, total activity changed by 0.3%.

## Filtered back projection

`FilteredBackProjection` is now one algorithm for parallel-hole SPECT, helical CT of third-generation scanners (with or
without a flying focal spot) and circular cone-beam CT (FDK). Give it the projections and the system matrix, then call
it with no arguments. It does not use the system matrix's transforms, so the image is not corrected for attenuation,
collimator blurring or scatter.

**SPECT, v3** (this crashed on any input, [#227](https://github.com/PyTomography/PyTomography/issues/227)):

```python
from pytomography.algorithms import FilteredBackProjection

fbp = FilteredBackProjection(projections, system_matrix)
recon = fbp(projections)
```

**SPECT, v4:**

```python
from pytomography.algorithms import FilteredBackProjection

recon = FilteredBackProjection(projections, system_matrix, filter="ram-lak")()
```

**Cone-beam CT, v3:**

```python
recon_FDK = system_matrix.backward(projections, projection_type="FBP")
```

**Cone-beam CT, v4** (the same image):

```python
recon_FDK = FilteredBackProjection(projections, system_matrix, filter="ram-lak")()
```

**Helical CT, new in v4:**

```python
from pytomography import io
from pytomography.metadata import ObjectMeta
from pytomography.projectors.CT import CTGen3SystemMatrix

projections, object_meta, proj_meta = io.CT.read_dicom_ct_pd(
    folder, object_meta=ObjectMeta(dr=(1.0, 1.0, 1.0), shape=(512, 512, 384)))
system_matrix = CTGen3SystemMatrix(object_meta, proj_meta, device="cpu")
recon = FilteredBackProjection(projections, system_matrix, filter="hann")()
```

With CuPy installed, the helical back projection runs as one fused CUDA kernel. The helical options are
`slice_thickness` (mm), `gpu_budget` (bytes), `Q` and `k_range`, passed to `FilteredBackProjection`.

```{important}
The default filter is now `"hann"`. In v3 it was the ramp alone: pass `filter="ram-lak"` for that.
```

### Filters

A filter is a window applied on top of the ramp. Frequencies are in cycles per mm, for every modality (SPECT spacings
are in mm too, so divide a cut-off in cycles per cm by 10).

| v3 | v4 |
|---|---|
| `pytomography.utils.RampFilter` | `filter="ram-lak"` |
| `pytomography.utils.HammingFilter(wl, wh)` | no direct equivalent; see below |
| | `"shepp-logan"`, `"hann"`, `"hamming"`, `"cosine"` |
| | `HannFilter(cutoff=0.8)`, `GeneralizedHammingFilter(alpha, cutoff)`, `CosineFilter(cutoff)`: the cut-off as a fraction of the Nyquist frequency |
| | `TabulatedFilter(frequencies, values)`, e.g. a measured scanner kernel |
| | any function of the frequency, e.g. a Butterworth window |

v4's `"hamming"` is the classic Hamming window, not v3's `HammingFilter`. v3's filter was 1 below `wl`, a cosine
taper to 0 at `wh`, and 0 above, with `wl` and `wh` as fractions of the Nyquist frequency. To keep it, pass it as a
function:

```python
import torch
from pytomography.utils import HannFilter

recon = FilteredBackProjection(projections, system_matrix, filter=HannFilter(cutoff=0.8))()

# a Butterworth window with a cut-off of 0.5 cycles per cm and order 5
butterworth = lambda f: 1 / torch.sqrt(1 + (f / 0.05) ** 10)
recon = FilteredBackProjection(projections, system_matrix, filter=butterworth)()

# v3's HammingFilter(wl=0.02, wh=0.93); proj_meta.dr is in mm
f_nyquist = 1 / (2 * proj_meta.dr[0])   # cycles per mm
wl, wh = 0.02 * f_nyquist, 0.93 * f_nyquist

def v3_hamming(f):
    f = f.abs()
    taper = 0.5 * (1 + torch.cos(torch.pi * (f - wl) / (wh - wl)))
    return torch.where(f <= wl, torch.ones_like(f), torch.where(f <= wh, taper, torch.zeros_like(f)))

recon = FilteredBackProjection(projections, system_matrix, filter=v3_hamming)()
```

### Removed

| v3 | v4 |
|---|---|
| `FilteredBackProjection(projections, system_matrix, filter=RampFilter)` then `fbp(projections)` | `FilteredBackProjection(projections, system_matrix, filter="ram-lak")()` |
| `pytomography.utils.RampFilter`, `HammingFilter` | `filter="ram-lak"`, a filter class, or a function (above) |
| cone beam: `system_matrix.backward(projections, projection_type="FBP")` | `FilteredBackProjection(projections, system_matrix, filter="ram-lak")()` |
| cone beam: `system_matrix.forward(object, FBP_post_weight=..., projection_type=...)` | removed |
| `projectors.CT.ct_conebeam_flatpanel_system_matrix.FBP_filter`, `get_discrete_ramp_FFT` | `pytomography.utils.ramp_filter`, `pytomography.utils.fourier_filters.ramp_filter_response` |
| `CTGen3SystemMatrix.forward(object, subset_idx, *args, **kwargs)` and `backward(...)` | no extra arguments: `backward(g, projection_type="FBP")` raises `TypeError` |

## PETSIRD: version 0.9 only

v4 reads PETSIRD 0.9, the format of ETSI's converters (STIR2PETSIRD, GATE to PETSIRD, CASToR) and STIR. The header is
read with ETSI's `petsird` package, which is an optional extra. The events are read by PyTomography's own compiled
reader: a 60-minute Siemens mMR scan takes about 10 s.

```bash
pip install "pytomography[petsird]"
```

The bundled copy of the 2023 prototype, `pytomography.io.PET.prd`, is removed. **Files in the prototype format can no
longer be read.** Convert the original data again with a current ETSI converter.

**v3:**

```python
from pytomography.io.PET import petsird

detector_ids, header = petsird.get_detector_ids(petsird_path, read_tof=True, return_header=True)
scanner_LUT = petsird.get_scanner_LUT_from_header(header)
tof_meta = petsird.get_TOF_meta_from_header(header)
```

**v4:** one call converts the file to PyTomography's standard PET form, as the GATE and GE readers do. Randoms,
scatter and the projectors then work on it unchanged.

```python
from pytomography import io
from pytomography.io.PET import petsird

header = petsird.read_header(petsird_path)
events, object_meta, proj_meta = io.PET.read_petsird(
    petsird_path, up="+x", towards_bed="+z", patient_position="HFS", header=header)
# events [N, 3] with TOF bins; proj_meta.info, proj_meta.tof_meta, proj_meta.weights_sensitivity
scanner_lut = io.PET.scanner_lut(proj_meta.info)
```

PETSIRD 0.9 does not say which way its gantry axes point or how the patient lies, so `read_petsird` takes them as
arguments. Files written by STIR2PETSIRD have `up="+x"` and `towards_bed="+z"`. Pass `header=` so that a large header
(2.9 GB of normalization tables for the mMR) is read only once. The reader also puts the file's energy window and
resolution into `info`, and the scatter simulation uses them.

## PET memory: one budget, and lazy sinograms

`pytomography.set_memory_budget(GB)` tells PyTomography how much RAM its memory-heavy steps may use. With it:

- steps that go through every crystal pair (normalization weights and sinograms, the list mode sensitivity image) run
  in blocks;
- a sinogram that would take more than a quarter of the budget is a `LazySinogram`, computed one subset of angles at a
  time when a reconstruction or the scatter scaling reads it;
- a reconstruction whose subsets would not fit stops before it starts, and says how many subsets would.

The results do not depend on the budget. Every PET tutorial sets one and runs in 25 GB.

```python
import pytomography

pytomography.set_memory_budget(20)   # GB of RAM
```

**Large sinograms can come back lazy, with or without a budget.** `listmode_to_sinogram`, `interpolate_sparse_sinogram`
and `get_sss_scatter_estimate` return a `LazySinogram` when the whole sinogram would take more than a quarter of the
budget, or more than 8 GB without a budget. A TOF sinogram of the Siemens mMR with 21 TOF bins takes 34.6 GB, so it is
lazy by default. The likelihoods and `sinogram_to_listmode` accept it as it is. It supports indexing along the angles
and arithmetic, which stays lazy. If your own code needs a tensor:

```python
# v3: always a tensor
sinogram = shared.listmode_to_sinogram(detector_ids, info, tof_meta=tof_meta)

# v4: ask for a tensor, or turn a lazy sinogram into one (this needs the memory of the whole sinogram)
sinogram = io.PET.listmode_to_sinogram(events, proj_meta.info, tof_meta=proj_meta.tof_meta, lazy=False)
sinogram = sinogram.to_dense()
```

**The list mode system matrix keeps the events on the GPU.** This is where most of its speed comes from. For 50.8 M
events it takes 1.2 GB of GPU memory. On a small GPU, keep them on the host: the projections are the same, only slower.

```python
system_matrix = PETLMSystemMatrix(object_meta, proj_meta, lor_device="cpu")
system_matrix.print_memory_usage()   # what stays on the device, and the expected peak of one subset
```

```{note}
Coming in 4.0 ([#294](https://github.com/PyTomography/PyTomography/pull/294)): `system_matrix.estimate_memory()` and
`plan_memory()` give a reconstruction's peak RAM and GPU memory before you run it, and the settings that fit a budget.
```

## Single scatter simulation

`get_sss_scatter_estimate` is called as before. Two lower-level functions changed:

```python
from pytomography.utils import sss

# v3: a dense sinogram tensor
scatter = sss.compute_sss_sparse_sinogram(object_meta, proj_meta, pet_image, attenuation_image)

# v4: a SparseSinogram; .to_dense() gives the tensor v3 returned
scatter = sss.compute_sss_sparse_sinogram(object_meta, proj_meta, pet_image, attenuation_image).to_dense()
```

The same holds for `compute_sss_sparse_sinogram_TOF`. The scatter simulation now takes the detector's energy window
and resolution from `proj_meta.info`: `energy_window_low_keV` and `energy_window_high_keV`, and `energy_resolution` as
the FWHM at 511 keV over 511 keV. The PETSIRD and GE readers fill them in from the file, and `read_gate` from its
`energy_window_keV=` and `energy_resolution=`. Without them, the v3 values are used (430 keV, no upper limit, 0.15).

## CT: new defaults change the images

The DICOM-CT-PD reader and `CTGen3SystemMatrix` give better images by default. To get v3's behaviour, turn them off:

```python
# v4 defaults
projections, object_meta, proj_meta = io.CT.read_dicom_ct_pd(folder)
system_matrix = CTGen3SystemMatrix(object_meta, proj_meta)

# as in v3 (apart from k, which now runs foot to head)
projections, object_meta, proj_meta = io.CT.read_dicom_ct_pd(folder, filter_low_signal=False, central_column="tag")
system_matrix = CTGen3SystemMatrix(object_meta, proj_meta, fov_mask=False)
```

- **Photon-starved rays are filtered** (`filter_low_signal=True`).
- **GE scans get the right central column.** GE counts `DetectorCentralElement` from the other end of the detector.
  v3 used the tag as stored, so GE reconstructions combined rays from opposite sides 1.4 mm apart.
- **Views are sorted by InstanceNumber.** This only changes which views share a subset. A folder can be passed.
- **Only voxels inside the field of view are modelled** (`fov_mask=True`). In v3 the voxels outside it reached 11 /mm
  (water is 0.019).
- **Reconstructions start from zero where no ray reaches**, instead of SART's initial value of 1.
- `CTGen3SystemMatrix` now works with its default device. In v3, only `device="cpu"` worked.
- The metadata no longer edits the tensors you give it, and `object_meta.affine` maps the image to patient
  coordinates.

## SPECT: much faster

Apart from [reading and saving](#reading-and-saving-data) and lengths in mm, SPECT code runs as before. Two transforms have an opt-in cache that trades memory for more speed:

```python
attenuation = SPECTAttenuationTransform(attenuation_map=attenuation_map, cache_probabilities=True)
psf = SPECTPSFTransform(psf_meta, cache_kernel=True)
```

`cache_probabilities` keeps the rotated attenuation map of every angle (about 1.6 GB at 128³). `cache_kernel` keeps
the Fourier transform of each PSF kernel (about 57 MB per distinct radius), so leave it off for body-contour orbits,
which have many radii.

**Your own transforms:** the projection loops now pass the angle index to `obj2obj` transforms as a Python `int`, not a
0-d tensor. Remove any `.item()` call on it.

```python
# v3
def forward(self, object_i, ang_idx):
    radius = self.radii[ang_idx.item()]

# v4
def forward(self, object_i, ang_idx):
    radius = self.radii[ang_idx]
```

**Saving:** `io.save_dicom` and `io.save_nifti` save any image, with every voxel in its place in the patient. The
modality and units come from `object_meta` (NM and counts for a SPECT reader's grid).

```python
from pytomography import io

io.save_dicom(recon, "recon_dicom", object_meta, reference=file_NM)
io.save_nifti(recon, "recon.nii.gz", object_meta)
```

## Numbers that change

Some bugs fixed in v4 changed results. Rerun anything you made with v3 that depends on them:

- **SPECT DICOM saved with `save_dcm(..., scale_by_number_projections=True)`** holds values N<sub>proj</sub> times too
  large when read with the DICOM rescale ([#230](https://github.com/PyTomography/PyTomography/issues/230)).
- **SPECT projections stored with a RescaleSlope other than 1** were read without it. v4 applies RescaleSlope and
  RescaleIntercept ([#232](https://github.com/PyTomography/PyTomography/issues/232)). Most vendors store a slope of 1.
- **List mode PET with randoms or scatter correction.** Half of all events read their randoms and scatter from the
  wrong sinogram bin ([#256](https://github.com/PyTomography/PyTomography/issues/256)). Most events got a neighbouring
  crystal pair's normalization weight ([#286](https://github.com/PyTomography/PyTomography/issues/286)). Together they
  change the GATE list mode tutorials' images by several percent.
- **TOF scatter estimates with `N_splits > 1`** were N_splits times too large, or NaN
  ([#258](https://github.com/PyTomography/PyTomography/issues/258)).
- **The scatter simulation's line integrals** were too long under parallelproj 1.x: the unscaled TOF estimate was
  12.8% too high. After scaling, the per-event estimate changes by about 1% RMS.
- **`gate.get_symmetry_histogram_from_ROOTfile(..., include_randoms=False)`** kept the randoms, and `True` dropped them.
  It now does what it says ([#289](https://github.com/PyTomography/PyTomography/issues/289)). Normalization weights you
  computed with `include_randoms=False` in v3 included the randoms.
- **Attenuation maps from a CT** are averaged over each SPECT voxel; v3 took the CT's value at the voxel's centre.
- **Cone-beam FDK with a centre-of-rotation offset** weighted the views with the offset's sign reversed. The micro-CT
  tutorial's FDK image changes by about 1 %.
- **DICOM images whose slices overlap or have gaps** were stretched along z: the readers took SliceThickness as the
  distance between slices ([#272](https://github.com/PyTomography/PyTomography/issues/272)).

## Tutorial data

The tutorials download their data with `pytomography.datasets.fetch()`, which checks it and unpacks it into
`PYTOMOGRAPHY_DATA` (or `~/pytomography_data`). You no longer download and unzip it by hand.

```python
# v3
folder = "/path/to/unzipped/SPECT/data"

# v4
from pytomography import datasets

folder = datasets.fetch("SPECT/Lu177-NEMA-SymT2")
```

`python -m pytomography.datasets list` lists the datasets.

## Everything removed or renamed

| v3 | v4 |
|---|---|
| parallelproj 1.x | parallelproj 2, from conda-forge |
| `pytomography.io.PET.prd` (the 2023 PETSIRD prototype) | ETSI's `petsird` 0.9: `pip install "pytomography[petsird]"` |
| `pytomography.utils.RampFilter` | `filter="ram-lak"` |
| `pytomography.utils.HammingFilter` | a filter class or a function ([Filters](#filters)) |
| `FilteredBackProjection.__call__(projections)` | `FilteredBackProjection(projections, system_matrix, ...)()` |
| cone beam `backward(..., projection_type)`, `forward(..., FBP_post_weight, projection_type)` | `FilteredBackProjection` |
| `ct_conebeam_flatpanel_system_matrix.FBP_filter`, `get_discrete_ramp_FFT` | `pytomography.utils.ramp_filter`, `pytomography.utils.fourier_filters.ramp_filter_response` |
| `CTGen3SystemMatrix.forward`/`backward` extra arguments | none |
| `compute_sss_sparse_sinogram(_TOF)` returning a tensor | `.to_dense()` of the `SparseSinogram` it returns |
| every v3 reader, `open_multifile`, `align_images_affine`, `save_dcm`, `affine_matrix` | [Reading and saving data](#reading-and-saving-data) |
