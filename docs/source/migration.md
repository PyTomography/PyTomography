% DRAFT for the v4 docs site. It replaces docs/source/migration.md on the docs/v4-site branch (same path, so the
% index's "Migrate to v4 <migration>" entry and every link to it keep working). Written from the diff of v3.4.0 and
% development at 0af8495 (10 October 2026). Before publishing:
%   - pyproject.toml still says requires-python >=3.9 and torch>=1.10.2; this page says 3.10 and 2.4, as install.md does.
%   - #294 (memory estimates and plans) is not merged; its section is marked "(coming in 4.0)". Drop it if it misses rc1.
%   - The ARF PSF section of the previous page is left out: spectarf and arf_to_psf are not in development yet.

# Moving from v3 to v4

PyTomography 4 is a major release. Most SPECT code written for v3.4 runs unchanged, and much faster. You need to change
your code or your environment if you use:

| If you use | What changes | Section |
|---|---|---|
| PET or CT | Install parallelproj 2 from conda-forge. parallelproj 1.x no longer works. | [parallelproj 2](#install-parallelproj-2-for-pet-and-ct) |
| `FilteredBackProjection`, `RampFilter`, `HammingFilter`, `projection_type='FBP'` | A new call, a new default filter, and the old filter classes are gone. | [Filtered back projection](#filtered-back-projection) |
| PETSIRD files | Only PETSIRD 0.9 is read, with an optional extra. The reading functions are new. | [PETSIRD](#petsird-version-09-only) |
| Large PET sinograms (TOF, clinical scanners) | They can come back as a `LazySinogram`. A memory budget sizes the big steps. | [PET memory](#pet-memory-one-budget-and-lazy-sinograms) |
| `compute_sss_sparse_sinogram` | It returns a `SparseSinogram`. | [Scatter](#single-scatter-simulation) |
| The DICOM-CT-PD reader or `CTGen3SystemMatrix` | Better defaults change the images. | [CT](#ct-new-defaults-change-the-images) |
| Your own SPECT transform | The angle index is an `int`. | [SPECT](#spect-nothing-to-change-much-faster) |
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
from pytomography.io.CT import dicom_ct_pd
from pytomography.metadata import ObjectMeta
from pytomography.projectors.CT import CTGen3SystemMatrix

projections, proj_meta = dicom_ct_pd.get_projections_and_metadata_gen3(folder)
object_meta = ObjectMeta(dr=(1.0, 1.0, 1.0), shape=(512, 512, 384))
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
are in cm, so divide a cut-off in cycles per cm by 10).

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

# v3's HammingFilter(wl=0.02, wh=0.93), for SPECT projections with a spacing in cm
f_nyquist = 1 / (2 * proj_meta.dr[0] * 10)   # cycles per mm
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

These calls still work in v4, on PETSIRD 0.9 files. The detector ids are then PETSIRD's own element numbers.

**v4:** convert the file to PyTomography's standard PET form, as the GATE and GE readers do. Randoms, scatter and the
projectors then work on it unchanged.

```python
from pytomography.io.PET import petsird, shared

header = petsird.read_header(petsird_path)
info = petsird.get_detector_info(header, up="+x", towards_bed="+z", patient_orientation="HFS")
detector_ids = petsird.get_detector_ids(petsird_path, info=info, header=header)   # [N, 3] with TOF bins
weights = petsird.get_sensitivity_weights(header, info=info)                      # normalization, per crystal pair
scanner_LUT = shared.get_scanner_LUT(info)
tof_meta = petsird.get_TOF_meta(header)
```

PETSIRD 0.9 does not say which way its gantry axes point or how the patient lies, so `get_detector_info` takes them as
arguments. Files written by STIR2PETSIRD have `up="+x"` and `towards_bed="+z"`. Pass `header=` so that a large header
(2.9 GB of normalization tables for the mMR) is read only once. `get_detector_info` also puts the file's energy window
and resolution into `info`, and the scatter simulation uses them.

```{note}
`get_detector_ids` now reads energy bins only when you ask (`read_energy=True`). In v3, it read them whenever the
scanner had more than one energy bin, which added two columns.
```

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
sinogram = shared.listmode_to_sinogram(detector_ids, info, tof_meta=tof_meta, lazy=False)
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
and resolution from `proj_meta.info`: `energy_window_low` and `energy_window_high` in keV, and `energy_resolution` as
the FWHM at 511 keV over 511 keV. The PETSIRD reader fills them in. Without them, the v3 values are used (430 keV, no
upper limit, 0.15), so GATE and GE estimates do not change.

## CT: new defaults change the images

The DICOM-CT-PD reader and `CTGen3SystemMatrix` give better images by default. To get v3's behaviour, turn them off:

```python
# v4 defaults
projections, proj_meta = dicom_ct_pd.get_projections_and_metadata_gen3(paths)
system_matrix = CTGen3SystemMatrix(object_meta, proj_meta)

# as in v3
projections, proj_meta = dicom_ct_pd.get_projections_and_metadata_gen3(paths, low_signal_filter=False, central_column="tag")
system_matrix = CTGen3SystemMatrix(object_meta, proj_meta, fov_mask=False)
```

- **Photon-starved rays are filtered** (`low_signal_filter=True`).
- **GE scans get the right central column.** GE counts `DetectorCentralElement` from the other end of the detector.
  v3 used the tag as stored, so GE reconstructions combined rays from opposite sides 1.4 mm apart.
- **Views are sorted by InstanceNumber.** This only changes which views share a subset. A folder can be passed.
- **Only voxels inside the field of view are modelled** (`fov_mask=True`). In v3 the voxels outside it reached 11 /mm
  (water is 0.019).
- **Reconstructions start from zero where no ray reaches**, instead of SART's initial value of 1.
- `CTGen3SystemMatrix` now works with its default device. In v3, only `device="cpu"` worked.
- The metadata no longer edits the tensors you give it, and `proj_meta.get_patient_affine(object_meta)` maps the image
  to patient coordinates.

## SPECT: nothing to change, much faster

SPECT code runs as before. Two transforms have an opt-in cache that trades memory for more speed:

```python
attenuation = SPECTAttenuationTransform(filepath=files_CT, cache_probabilities=True)
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

**Saving:** `pytomography.io.save_dicom` and `save_nifti` save any image, with every voxel in its place in the patient.
SPECT images are saved as NM by default.

```python
from pytomography.io import save_dicom, save_nifti

save_dicom(recon, "recon_dicom", object_meta, modality="NM", reference=file_NM)
save_nifti(recon, "recon.nii.gz", object_meta)
```

`dicom.save_dcm` still works.

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
| `petsird.get_detector_ids(read_energy=None)` | `read_energy=False` by default |
