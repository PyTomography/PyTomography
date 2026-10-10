# Moving from v3 to v4

PyTomography 4 is a major release because it changes three things on purpose: PET needs parallelproj 2, SPECT PSFs have a new interface, and older Python and PyTorch versions are no longer supported. Most SPECT code written for v3.4 runs unchanged, and it runs much faster.

```{note}
This guide describes v4.0 as planned for the 30 October 2026 release. The SPECT, StarGuide and PSF speedups are merged. The parallelproj 2 change is in review ([#238](https://github.com/PyTomography/PyTomography/pull/238)), and the fixes listed at the end are still to be merged. This page is updated as they land.
```

## What you get

| Workload | v3.4 | v4.0 |
|---|---|---|
| SPECT projector, 128³, 96 angles, attenuation and PSF, one forward projection | 538 ms | 46 ms (29 ms with `cache_probabilities=True`) |
| GE StarGuide NEMA IQ phantom, OSEM 10×10 | 38.0 s | 6.2 s |
| Ac-225 SIMIND phantom with a Monte Carlo PSF operator, OSEM 50×4 | 299 s | 18.5 s |
| GATE list-mode TOF PET, 50.8 M events, OSEM 2×14 | 3.6 s | 1.3 s |
| Peak host memory, PET single scatter simulation | 44 GB | 10.5 GB |

Timings are from an RTX 5090. Reconstructions agree with v3.4 to within 1e-5 of their maximum.

## Requirements

| | v3.4 | v4.0 |
|---|---|---|
| Python | 3.9 or newer | 3.10 or newer |
| PyTorch | 1.10 or newer | 2.4 or newer |
| parallelproj (PET, CT) | 1.x | 2.x, from conda-forge |
| pydicom | 3.0 or newer | 3.0 or newer |
| SPECTPSFToolbox | optional | deprecated; replaced by ARF-based PSFs |
| CuPy | not used | optional, for fast helical CT filtered back projection |

## PET: parallelproj 2 (in review)

Install parallelproj 2 from conda-forge as described in [Installation](install.md). `PETLMSystemMatrix` gains two arguments:

- `lor_device` (default `pytomography.device`) keeps the event geometry on the GPU. `lor_device="cpu"` uses less GPU memory and gives identical projections, more slowly.
- `sort_events` (default `True`) orders events for faster projection.

`system_matrix.print_memory_usage()` reports what stays on the device and the expected peak memory per subset.

```{note}
parallelproj 2 clips rays at the faces of the image, where 1.x integrated out to one voxel beyond the outermost voxel centres. On its own that changes a GATE TOF list-mode reconstruction by 3.9% RMS. The PET system matrices therefore pad the object with one voxel of zeros, as the CT ones do, so the line integrals at the faces are those of 1.x again: the non-TOF reconstruction agrees with 1.x to 0.002% ([#251](https://github.com/PyTomography/PyTomography/pull/251), in review).
```

## SPECT: faster projectors, opt-in caches

Nothing needs to change in your code. Two transforms gain an opt-in cache that trades memory for speed:

```python
attenuation = SPECTAttenuationTransform(filepath=files_CT, cache_probabilities=True)
psf = SPECTPSFTransform(psf_meta, cache_kernel=True)
```

`cache_probabilities` stores the rotated attenuation map of every angle (about 1.6 GB at 128³). `cache_kernel` stores the Fourier transform of each PSF kernel (about 57 MB per distinct radius), so leave it off for body-contour orbits with many radii.

If you wrote a custom `obj2obj` transform, note that the projection loops now pass the angle index as a Python `int` instead of a 0-d tensor. Remove any `.item()` call on it.

## SPECT: PSFs from ARF tables (experimental)

v4 introduces PSFs built from angular response function (ARF) tables: one function takes an ARF and returns a PSF kernel at every source-detector distance the reconstruction needs. Kernels can have any shape, and the transform uses a true adjoint. In v4.0 this interface is experimental and may change. The [spectarf](https://github.com/PyTomography) generator, which builds ARF tables with GPU Monte Carlo, ships with v4.0 as its own package.

The Gaussian PSF from `get_psfmeta_from_scanner_params` still works. SPECTPSFToolbox operators still work in v4.0 but are deprecated and will be removed in v4.1.

## Filtered back projection: one class for SPECT and CT (in review)

`FilteredBackProjection` works again ([#227](https://github.com/PyTomography/PyTomography/issues/227)), now for SPECT with parallel-hole collimators, for 3rd-generation helical CT (weighted FBP, with or without a flying focal spot) and for cone-beam CT (FDK) ([#263](https://github.com/PyTomography/PyTomography/pull/263)). Give it the projections and the system matrix, then call it:

```python
from pytomography.algorithms import FilteredBackProjection

image = FilteredBackProjection(projections, system_matrix, filter='hann')()
```

It ignores the system matrix's transforms, so there is no attenuation, PSF or scatter correction. `filter` is one of `'ram-lak'`, `'shepp-logan'`, `'hann'`, `'hamming'` and `'cosine'`, a window with a cut-off such as `HannFilter(cutoff=0.8)`, or any function of frequency in cycles per mm. For example, a SPECT Butterworth filter with a cut-off of 0.5 cycles/cm and order 5 is `lambda f: 1 / torch.sqrt(1 + (f / 0.05) ** 10)`. With CuPy installed, the helical CT back projection runs as a fused CUDA kernel, which is much faster.

The old FBP interface is removed:

| v3.4 | v4.0 |
|---|---|
| `FilteredBackProjection(proj, angles)(...)` | `FilteredBackProjection(proj, system_matrix, filter='ram-lak')()` |
| `utils.RampFilter()` | `filter='ram-lak'` |
| `utils.HammingFilter(...)` | `filter='hamming'` |
| `system_matrix.backward(proj, projection_type='FBP')` (cone beam) | `FilteredBackProjection(proj, system_matrix, filter='ram-lak')()` |
| `system_matrix.forward(..., FBP_post_weight=..., projection_type=...)` (cone beam) | removed |

## Fixes planned for v4.0

- `FilteredBackProjection` works again, for SPECT and CT ([#227](https://github.com/PyTomography/PyTomography/issues/227), see above).
- `save_dcm(..., scale_by_number_projections=True)` no longer writes values N<sub>proj</sub> times too large ([#230](https://github.com/PyTomography/PyTomography/issues/230)). **If you saved DICOM files this way with v3.4, check their values.**
- SPECT DICOM rescale tags are applied when reading projections ([#232](https://github.com/PyTomography/PyTomography/issues/232)).
- Multi-bed reading works with pydicom 3.
