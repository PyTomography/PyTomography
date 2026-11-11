# PyTomography 4.0.0rc2 on the RTX 5060 Laptop GPU, against rc1

Every tutorial run overnight, 10 to 11 Nov 2026, on the laptop of `laptop-rtx5060-2026-11-09` (rc1, 9 to 10 Nov), to
compare rc2 with rc1 on the same machine. The reconstruction benchmarks of the same night are on `feature/benchmarks`,
in `benchmarks/results/2026-11-11-rtx-5060-laptop-gpu-rc2`.

**33 of 34 passed, as on rc1.** t_PETGATE_DIP ran out of GPU memory in the same cell (22) with the same error. With
`START_CHANNELS = 8` it passed in 1912 s, against 1936 s on rc1, at a 6.37 GB GPU peak. Its Deep Image Prior image
ended with contrast 4.13 and noise 0.50, against 4.08 and 0.51 on rc1. The OSEM images it compares with are the same
to two decimals. Overall, the 33 tutorials that passed took 245.7 min on rc2 against 261.1 min on rc1 (-6%). Every
change of more than 10% comes from first-use costs or from programs other than PyTomography (below).

## Which rc2

**Commit bef81841** ("Version 4.0.0rc2 (#447)" on `development`), not the PyPI release. On the evening of 10 Nov,
4.0.0rc2 was not on PyPI (`pip index versions pytomography --pre` listed rc1 only) and the repository had no
`v4.0.0rc2` tag. Set-up:

- **Environment:** `conda create --clone pytomo-rc1 -n pytomo-rc2`, then
  `pip install --no-deps --force-reinstall "git+https://github.com/PyTomography/PyTomography@bef81841"`, which builds
  as 4.0.0rc2. The check printed `4.0.0rc2`, a path in pytomo-rc2's site-packages, and `True`. `pip list` and
  `conda list` differ from rc1's in the PyTomography line only. PYTHONPATH was not set, and the rc1 environment is
  unchanged.
- **Notebooks:** `git worktree add --detach ../pt-rc2 bef81841`. Against the 9 Nov checkout (development at 4b47f4d),
  two notebooks differ. In t_PETGATE_DIP, one cell sets the network's size (#442). In t_cgls, only a markdown cell
  changed (#444). The runner is the same.

## How it was run

As on rc1 (`scripts/run_rc2.ps1`, the 9 Nov `run_ordered.ps1` with the rc2 paths):

- **Runner:** one `docs/tools/run_tutorials.py` call per tutorial, with `--no-viewer --ram-cap-gb 25
  --max-kernel-gb 28 --min-free-gb 2`, at below-normal priority, from PowerShell.
- **Order:** SPECT, then the two tutorials of no particular modality, CT, PET (t_PETGATE_DIP last), then the two
  Monte Carlo tutorials with a 3 h per-cell limit, as on rc1, and then t_PETGATE_DIP at 8 channels (an untracked copy
  of the notebook, removed afterwards).
- **Fresh caches:** every round used a new, empty `PYTOMOGRAPHY_OUTPUT`, shared within each of the desktop's two
  rounds, as on rc1. spectarf's ARF cache (`SPECTARF_CACHE`) and CuPy's kernel cache (`CUPY_CACHE_DIR`) also pointed
  at new, empty folders: the rc1 run started with both empty, and the 273 MB of ARFs it left would otherwise have
  spared rc2 every ARF generation.
- **Same conditions:** the same tutorial data (`PYTOMOGRAPHY_DATA`), SIMIND 8.0, NVIDIA driver 621.14, and Windows
  power mode "Best performance" on AC power. The GPU's enforced power limit read 105 W at the start; it was not
  logged through the night. The GPU-bound cells match rc1 closely: t_arf_psf's two ARF cells took 233.0 and 360.3 s,
  against 233.3 and 360.4 s, and t_ac225_dualpeak 1256.6 s against 1256.3. So the GPU ran as it did on 9 Nov.
- **Timeline:** 10 Nov 23:17 to 11 Nov 04:07 local time (`progress.log`); the benchmarks followed until 04:43.

## Flags, and what they are

Flagged: every failure, every time that changed by more than 10%, and every peak (GPU or RAM) that rose by more than
0.3 GB.

- **t_PETGATE_DIP failed,** as on rc1: it needs 8 GB of GPU memory at the published size. At 8 channels it passes.
- **t_siminddata (-71%) and t_dicomdata (-52%) ran first.** rc1's times include one-off costs of a new environment
  and machine; rc2's environment is a copy, so it skipped them. The difference is in cells that do no
  reconstruction work:
  - t_siminddata's first import took 39.2 s on rc1 and 6.3 s here.
  - t_dicomdata's system matrix took 6.2 s against 0.4, and its first OSEM 10.4 s against 3.8.

  The later SPECT tutorials, the same code warmed up, are within 7% (t_algorithms -5%).
- **t_ac225_simind_recon (+32%: 449.5 s against 340.0) was a one-off.** All of it was in the cell that generates the
  ARFs (spectarf, the same in both environments): 266.8 s against 152.6. The ARF PSF reconstruction took 144.8 s
  against 150.4.

  Re-run after the benchmarks, one environment after the other with empty caches and output folders (`scripts/recheck.ps1`), it
  took 332.6 s on rc2 and 347.5 s on rc1, with the ARFs in 149.6 and 158.7 s (`recheck/`). The benchmarks' ARF
  generation, that night, took 148.8 s against 149.0 on rc1.
- **t_spect_mc (-11%) and t_spect_mc2 (-10%) are SIMIND.** All of the difference is in the cell that runs SIMIND, on
  20 processes on the CPU: 3310 s against 3729, and 2957 s against 3294. SIMIND is unchanged, and the cells around it
  are within a second. So this is SIMIND's run-to-run variation (the time of night, the CPU's temperature), not rc2.
- **t_starguide's GPU peak rose 0.39 GB (1.10 to 1.49), from rc2's SPECT back projection (#444).** Every SPECT
  tutorial that uses the PSF or attenuation rotation peaks 0.05 to 0.12 GB higher on rc2. t_starguide's system matrix
  is the largest, and it rose most. The tutorials whose peak comes from ARF PSFs (t_arf_psf, t_ac225_*) peak the same.
- **Peak RAM rose 0.4 to 1.0 GB in six tutorials:** t_uncertainty_spect, t_starguide, t_arf_psf, t_cgls, t_plotting and
  t_CT_microct. This is the kernel's committed memory, which on Windows includes what CUDA maps. In the SPECT ones it
  goes with the GPU rises above. But t_plotting and t_CT_microct run no SPECT projection, and t_CT_microct's GPU peak
  is the same to 0.01 GB. Peak committed memory moves by about a gigabyte between runs of unchanged code:
  t_PETGATE_scat_sino peaked 1.7 GB lower here. So these rises are within that variation, except for the SPECT part.

`cell_timings.json` gives every cell's time on rc2. `scripts/cell_diff.py` compares a tutorial's cells with rc1's.

## Files

| | |
|---|---|
| `report.json`, `report.md` | the runner's report for all 35 runs (34 tutorials and the 8-channel DIP), in run order |
| `cell_timings.json` | each code cell's wall time on rc2 |
| `progress.log` | each tutorial's start and result, local time (UTC-8) |
| `recheck/` | t_ac225_simind_recon re-run on rc2 and rc1 |
| `scripts/` | the run script, the 8-channel DIP copy, the comparison, the cell comparison and the re-check |

The executed notebooks (19 MB) are not included; they are on the laptop in `C:\v4runs\laptop-rc2\runs`.

## Totals

| Group | rc2 (min) | rc1 (min) | Change |
|---|---|---|---|
| SPECT | 74.1 | 73.6 | +1% |
| Any modality | 0.2 | 0.2 | -3% |
| CT and proton CT | 17.7 | 18.9 | -6% |
| PET | 49.0 | 51.0 | -4% |
| Monte Carlo SPECT | 104.7 | 117.3 | -11% |
| t_PETGATE_DIP at 8 channels | 31.9 | 32.3 | -1% |
| **The 33 tutorials that passed on both** | **245.7** | **261.1** | **-6%** |

## The flags in one table

| Tutorial | Flag |
|---|---|
| t_siminddata | time -71% |
| t_dicomdata | time -52% |
| t_uncertainty_spect | RAM +0.60 GB |
| t_starguide | GPU +0.39 GB; RAM +0.50 GB |
| t_ac225_simind_recon | time +32% |
| t_arf_psf | RAM +0.60 GB |
| t_cgls | RAM +0.70 GB |
| t_plotting | RAM +0.40 GB |
| t_CT_microct | RAM +1.00 GB |
| t_PETGATE_DIP | **failed** (as on rc1) |
| t_spect_mc | time -11% |
| t_spect_mc2 | time -10% |

## Every tutorial

### SPECT

| Tutorial | rc2 | Time rc2 (s) | Time rc1 (s) | Change | GPU rc2 (GB) | GPU rc1 (GB) | RAM rc2 (GB) | RAM rc1 (GB) | Flags |
|---|---|---|---|---|---|---|---|---|---|
| t_siminddata | passed | 19.7 | 67.7 | -71% | 0.48 | 0.39 | 4.6 | 4.5 | time -71% |
| t_dicomdata | passed | 11.5 | 23.8 | -52% | 0.47 | 0.39 | 4.7 | 4.5 | time -52% |
| t_algorithms | passed | 123.4 | 130.5 | -5% | 1.66 | 1.58 | 6.9 | 6.7 |  |
| t_accuracy_known_truth | passed | 56.5 | 56.2 | +1% | 0.51 | 0.42 | 4.9 | 4.7 |  |
| t_dicommultibed | passed | 20.4 | 19.3 | +6% | 0.49 | 0.44 | 5.2 | 5.2 |  |
| t_dualpeak | passed | 17.0 | 16.6 | +2% | 0.55 | 0.47 | 4.9 | 4.6 |  |
| t_uncertainty_spect | passed | 31.4 | 31.8 | -1% | 0.76 | 0.71 | 8.0 | 7.4 | RAM +0.60 GB |
| t_starguide | passed | 40.2 | 40.4 | -0% | 1.49 | 1.10 | 6.1 | 5.6 | GPU +0.39 GB; RAM +0.50 GB |
| t_ac225_simind_recon | passed | 449.5 | 340.0 | +32% | 2.11 | 2.11 | 9.7 | 9.6 | time +32% |
| t_CardiacReorientation | passed | 7.8 | 8.0 | -3% | 0.07 | 0.06 | 4.0 | 3.9 |  |
| t_algorithms_measured | passed | 303.1 | 310.9 | -3% | 1.77 | 1.69 | 7.3 | 7.1 |  |
| t_fbp_spect | passed | 11.3 | 12.1 | -7% | 0.53 | 0.43 | 4.9 | 4.7 |  |
| t_arf_psf | passed | 605.9 | 606.4 | -0% | 2.80 | 2.80 | 10.2 | 9.6 | RAM +0.60 GB |
| t_ac225_dicom_recon | passed | 453.9 | 453.8 | +0% | 1.84 | 1.84 | 8.9 | 8.8 |  |
| t_ac225_dualpeak | passed | 1256.6 | 1256.3 | +0% | 1.68 | 1.68 | 7.6 | 7.5 |  |
| t_cgls | passed | 474.5 | 467.6 | +1% | 0.80 | 0.75 | 7.1 | 6.4 | RAM +0.70 GB |
| t_kernel_deep_kernel_spect_recon | passed | 560.5 | 576.7 | -3% | 1.55 | 1.44 | 9.3 | 9.1 |  |

Passed on both: 74.1 min on rc2, 73.6 min on rc1 (+1%).

### Any modality

| Tutorial | rc2 | Time rc2 (s) | Time rc1 (s) | Change | GPU rc2 (GB) | GPU rc1 (GB) | RAM rc2 (GB) | RAM rc1 (GB) | Flags |
|---|---|---|---|---|---|---|---|---|---|
| t_examplesystemmatrix | passed | 6.5 | 6.7 | -3% | 0.00 | 0.00 | 2.7 | 2.7 |  |
| t_plotting | passed | 6.9 | 7.1 | -3% | 0.00 | 0.00 | 3.4 | 3.0 | RAM +0.40 GB |

Passed on both: 0.2 min on rc2, 0.2 min on rc1 (-3%).

### CT and proton CT

| Tutorial | rc2 | Time rc2 (s) | Time rc1 (s) | Change | GPU rc2 (GB) | GPU rc1 (GB) | RAM rc2 (GB) | RAM rc1 (GB) | Flags |
|---|---|---|---|---|---|---|---|---|---|
| t_CT_GEN3 | passed | 459.8 | 503.0 | -9% | 1.63 | 1.63 | 14.5 | 14.5 |  |
| t_CT_microct | passed | 56.0 | 59.6 | -6% | 6.66 | 6.66 | 13.1 | 12.1 | RAM +1.00 GB |
| t_fbp_ct | passed | 441.4 | 457.7 | -4% | 1.30 | 1.30 | 11.9 | 11.9 |  |
| t_pct_abdomen | passed | 107.1 | 112.1 | -4% | 0.82 | 0.82 | 6.7 | 6.7 |  |

Passed on both: 17.7 min on rc2, 18.9 min on rc1 (-6%).

### PET

| Tutorial | rc2 | Time rc2 (s) | Time rc1 (s) | Change | GPU rc2 (GB) | GPU rc1 (GB) | RAM rc2 (GB) | RAM rc1 (GB) | Flags |
|---|---|---|---|---|---|---|---|---|---|
| t_pet_introduction | passed | 418.5 | 437.8 | -4% | 2.91 | 2.91 | 16.4 | 16.4 |  |
| t_GE_HDF5 | passed | 165.9 | 166.2 | -0% | 1.80 | 1.80 | 9.2 | 9.2 |  |
| t_uncertainty_pet | passed | 120.7 | 120.0 | +1% | 2.07 | 2.07 | 10.8 | 10.8 |  |
| t_PETSIRD | passed | 260.9 | 266.4 | -2% | 5.98 | 5.98 | 21.8 | 21.8 |  |
| t_PETGATE_scat_sino | passed | 881.5 | 943.8 | -7% | 1.53 | 1.53 | 16.2 | 17.9 |  |
| t_PETGATE_scat_sinoTOF | passed | 775.2 | 808.0 | -4% | 5.61 | 5.61 | 18.3 | 18.3 |  |
| t_PETGATE_scat_lm | passed | 160.7 | 162.4 | -1% | 2.88 | 2.88 | 19.2 | 19.2 |  |
| t_PETGATE_scat_lmTOF | passed | 156.9 | 157.7 | -1% | 3.44 | 3.44 | 19.4 | 19.4 |  |
| t_PETGATE_DIP | failed | 703.2 | 739.8 |  |  |  | 19.0 | 19.0 | **failed** (as on rc1) |

Passed on both: 49.0 min on rc2, 51.0 min on rc1 (-4%).

### Monte Carlo SPECT

| Tutorial | rc2 | Time rc2 (s) | Time rc1 (s) | Change | GPU rc2 (GB) | GPU rc1 (GB) | RAM rc2 (GB) | RAM rc1 (GB) | Flags |
|---|---|---|---|---|---|---|---|---|---|
| t_spect_mc | passed | 3318.3 | 3737.9 | -11% | 0.58 | 0.46 | 4.9 | 4.6 | time -11% |
| t_spect_mc2 | passed | 2963.9 | 3301.1 | -10% | 0.53 | 0.46 | 5.0 | 4.8 | time -10% |

Passed on both: 104.7 min on rc2, 117.3 min on rc1 (-11%).

### t_PETGATE_DIP at 8 channels

| Tutorial | rc2 | Time rc2 (s) | Time rc1 (s) | Change | GPU rc2 (GB) | GPU rc1 (GB) | RAM rc2 (GB) | RAM rc1 (GB) | Flags |
|---|---|---|---|---|---|---|---|---|---|
| t_PETGATE_DIP_c8 | passed | 1912.1 | 1936.1 | -1% | 6.37 | 6.58 | 18.9 | 18.9 |  |

Passed on both: 31.9 min on rc2, 32.3 min on rc1 (-1%).

