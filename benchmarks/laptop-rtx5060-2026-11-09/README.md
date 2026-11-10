# Every tutorial on an RTX 5060 Laptop GPU, against the RTX 5090 desktop

These are the results of running all 34 tutorials of PyTomography 4.0.0rc1 on a laptop, 9 and 10 Nov 2026, for comparing
with the desktop's run of 8 Nov. They follow the hand-off page "Laptop Tutorial Benchmark" (its database has the same
results, exported in `db/`). This branch is for moving the results between machines. It is not meant to be merged
as it is: the executed notebooks alone are 19 MB.

## The two machines

| | Laptop | Desktop |
|---|---|---|
| GPU | RTX 5060 Laptop GPU, 8 GB (about 6.9 GB usable by CUDA on Windows), 384 GB/s | RTX 5090, 32 GB, 1792 GB/s |
| CPU | Core Ultra 9 275HX, 24 cores | Core Ultra 9 285K, 24 cores |
| RAM | 32 GB (about 22 GB available, after removing McAfee) | 64 GB |
| Tutorial data | internal SSD | USB hard disk |
| NVIDIA driver | 621.14 Studio (CUDA 13.5); Acer's 577.13 could not run the cu130 stack | |
| Power | Windows "Best performance", on AC | |

Both run Windows, with Python 3.12.16, PyTorch 2.14.1+cu130 and PyTomography 4.0.0rc1 from PyPI. The other packages are
the same too: parallelproj-core and libparallelproj 2.0.7 (cuda130), CuPy 14.2.0, spectarf 0.1.0, numba-cuda 0.30.4,
nvidia-cuda-nvcc-cu12 12.8.93, MONAI 1.6.1, petsird 0.9.1. SIMIND is 8.0. `db/machine/laptop.json` has the details.

## Results

33 of 34 passed. t_PETGATE_DIP ran out of GPU memory, as expected: it needs 8.0 GB.

| Group | Tutorials | Desktop | Laptop | Slower |
|---|---|---|---|---|
| SPECT, not Monte Carlo | 17 | 20.7 min | 73.6 min | 3.6× |
| Monte Carlo SPECT (SIMIND 8.0, 20 processes) | 2 | 77.6 min | 117.3 min | 1.5× |
| CT and proton CT | 4 | 8.3 min | 18.9 min | 2.3× |
| PET (t_PETGATE_DIP failed) | 8 | 36.8 min | 51.0 min | 1.4× |
| t_examplesystemmatrix, t_plotting | 2 | 0.2 min | 0.2 min | 1.0× |
| **Everything that passed** | **33** | **143.6 min** | **261.1 min** | **1.8×** |

Each tutorial's time, peak RAM and peak GPU memory on both machines are in `laptop_vs_desktop.md` and `.csv`.

- **GPU-heavy SPECT slows the most, 3 to 4×,** close to the memory-bandwidth ratio (4.7×). In the Ac-225 tutorials,
  the reconstructions with ARF kernels take most of the time: t_ac225_dualpeak's three OSEM runs took 280, 280 and
  560 s here, against 64, 61 and 126 s in the repo's desktop run (4.4×). Generating the ARFs takes about 130 to
  150 s per tutorial here. `cell_timings.json` gives every cell.
- **PET slows only 1.4×,** probably because these tutorials are mostly reading data, from the laptop's SSD against the
  desktop's USB disk.
- **t_siminddata (13.5×) ran first,** so its time includes one-off kernel compiling. Leave it out of a ranking.
- **Every tutorial peaked lower in RAM on the laptop.** t_PETSIRD peaked at 21.8 GB here against 25.2 GB. Where the work
  is the same, the GPU peaks are the same; some SPECT and list-mode PET tutorials use less on the smaller card.
- **t_CT_microct fits** in 8 GB, at 6.66 GB, without binning.

## How it was run

- **Runner and flags:** `docs/tools/run_tutorials.py`, with the page's flags (`--ram-cap-gb 25 --max-kernel-gb 28
  --min-free-gb 2 --no-viewer`), at below-normal priority.
- **One runner call per tutorial,** in Luke's order: SPECT, then the two tutorials of no particular modality, CT, PET,
  and Monte Carlo SPECT last (`scripts/run_ordered.ps1`). `PYTOMOGRAPHY_OUTPUT` was shared per desktop round, so cached
  steps were reused as they were on the desktop.
- **The two Monte Carlo tutorials ran with `--timeout 10800`.** Their SIMIND cell takes just over the default hour
  here, so the first t_spect_mc stopped at 3609 s (`reports/t_spect_mc_timeout1h.json`). The tutorials themselves are
  unchanged.
- **Tutorial data came only from `datasets.fetch()`:** 13 datasets, 40.8 GB on disk.
- **Timeline:** `progress.log` gives each tutorial's start and end, in local time (UTC−8).

## Files

| | |
|---|---|
| `laptop_vs_desktop.md`, `.csv` | every tutorial, laptop against desktop: time, peak RAM, peak GPU, slowdown |
| `reports/<tutorial>.json` | the runner's report for each tutorial; also `t_spect_mc_timeout1h` (the first try) and `t_PETGATE_DIP_c8` (below) |
| `notebooks/<tutorial>.ipynb` | the executed notebooks, with their outputs and per-cell timestamps |
| `cell_timings.json` | each code cell's wall time, from those timestamps (`scripts/cell_timings.py`) |
| `db/` | the hand-off page's database: `results/`, `machine/laptop.json`, `summary/laptop.json` (with the follow-ups) |
| `dip_network_memory.txt` | GPU memory to fit t_PETGATE_DIP's network at 12, 10, 8 and 6 channels (`scripts/measure_dip.py`) |
| `scripts/` | the run scripts, the table's script (`summarise.py`) and the measurement scripts |

## Follow-ups

- **[PyTomography/PyTomography#442](https://github.com/PyTomography/PyTomography/pull/442)** (closes #439): one cell at
  the top of t_PETGATE_DIP sets the network's size. With `START_CHANNELS = 8` the tutorial passed here in 1936 s, at a
  6.58 GB GPU peak (`reports/t_PETGATE_DIP_c8.json`, `notebooks/t_PETGATE_DIP_c8.ipynb`). The image ends with contrast
  4.08 and noise 0.51, against 4.24 and 0.48 at 12. The published default stays at 12; the desktop should confirm its
  images are unchanged.
- **SIMIND:** the downloads page's Windows "Ver 8.0" link serves the macOS file. The laptop used Luke's own copy of
  `smc_win_intel_64_current.zip` (18,083,636 bytes, SHA-256
  `fdd7246c5aa86a0bfdc357f7abe93ed0904c58215deef43b68e61d76c6b138aa`). Mirroring it needs Michael Ljungberg's
  permission.
- **The Installation page:**
  - the cu130 stack needs NVIDIA driver 580 or newer, which a new laptop may not have;
  - the CUDA 12 build of libparallelproj is named `cuda129`;
  - the cu128 example now installs PyTorch 2.11.
