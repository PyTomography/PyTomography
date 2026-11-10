# RTX 5060 Laptop GPU, 10 Nov 2026

All 13 cases passed, in 35.5 min, on the laptop that ran the tutorial benchmark (branch `benchmark/laptop-rtx5060`):

- **Machine:** RTX 5060 Laptop GPU (8 GB), Core Ultra 9 275HX (24 cores), 32 GB RAM with about 21 GB free.
- **Software:** Windows 11, NVIDIA driver 621.14, Windows power mode "Best performance", on AC power.
- **Environment:** the same as the tutorial runs. PyTomography 4.0.0rc1 and PyTorch 2.14.1+cu130 come from PyPI. parallelproj-core 2.0.7 and libparallelproj 2.0.7 (cuda130) come from conda-forge, and `machine` in the case files leaves them out (below).

## How it was run

The run was started from PowerShell, at commit 6a2723e, with these settings:

- **PATH:** the environment's DLL folders first, as the tutorial runner's `--path-prefix` does.
- **Data:** `PYTOMOGRAPHY_DATA` pointed at the tutorial data.
- **PET events:** `PYTOMOGRAPHY_OUTPUT` pointed at the tutorials' output folder. Every PET event file was already there (`caches_present` in the PET case files).

The command:

```
python run_laptop.py --min-free-gb 2
```

`run_laptop.py` imports `benchmarks/run.py` and calls its `main()` unchanged, after setting every case's RAM in `CASES` to 0:

```python
run.CASES = [(case, recon, gpu, 0) for case, recon, gpu, ram in run.CASES]
```

Without it, `run.py` would have skipped cases this laptop can run:

- **With the defaults:** the check before each case asks for `--min-free-gb` (default 10) plus the case's RAM from `CASES`, which holds the desktop's tutorial peaks. With about 21 GB free, `run.py` would have skipped 9 of the 13 cases. Of those that started, any case committing more than about 11 GB would have been stopped by `run_capped.py`.
- **With `--min-free-gb 2`:** the five PET cases would still have been skipped, because they need 22 to 25 GB free.

`run_capped.py` (stopped above 25 GB used, or below 2 GB free) and the GPU check before each case were left as they are.

`--min-free-gb 2` is the value the laptop's tutorial runs used. The smoke test (`--quick`) passed all 13 cases first; its folder is not included.

## Notes for `run.py`

- **Each case's own peak is far below its `CASES` RAM:**

  | Cases | Measured peak | RAM in `CASES` |
  |---|---|---|
  | PET | 11.2 to 16.9 GB | 20 to 23 GB |
  | spect_ac225_arf | 10.1 GB | 14 GB |
  | ct_helical_fbp, ct_microct | 9.6 to 10.1 GB | 15 GB |
  | pct_abdomen | 6.8 GB | 9 GB |

  Taking `CASES` from these peaks, measured on one machine and the other, would let a 32 GB machine run every case.
- **ct_microct passed the GPU check by 0.01 GB.** It needed 6.7 + 1.0 GB, and 7.71 GB was free. Its own peak was 6.06 GB (PyTorch) and 5.71 GB (on the device).
- **`machine()` misses parallelproj when it comes from conda-forge.** `parallelproj-core` installs only `parallelproj_core.pyd`, with no package metadata. `parallelproj_core.__version__` gives 2.0.7, and `parallelproj_core.cuda_enabled` says whether it is the CUDA build.
- **The Ac-225 OSEM ran only one repeat.** Its first repeat (278 s) is over `--max-seconds` (90).
- **ct_helical_fbp's loading depends on the disk cache.** It took 46 s in the smoke test, cold, and 12 s here.
