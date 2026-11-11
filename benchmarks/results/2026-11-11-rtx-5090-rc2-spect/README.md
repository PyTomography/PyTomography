# RTX 5090 desktop, PyTomography 4.0.0rc2, SPECT cases only, 11 Nov 2026

The three SPECT cases passed, in 5 min. They are the desktop's first benchmark results: the full run of the 13 cases
on rc2 was stopped overnight by a Windows Update restart, and the CT and PET cases were not run again.

- **rc2 here is commit bef81841** ("Version 4.0.0rc2 (#447)" on `development`), the tree that `main` has at the rc2
  merge (6a7ae835), installed as a wheel into a clone of the rc1 environment (`pip install --no-deps --force-reinstall`
  of a checkout at bef81841). Checked before the run: version 4.0.0rc2, imported from site-packages, with the exact
  back projection (#444).
- **Command:** `python benchmarks/run.py --only spect_lu177,spect_ac225_arf --out benchmarks/results/2026-11-11-rtx-5090-rc2-spect`,
  with `run.py` at c923248, at below-normal priority, right after the 19 SPECT tutorials on the same build (all passed).
- **Machine:** RTX 5090 (32 GB), Core Ultra 9 285K, 64 GB RAM, PyTorch 2.14.1+cu130.

## Against the RTX 5060 laptop on the same build

The laptop's results are in `2026-11-11-rtx-5060-laptop-gpu-rc2`. Times are the median of the repeats for a
reconstruction (its first run in brackets), else the single run.

| Case | Step | RTX 5090 (s) | RTX 5060 laptop (s) | Laptop / desktop | GPU 5090 / 5060 (GB) |
|---|---|---|---|---|---|
| spect_lu177/osem | OSEM 4 x 8 | 0.41 (1.44) | 1.82 (3.8) | 4.4x | 0.55 / 0.55 |
| spect_lu177/bsrem | BSREM 40 x 8 | 11.17 (12.18) | 33.29 (34.9) | 3.0x | 0.63 / 0.63 |
| spect_ac225_arf/osem | ARFs (generated) | 37.43 | 148.76 | 4.0x | 1.45 / 1.44 |
| spect_ac225_arf/osem | system matrix | 4.93 | 15.18 | 3.1x | 8.17 / 5.18 |
| spect_ac225_arf/osem | OSEM 100 x 3 | 60.30 (61.38) | 279.01 (281.1) | 4.6x | 1.39 / 1.39 |

- **GPU memory** is the same on both machines, except the Ac-225 system matrix step: 8.17 GB on the desktop against
  5.18 on the laptop. That is the device's used memory, which includes what PyTorch's caching allocator keeps; PyTorch's
  own peak there is 2.98 GB.
- **Peak RAM** is about 2 GB higher on the desktop for every step (for example 7.2 GB against 5.0 for OSEM 4 x 8): the
  process's committed memory, which includes the CUDA libraries loaded for this GPU.

No desktop run on rc1 exists to compare with. The SPECT tutorials, on the same build, took the same time as on rc1
(within the run-to-run variation of a few seconds), so the exact transpose costs no time here either.
