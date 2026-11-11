# RTX 5060 Laptop GPU, PyTomography 4.0.0rc2, 11 Nov 2026

All 13 cases passed in 35.8 min, on the same laptop as `2026-11-10-rtx-5060-laptop-gpu` (rc1), with the same
environment except PyTomography.

- **rc2 here is commit bef81841** ("Version 4.0.0rc2 (#447)" on `development`), not a PyPI release. On the evening of
  10 Nov, rc2 was not on PyPI (`pip index versions pytomography --pre` listed rc1 only) and there was no `v4.0.0rc2`
  tag. So the rc1 environment was cloned (`conda create --clone`) and only PyTomography was replaced:
  `pip install --no-deps --force-reinstall "git+https://github.com/PyTomography/PyTomography@bef81841"`, which builds
  as 4.0.0rc2. `pip list` differs from rc1's in that line only.
- **Command:** `python benchmarks/run.py --min-free-gb 2 --out benchmarks/results/2026-11-11-rtx-5060-laptop-gpu-rc2`,
  with `run.py` at c923248 as it is. The rc1 run needed a wrapper that dropped the RAM check; c923248's `CASES`
  made it unnecessary.
- **Data:** `PYTOMOGRAPHY_OUTPUT` was the rc2 tutorial run's round 2 folder, which held the PET event files
  (`caches_present` in the PET case files), as on rc1.
- **When:** 04:07 to 04:43 local time, after the rc2 tutorials. Windows power mode "Best performance", on AC power.

## rc2 against rc1

- **Reconstructions:** every step is within 8% of rc1, and most within 2%. rc2's change to SPECT back projection (#444)
  costs no time here. OSEM 4 x 8 takes 1.82 s against 1.78; BSREM 40 x 8 takes 33.3 s against 32.7; Ac-225 OSEM
  100 x 3 takes 279 s against 278.
- **GPU memory:** SPECT's reconstruction steps use 0.08 to 0.09 GB more on rc2. The one other memory change is
  OSEM 4 x 8's peak RAM, 0.3 GB higher. Nothing passes the 0.3 GB flag. Every CT and PET step's peaks are the same
  as on rc1.
- **Flagged steps** (a change of more than 10%):
  - **ct_helical_fbp's load (+265%),** with smaller rises in two other loads: the disk cache. On rc1 the smoke test
    had read the same files minutes before, and its own cold load of this case took 45.9 s. Here the case started
    cold and took 42.7 s.
  - **ct_microct/sart's search for the centre of rotation (-12%)** and **pct_abdomen/sart's starting FBP (-11%):**
    these steps run once, unrepeated. The same work in the sibling cases changed by -2% (ct_microct/fdk's search)
    and -8% (pct_abdomen/fbp's repeated FBP). CT and proton CT did not change between rc1 and rc2.

Steps under 1 s are left out of the flags: their noise is larger than 10%.

Times are the median of the repeats for a reconstruction (its first run in brackets), else the single run. RAM is the
process's peak committed memory. GPU is the larger of PyTorch's peak and the device's.

| Case | Step | rc2 (s) | rc1 (s) | Change | RAM rc2 / rc1 (GB) | GPU rc2 / rc1 (GB) | Flags |
|---|---|---|---|---|---|---|---|
| spect_lu177/osem | load | 0.78 | 0.87 | -10% | 3.8 / 3.8 | 0.08 / 0.08 |  |
| spect_lu177/osem | system matrix | 0.43 | 0.56 | -23% | 3.6 / 3.6 | 0.06 / 0.06 |  |
| spect_lu177/osem | OSEM 4 x 8 | 1.82 (3.8) | 1.78 (3.5) | +2% | 5.0 / 4.7 | 0.55 / 0.47 |  |
| spect_lu177/bsrem | load | 0.76 | 0.78 | -3% | 3.8 / 3.8 | 0.08 / 0.08 |  |
| spect_lu177/bsrem | system matrix | 0.42 | 0.43 | -2% | 3.6 / 3.6 | 0.06 / 0.06 |  |
| spect_lu177/bsrem | prior weights | 0.45 | 0.44 | +2% | 5.1 / 5.0 | 0.06 / 0.06 |  |
| spect_lu177/bsrem | BSREM 40 x 8 | 33.29 (34.9) | 32.73 (33.9) | +2% | 6.4 / 6.2 | 0.63 / 0.54 |  |
| spect_ac225_arf/osem | load | 1.88 | 1.31 | +44% | 3.7 / 3.7 | 0.13 / 0.13 | time +44% |
| spect_ac225_arf/osem | ARFs (generated) | 148.76 | 148.99 | -0% | 5.4 / 5.3 | 1.44 / 1.38 |  |
| spect_ac225_arf/osem | system matrix | 15.18 | 14.97 | +1% | 10.0 / 10.1 | 5.18 / 5.18 |  |
| spect_ac225_arf/osem | OSEM 100 x 3 | 279.01 (281.1) | 278.39 (280.1) | +0% | 6.3 / 6.1 | 1.39 / 1.30 |  |
| ct_helical_fbp/fbp | load | 42.72 | 11.69 | +265% | 7.2 / 7.2 | 1.79 / 1.79 | time +265% |
| ct_helical_fbp/fbp | system matrix | 0.00 | 0.00 | +50% | 5.6 / 5.6 | 0.00 / 0.00 |  |
| ct_helical_fbp/fbp | FBP (Hann) | 53.93 (54.2) | 53.97 (54.1) | -0% | 9.6 / 9.6 | 2.10 / 2.10 |  |
| ct_microct/fdk | load | 4.27 | 3.90 | +9% | 3.9 / 3.9 | 0.28 / 0.28 |  |
| ct_microct/fdk | centre of rotation (25 FDKs of 4 slices) | 16.55 | 16.92 | -2% | 4.7 / 4.7 | 1.04 / 1.04 |  |
| ct_microct/fdk | system matrix | 0.00 | 0.00 | -50% | 3.7 / 3.7 | 0.26 / 0.26 |  |
| ct_microct/fdk | FDK | 10.64 (10.7) | 10.74 (10.8) | -1% | 6.3 / 6.3 | 2.81 / 2.81 |  |
| ct_microct/sart | load | 3.73 | 3.87 | -4% | 3.9 / 3.9 | 0.28 / 0.28 |  |
| ct_microct/sart | centre of rotation (25 FDKs of 4 slices) | 17.76 | 20.18 | -12% | 4.7 / 4.7 | 1.04 / 1.04 | time -12% |
| ct_microct/sart | system matrix | 0.00 | 0.00 | +0% | 3.7 / 3.7 | 0.26 / 0.26 |  |
| ct_microct/sart | FDK (the starting image) | 10.77 | 10.79 | -0% | 6.3 / 6.3 | 2.81 / 2.81 |  |
| ct_microct/sart | OS-SART 3 x 5 | 14.71 (15.2) | 15.55 (15.9) | -5% | 10.1 / 10.1 | 6.06 / 6.06 |  |
| pct_abdomen/fbp | load | 0.95 | 0.83 | +14% | 4.7 / 4.7 | 0.00 / 0.00 |  |
| pct_abdomen/fbp | FBP, two passes | 12.61 (13.8) | 13.71 (15.2) | -8% | 6.8 / 6.8 | 1.16 / 1.16 |  |
| pct_abdomen/sart | load | 0.57 | 0.87 | -34% | 4.7 / 4.7 | 0.00 / 0.00 |  |
| pct_abdomen/sart | FBP, two passes (the starting image) | 13.90 | 15.64 | -11% | 6.8 / 6.8 | 1.16 / 1.16 | time -11% |
| pct_abdomen/sart | SART, 4 passes | 83.19 (83.1) | 83.99 (84.5) | -1% | 6.6 / 6.6 | 0.69 / 0.69 |  |
| pet_gate_listmode/osem | load (events, attenuation, normalization, randoms) | 10.88 | 8.59 | +27% | 11.1 / 11.1 | 0.02 / 0.02 | time +27% |
| pet_gate_listmode/osem | system matrix and additive term | 10.89 | 10.21 | +7% | 10.3 / 10.3 | 2.65 / 2.65 |  |
| pet_gate_listmode/osem | OSEM 4 x 14 | 7.69 (9.3) | 7.72 (9.5) | -0% | 11.2 / 11.2 | 2.83 / 2.83 |  |
| pet_gate_listmode/osem_tof | load (events, attenuation, normalization, randoms) | 10.00 | 9.60 | +4% | 11.4 / 11.4 | 0.02 / 0.02 |  |
| pet_gate_listmode/osem_tof | system matrix and additive term | 10.41 | 10.28 | +1% | 10.6 / 10.7 | 2.93 / 2.93 |  |
| pet_gate_listmode/osem_tof | OSEM 4 x 14 | 9.81 (11.3) | 9.81 (11.3) | -0% | 11.6 / 11.6 | 3.03 / 3.03 |  |
| pet_gate_sinogram/osem | load (events, attenuation, normalization, randoms) | 8.11 | 7.55 | +7% | 11.1 / 11.1 | 0.02 / 0.02 |  |
| pet_gate_sinogram/osem | bin the prompts into a sinogram | 3.54 | 3.77 | -6% | 9.2 / 9.2 | 0.01 / 0.01 |  |
| pet_gate_sinogram/osem | system matrix and additive term | 2.42 | 2.35 | +3% | 14.2 / 14.2 | 1.51 / 1.51 |  |
| pet_gate_sinogram/osem | OSEM 2 x 24 | 3.93 (6.8) | 3.97 (6.8) | -1% | 11.9 / 12.0 | 0.44 / 0.44 |  |
| pet_gate_sinogram/sss | load (events, attenuation, normalization, randoms) | 7.53 | 7.61 | -1% | 11.1 / 11.1 | 0.02 / 0.02 |  |
| pet_gate_sinogram/sss | bin the prompts into a sinogram | 3.65 | 3.68 | -1% | 9.2 / 9.2 | 0.01 / 0.01 |  |
| pet_gate_sinogram/sss | system matrix and additive term | 2.06 | 2.12 | -3% | 14.2 / 14.2 | 1.51 / 1.51 |  |
| pet_gate_sinogram/sss | OSEM 2 x 24 (the image the scatter is simulated from) | 6.50 | 6.64 | -2% | 12.0 / 12.0 | 0.44 / 0.44 |  |
| pet_gate_sinogram/sss | SSS (steps of 4) | 75.58 (76.4) | 75.60 (76.6) | -0% | 14.2 / 14.2 | 2.29 / 2.29 |  |
| pet_gate_sinogram/osem_tof | load (events, attenuation, normalization, randoms) | 7.70 | 7.84 | -2% | 11.4 / 11.4 | 0.02 / 0.02 |  |
| pet_gate_sinogram/osem_tof | bin the prompts into a sinogram | 10.60 | 10.96 | -3% | 8.9 / 8.9 | 0.01 / 0.01 |  |
| pet_gate_sinogram/osem_tof | system matrix and additive term | 2.35 | 2.51 | -6% | 15.9 / 15.9 | 3.71 / 3.71 |  |
| pet_gate_sinogram/osem_tof | OSEM 2 x 28 | 27.67 (29.6) | 27.59 (29.7) | +0% | 16.9 / 16.9 | 5.56 / 5.56 |  |

Flags:
- spect_ac225_arf/osem / load: time +44%
- ct_helical_fbp/fbp / load: time +265%
- ct_microct/sart / centre of rotation (25 FDKs of 4 slices): time -12%
- pct_abdomen/sart / FBP, two passes (the starting image): time -11%
- pet_gate_listmode/osem / load (events, attenuation, normalization, randoms): time +27%
