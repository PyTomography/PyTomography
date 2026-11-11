NVIDIA GeForce RTX 5090, Intel(R) Core(TM) Ultra 9 285K, 68.1 GB RAM; PyTomography 4.0.0rc2, PyTorch 2.14.1+cu130

| Case | Step | First run (s) | Repeats (s, median) | Peak RAM (GB) | Peak GPU (GB) |
|---|---|---|---|---|---|
| spect_lu177/osem | load | 0.64 |  | 5.9 | 0.18 |
| spect_lu177/osem | system matrix | 0.15 |  | 5.7 | 0.06 |
| spect_lu177/osem | OSEM 4 x 8 | 1.44 | 0.41 | 7.2 | 0.55 |
| spect_lu177/bsrem | load | 0.64 |  | 6.1 | 0.18 |
| spect_lu177/bsrem | system matrix | 0.15 |  | 6.0 | 0.06 |
| spect_lu177/bsrem | prior weights | 0.41 |  | 7.2 | 0.06 |
| spect_lu177/bsrem | BSREM 40 x 8 | 12.18 | 11.173 | 8.7 | 0.63 |
| spect_ac225_arf/osem | load | 0.96 |  | 5.7 | 0.22 |
| spect_ac225_arf/osem | ARFs (generated) | 37.43 |  | 7.7 | 1.45 |
| spect_ac225_arf/osem | system matrix | 4.93 |  | 15.3 | 8.17 |
| spect_ac225_arf/osem | OSEM 100 x 3 | 61.38 | 60.298 | 8.6 | 1.39 |
