NVIDIA GeForce RTX 5060 Laptop GPU, Intel(R) Core(TM) Ultra 9 275HX, 33.8 GB RAM; PyTomography 4.0.0rc1, PyTorch 2.14.1+cu130

| Case | Step | First run (s) | Repeats (s, median) | Peak RAM (GB) | Peak GPU (GB) |
|---|---|---|---|---|---|
| spect_lu177/osem | load | 0.87 |  | 3.8 | 0.08 |
| spect_lu177/osem | system matrix | 0.56 |  | 3.6 | 0.06 |
| spect_lu177/osem | OSEM 4 x 8 | 3.48 | 1.779 | 4.7 | 0.47 |
| spect_lu177/bsrem | load | 0.78 |  | 3.8 | 0.08 |
| spect_lu177/bsrem | system matrix | 0.43 |  | 3.6 | 0.06 |
| spect_lu177/bsrem | prior weights | 0.44 |  | 5.0 | 0.06 |
| spect_lu177/bsrem | BSREM 40 x 8 | 33.87 | 32.727 | 6.2 | 0.54 |
| spect_ac225_arf/osem | load | 1.31 |  | 3.7 | 0.13 |
| spect_ac225_arf/osem | ARFs (generated) | 148.99 |  | 5.3 | 1.38 |
| spect_ac225_arf/osem | system matrix | 14.97 |  | 10.1 | 5.18 |
| spect_ac225_arf/osem | OSEM 100 x 3 | 280.14 | 278.391 | 6.1 | 1.30 |
| ct_helical_fbp/fbp | load | 11.69 |  | 7.2 | 1.79 |
| ct_helical_fbp/fbp | system matrix | 0.00 |  | 5.6 | 0.00 |
| ct_helical_fbp/fbp | FBP (Hann) | 54.11 | 53.971 | 9.6 | 2.10 |
| ct_microct/fdk | load | 3.90 |  | 3.9 | 0.28 |
| ct_microct/fdk | centre of rotation (25 FDKs of 4 slices) | 16.92 |  | 4.7 | 1.04 |
| ct_microct/fdk | system matrix | 0.00 |  | 3.7 | 0.26 |
| ct_microct/fdk | FDK | 10.78 | 10.743 | 6.3 | 2.81 |
| ct_microct/sart | load | 3.87 |  | 3.9 | 0.28 |
| ct_microct/sart | centre of rotation (25 FDKs of 4 slices) | 20.18 |  | 4.7 | 1.04 |
| ct_microct/sart | system matrix | 0.00 |  | 3.7 | 0.26 |
| ct_microct/sart | FDK (the starting image) | 10.79 |  | 6.3 | 2.81 |
| ct_microct/sart | OS-SART 3 x 5 | 15.95 | 15.551 | 10.1 | 6.06 |
| pct_abdomen/fbp | load | 0.83 |  | 4.7 | 0.00 |
| pct_abdomen/fbp | FBP, two passes | 15.16 | 13.707 | 6.8 | 1.16 |
| pct_abdomen/sart | load | 0.87 |  | 4.7 | 0.00 |
| pct_abdomen/sart | FBP, two passes (the starting image) | 15.64 |  | 6.8 | 1.16 |
| pct_abdomen/sart | SART, 4 passes | 84.48 | 83.991 | 6.6 | 0.69 |
| pet_gate_listmode/osem | load (events, attenuation, normalization, randoms) | 8.59 |  | 11.1 | 0.02 |
| pet_gate_listmode/osem | system matrix and additive term | 10.21 |  | 10.3 | 2.65 |
| pet_gate_listmode/osem | OSEM 4 x 14 | 9.50 | 7.719 | 11.2 | 2.83 |
| pet_gate_listmode/osem_tof | load (events, attenuation, normalization, randoms) | 9.60 |  | 11.4 | 0.02 |
| pet_gate_listmode/osem_tof | system matrix and additive term | 10.28 |  | 10.7 | 2.93 |
| pet_gate_listmode/osem_tof | OSEM 4 x 14 | 11.32 | 9.814 | 11.6 | 3.03 |
| pet_gate_sinogram/osem | load (events, attenuation, normalization, randoms) | 7.55 |  | 11.1 | 0.02 |
| pet_gate_sinogram/osem | bin the prompts into a sinogram | 3.77 |  | 9.2 | 0.01 |
| pet_gate_sinogram/osem | system matrix and additive term | 2.35 |  | 14.2 | 1.51 |
| pet_gate_sinogram/osem | OSEM 2 x 24 | 6.83 | 3.972 | 12.0 | 0.44 |
| pet_gate_sinogram/sss | load (events, attenuation, normalization, randoms) | 7.61 |  | 11.1 | 0.02 |
| pet_gate_sinogram/sss | bin the prompts into a sinogram | 3.68 |  | 9.2 | 0.01 |
| pet_gate_sinogram/sss | system matrix and additive term | 2.12 |  | 14.2 | 1.51 |
| pet_gate_sinogram/sss | OSEM 2 x 24 (the image the scatter is simulated from) | 6.64 |  | 12.0 | 0.44 |
| pet_gate_sinogram/sss | SSS (steps of 4) | 76.56 | 75.599 | 14.2 | 2.29 |
| pet_gate_sinogram/osem_tof | load (events, attenuation, normalization, randoms) | 7.84 |  | 11.4 | 0.02 |
| pet_gate_sinogram/osem_tof | bin the prompts into a sinogram | 10.96 |  | 8.9 | 0.01 |
| pet_gate_sinogram/osem_tof | system matrix and additive term | 2.51 |  | 15.9 | 3.71 |
| pet_gate_sinogram/osem_tof | OSEM 2 x 28 | 29.66 | 27.59 | 16.9 | 5.56 |
