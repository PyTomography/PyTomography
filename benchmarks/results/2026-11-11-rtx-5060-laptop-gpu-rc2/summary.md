NVIDIA GeForce RTX 5060 Laptop GPU, Intel(R) Core(TM) Ultra 9 275HX, 33.8 GB RAM; PyTomography 4.0.0rc2, PyTorch 2.14.1+cu130

| Case | Step | First run (s) | Repeats (s, median) | Peak RAM (GB) | Peak GPU (GB) |
|---|---|---|---|---|---|
| spect_lu177/osem | load | 0.78 |  | 3.8 | 0.08 |
| spect_lu177/osem | system matrix | 0.43 |  | 3.6 | 0.06 |
| spect_lu177/osem | OSEM 4 x 8 | 3.77 | 1.82 | 5.0 | 0.55 |
| spect_lu177/bsrem | load | 0.76 |  | 3.8 | 0.08 |
| spect_lu177/bsrem | system matrix | 0.42 |  | 3.6 | 0.06 |
| spect_lu177/bsrem | prior weights | 0.45 |  | 5.1 | 0.06 |
| spect_lu177/bsrem | BSREM 40 x 8 | 34.88 | 33.289 | 6.4 | 0.63 |
| spect_ac225_arf/osem | load | 1.88 |  | 3.7 | 0.13 |
| spect_ac225_arf/osem | ARFs (generated) | 148.76 |  | 5.4 | 1.44 |
| spect_ac225_arf/osem | system matrix | 15.18 |  | 10.0 | 5.18 |
| spect_ac225_arf/osem | OSEM 100 x 3 | 281.11 | 279.01 | 6.3 | 1.39 |
| ct_helical_fbp/fbp | load | 42.72 |  | 7.2 | 1.79 |
| ct_helical_fbp/fbp | system matrix | 0.00 |  | 5.6 | 0.00 |
| ct_helical_fbp/fbp | FBP (Hann) | 54.16 | 53.931 | 9.6 | 2.10 |
| ct_microct/fdk | load | 4.27 |  | 3.9 | 0.28 |
| ct_microct/fdk | centre of rotation (25 FDKs of 4 slices) | 16.55 |  | 4.7 | 1.04 |
| ct_microct/fdk | system matrix | 0.00 |  | 3.7 | 0.26 |
| ct_microct/fdk | FDK | 10.72 | 10.643 | 6.3 | 2.81 |
| ct_microct/sart | load | 3.73 |  | 3.9 | 0.28 |
| ct_microct/sart | centre of rotation (25 FDKs of 4 slices) | 17.76 |  | 4.7 | 1.04 |
| ct_microct/sart | system matrix | 0.00 |  | 3.7 | 0.26 |
| ct_microct/sart | FDK (the starting image) | 10.77 |  | 6.3 | 2.81 |
| ct_microct/sart | OS-SART 3 x 5 | 15.25 | 14.71 | 10.1 | 6.06 |
| pct_abdomen/fbp | load | 0.95 |  | 4.7 | 0.00 |
| pct_abdomen/fbp | FBP, two passes | 13.76 | 12.608 | 6.8 | 1.16 |
| pct_abdomen/sart | load | 0.57 |  | 4.7 | 0.00 |
| pct_abdomen/sart | FBP, two passes (the starting image) | 13.90 |  | 6.8 | 1.16 |
| pct_abdomen/sart | SART, 4 passes | 83.09 | 83.191 | 6.6 | 0.69 |
| pet_gate_listmode/osem | load (events, attenuation, normalization, randoms) | 10.88 |  | 11.1 | 0.02 |
| pet_gate_listmode/osem | system matrix and additive term | 10.89 |  | 10.3 | 2.65 |
| pet_gate_listmode/osem | OSEM 4 x 14 | 9.35 | 7.69 | 11.2 | 2.83 |
| pet_gate_listmode/osem_tof | load (events, attenuation, normalization, randoms) | 10.00 |  | 11.4 | 0.02 |
| pet_gate_listmode/osem_tof | system matrix and additive term | 10.41 |  | 10.6 | 2.93 |
| pet_gate_listmode/osem_tof | OSEM 4 x 14 | 11.26 | 9.806 | 11.6 | 3.03 |
| pet_gate_sinogram/osem | load (events, attenuation, normalization, randoms) | 8.11 |  | 11.1 | 0.02 |
| pet_gate_sinogram/osem | bin the prompts into a sinogram | 3.54 |  | 9.2 | 0.01 |
| pet_gate_sinogram/osem | system matrix and additive term | 2.42 |  | 14.2 | 1.51 |
| pet_gate_sinogram/osem | OSEM 2 x 24 | 6.77 | 3.933 | 11.9 | 0.44 |
| pet_gate_sinogram/sss | load (events, attenuation, normalization, randoms) | 7.53 |  | 11.1 | 0.02 |
| pet_gate_sinogram/sss | bin the prompts into a sinogram | 3.65 |  | 9.2 | 0.01 |
| pet_gate_sinogram/sss | system matrix and additive term | 2.06 |  | 14.2 | 1.51 |
| pet_gate_sinogram/sss | OSEM 2 x 24 (the image the scatter is simulated from) | 6.50 |  | 12.0 | 0.44 |
| pet_gate_sinogram/sss | SSS (steps of 4) | 76.42 | 75.582 | 14.2 | 2.29 |
| pet_gate_sinogram/osem_tof | load (events, attenuation, normalization, randoms) | 7.70 |  | 11.4 | 0.02 |
| pet_gate_sinogram/osem_tof | bin the prompts into a sinogram | 10.60 |  | 8.9 | 0.01 |
| pet_gate_sinogram/osem_tof | system matrix and additive term | 2.35 |  | 15.9 | 3.71 |
| pet_gate_sinogram/osem_tof | OSEM 2 x 28 | 29.64 | 27.67 | 16.9 | 5.56 |
