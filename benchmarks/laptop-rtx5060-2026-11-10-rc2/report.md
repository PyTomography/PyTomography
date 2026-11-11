| Tutorial | Status | Time (s) | Peak GPU (GB) | Peak RAM (GB) | Error |
|---|---|---|---|---|---|
| t_siminddata | passed | 19.7 | 0.48 | 4.7 |  |
| t_dicomdata | passed | 11.5 | 0.47 | 5.0 |  |
| t_algorithms | passed | 123.4 | 1.66 | 7.5 |  |
| t_accuracy_known_truth | passed | 56.5 | 0.51 | 4.9 |  |
| t_dicommultibed | passed | 20.4 | 0.49 | 6.1 |  |
| t_dualpeak | passed | 17.0 | 0.55 | 4.9 |  |
| t_uncertainty_spect | passed | 31.4 | 0.76 | 8.2 |  |
| t_starguide | passed | 40.2 | 1.49 | 6.1 |  |
| t_ac225_simind_recon | passed | 449.5 | 2.11 | 9.7 |  |
| t_CardiacReorientation | passed | 7.8 | 0.07 | 4.0 |  |
| t_algorithms_measured | passed | 303.1 | 1.77 | 7.6 |  |
| t_fbp_spect | passed | 11.3 | 0.53 | 5.1 |  |
| t_arf_psf | passed | 605.9 | 2.8 | 10.2 |  |
| t_ac225_dicom_recon | passed | 453.9 | 1.84 | 8.9 |  |
| t_ac225_dualpeak | passed | 1256.6 | 1.68 | 8.0 |  |
| t_cgls | passed | 474.5 | 0.8 | 7.1 |  |
| t_kernel_deep_kernel_spect_recon | passed | 560.5 | 1.55 | 9.3 |  |
| t_examplesystemmatrix | passed | 6.5 | 0.0 | 2.7 |  |
| t_plotting | passed | 6.9 | 0.0 | 3.4 |  |
| t_CT_GEN3 | passed | 459.8 | 1.63 | 14.5 |  |
| t_CT_microct | passed | 56.0 | 6.66 | 13.1 |  |
| t_fbp_ct | passed | 441.4 | 1.3 | 11.9 |  |
| t_pct_abdomen | passed | 107.1 | 0.82 | 7.0 |  |
| t_pet_introduction | passed | 418.5 | 2.91 | 16.4 |  |
| t_GE_HDF5 | passed | 165.9 | 1.8 | 9.3 |  |
| t_uncertainty_pet | passed | 120.7 | 2.07 | 10.8 |  |
| t_PETSIRD | passed | 260.9 | 5.98 | 21.9 |  |
| t_PETGATE_scat_sino | passed | 881.5 | 1.53 | 17.8 |  |
| t_PETGATE_scat_sinoTOF | passed | 775.2 | 5.61 | 18.4 |  |
| t_PETGATE_scat_lm | passed | 160.7 | 2.88 | 19.2 |  |
| t_PETGATE_scat_lmTOF | passed | 156.9 | 3.44 | 19.5 |  |
| t_PETGATE_DIP | failed | 703.2 |  | 19.0 | cell 22: OutOfMemoryError: CUDA out of memory. Tried to allocate 226.00 MiB. GPU 0 has a total capacity of 7.93 GiB of which 0 bytes is free. Including non-PyTorch memory, this process has 17179869184.00 GiB memory in use. Of the allocated memory 6.70 GiB is allocated by PyTorch, and 261.03 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLO |
| t_spect_mc | passed | 3318.3 | 0.58 | 4.9 |  |
| t_spect_mc2 | passed | 2963.9 | 0.53 | 4.9 |  |
| t_PETGATE_DIP_c8 | passed | 1912.1 | 6.37 | 19.1 |  |
