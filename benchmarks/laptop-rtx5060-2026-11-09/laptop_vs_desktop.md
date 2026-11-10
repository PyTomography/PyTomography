Ran 34 of 34; passed 33.
Laptop total 273.4 min for the 34 run (desktop 158.0 min for the same set); passed only: laptop 261.1 min vs desktop 143.6 min = 1.82x.
Biggest slowdowns: t_siminddata 13.54x, t_fbp_ct 4.2x, t_ac225_dualpeak 4.09x, t_arf_psf 4.02x, t_ac225_dicom_recon 3.57x
Smallest: t_plotting 0.71x, t_uncertainty_pet 1.02x, t_dicommultibed 1.07x, t_CT_microct 1.15x, t_GE_HDF5 1.15x
SPECT: 19 passed, laptop 191.0 min vs desktop 98.3 min (1.94x)
Any: 2 passed, laptop 0.2 min vs desktop 0.2 min (0.99x)
CT: 3 passed, laptop 17.0 min vs desktop 7.5 min (2.27x)
Proton CT: 1 passed, laptop 1.9 min vs desktop 0.8 min (2.29x)
PET: 8 passed, laptop 51.0 min vs desktop 36.8 min (1.39x)

| Tutorial | Modality | Desktop | Laptop | Slower | Laptop RAM / GPU (GB) | Desktop RAM / GPU (GB) | Laptop result |
|---|---|---|---|---|---|---|---|
| t_siminddata | SPECT | 5 s | 68 s | 13.54× | 4.5 / 0.39 | 6.8 / 0.39 | passed |
| t_dicomdata | SPECT | 7 s | 24 s | 3.4× | 4.5 / 0.39 | 6.8 / 0.39 | passed |
| t_algorithms | SPECT | 41 s | 2.2 min | 3.18× | 6.7 / 1.58 | 9.0 / 1.58 | passed |
| t_accuracy_known_truth | SPECT | 17 s | 56 s | 3.31× | 4.7 / 0.42 | 6.9 / 0.42 | passed |
| t_dicommultibed | SPECT | 18 s | 19 s | 1.07× | 5.2 / 0.44 | 8.1 / 0.44 | passed |
| t_dualpeak | SPECT | 7 s | 17 s | 2.37× | 4.6 / 0.47 | 6.9 / 0.47 | passed |
| t_uncertainty_spect | SPECT | 18 s | 32 s | 1.77× | 7.4 / 0.71 | 10.2 / 0.71 | passed |
| t_starguide | SPECT | 16 s | 40 s | 2.52× | 5.6 / 1.1 | 7.9 / 1.1 | passed |
| t_ac225_simind_recon | SPECT | 1.6 min | 5.7 min | 3.54× | 9.6 / 2.11 | 14.0 / 3.08 | passed |
| t_CardiacReorientation | SPECT | 4 s | 8 s | 2.0× | 3.9 / 0.06 | 6.2 / 0.06 | passed |
| t_algorithms_measured | SPECT | 1.6 min | 5.2 min | 3.34× | 7.1 / 1.69 | 9.4 / 1.69 | passed |
| t_fbp_spect | SPECT | 6 s | 12 s | 2.02× | 4.7 / 0.43 | 7.1 / 0.43 | passed |
| t_arf_psf | SPECT | 2.5 min | 10.1 min | 4.02× | 9.6 / 2.8 | 10.5 / 1.98 | passed |
| t_ac225_dicom_recon | SPECT | 2.1 min | 7.6 min | 3.57× | 8.8 / 1.84 | 14.4 / 2.98 | passed |
| t_ac225_dualpeak | SPECT | 5.1 min | 20.9 min | 4.09× | 7.5 / 1.68 | 11.4 / 3.04 | passed |
| t_cgls | SPECT | 2.2 min | 7.8 min | 3.54× | 6.4 / 0.75 | 9.4 / 0.75 | passed |
| t_kernel_deep_kernel_spect_recon | SPECT | 3.2 min | 9.6 min | 2.96× | 9.1 / 1.44 | 11.7 / 1.44 | passed |
| t_examplesystemmatrix | Any | 4 s | 7 s | 1.68× | 2.7 / 0.0 | 4.4 / 0.0 | passed |
| t_plotting | Any | 10 s | 7 s | 0.71× | 3.0 / 0.0 | 5.8 / 0.0 | passed |
| t_CT_GEN3 | CT | 4.8 min | 8.4 min | 1.74× | 14.5 / 1.63 | 17.1 / 1.63 | passed |
| t_CT_microct | CT | 52 s | 60 s | 1.15× | 12.1 / 6.66 | 14.9 / 6.66 | passed |
| t_fbp_ct | CT | 1.8 min | 7.6 min | 4.2× | 11.9 / 1.3 | 14.5 / 1.3 | passed |
| t_pct_abdomen | Proton CT | 49 s | 1.9 min | 2.29× | 6.7 / 0.82 | 9.2 / 0.82 | passed |
| t_pet_introduction | PET | 6.2 min | 7.3 min | 1.17× | 16.4 / 2.91 | 18.8 / 2.98 | passed |
| t_GE_HDF5 | PET | 2.4 min | 2.8 min | 1.15× | 9.2 / 1.8 | 11.4 / 1.8 | passed |
| t_uncertainty_pet | PET | 2.0 min | 2.0 min | 1.02× | 10.8 / 2.07 | 13.1 / 2.07 | passed |
| t_PETSIRD | PET | 2.4 min | 4.4 min | 1.84× | 21.8 / 5.98 | 25.2 / 5.98 | passed |
| t_PETGATE_scat_sino | PET | 9.6 min | 15.7 min | 1.65× | 17.9 / 1.53 | 20.2 / 1.53 | passed |
| t_PETGATE_scat_sinoTOF | PET | 11.0 min | 13.5 min | 1.22× | 18.3 / 5.61 | 20.7 / 5.61 | passed |
| t_PETGATE_scat_lm | PET | 89 s | 2.7 min | 1.82× | 19.2 / 2.88 | 21.9 / 3.22 | passed |
| t_PETGATE_scat_lmTOF | PET | 1.7 min | 2.6 min | 1.58× | 19.4 / 3.44 | 22.6 / 3.9 | passed |
| t_PETGATE_DIP | PET | 14.4 min | 12.3 min |  | 19.0 /  | 22.2 / 8.02 | failed |
| t_spect_mc | SPECT | 41.0 min | 62.3 min | 1.52× | 4.6 / 0.46 | 7.0 / 0.46 | passed |
| t_spect_mc2 | SPECT | 36.6 min | 55.0 min | 1.5× | 4.8 / 0.46 | 7.1 / 0.46 | passed |
